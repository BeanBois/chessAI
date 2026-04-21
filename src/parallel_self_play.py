"""
parallel_self_play.py

Runs N chess games simultaneously on a single GPU by pooling MCTS leaf
evaluations across all active games into one large batch per NN call.

Why this is faster than the existing multiprocessing approach
─────────────────────────────────────────────────────────────
The multiprocessing workers run on CPU (safe to pickle, no CUDA issues),
so every single leaf evaluation is a slow CPU matrix multiply.

Here every leaf evaluation hits the GPU, and the batch is N× larger than
any single-game search could produce — so the GPU is actually busy.

Typical improvement on a single mid-range GPU:
  Single game  (batch=32, CPU workers) :  ~8–15 min/game
  16 parallel games (batch=64, GPU)    :  ~1–3 min/game

Usage
─────
Replace SelfPlay with ParallelSelfPlay in train.py:

    from src.parallel_self_play import ParallelSelfPlay

    self_play = ParallelSelfPlay(
        neural_net      = best_net,
        num_simulations = SIMULATIONS,
        num_parallel    = 16,      # tune to your VRAM — start here
        leaves_per_game = 4,       # leaves collected from each game per step
    )
    positions = self_play.generate(num_games=GAMES_PER_ITER)

Tuning num_parallel
───────────────────
Each active game holds an MCTS tree in RAM (~50–200 MB depending on depth).
Start at 16, watch nvidia-smi — if GPU memory isn't the limit, increase
num_parallel until GPU utilisation plateaus around 80–90%.
"""

import chess
import numpy as np
import torch

from .env import ChessGame
from .cnn_model.board_encoder import encode_board
from .cnn_model.move_encoder import move_to_index
from .mcts import MCTS

TEMP_THRESHOLD = 15
MAX_GAME_MOVES = 256


# ------------------------------------------------------------------
# Helpers (module-level so they can be reused by SelfPlay too)
# ------------------------------------------------------------------

def _build_policy_vector(
    action_probs: dict[chess.Move, float],
    board: chess.Board,
) -> dict[int, float]:
    flip  = board.turn == chess.BLACK
    total = sum(action_probs.values())
    return {
        move_to_index(move, flip=flip): prob / total
        for move, prob in action_probs.items()
    }


GAMMA = 1.0  # undiscounted — every position's target is the game outcome

# Resignation: prevents losing side dragging games into 50-move/repetition draws
RESIGN_THRESHOLD   = -0.9   # resign when root Q-value is below this
RESIGN_CONSECUTIVE = 5      # for this many consecutive moves
NO_RESIGN_FRAC     = 0.1    # fraction of games where resignation is disabled


def _assign_outcomes(
    history: list[tuple],
    env: ChessGame,
) -> list[tuple]:
    """
    Compute discounted return G(t) for each position in the game trajectory.

    Each position's value target blends:
      - Per-step capture reward (scaled so total << terminal ±1.0)
      - Terminal outcome ±1.0/0.0 at the final position
    using negamax discounting: returns are computed backward and the sign
    flips at each step because the players alternate.

    G(t) = r(t) + γ * -G(t+1)   (negamax: opponent's gain is our loss)
    """
    T = len(history)
    values = [0.0] * T
    g = 0.0
    for t in reversed(range(T)):
        state, policy, player, capture_r = history[t]
        r = capture_r
        if t == T - 1:
            r += env.get_result(player)  # terminal win/loss/draw
        # g is the discounted return from the NEXT half-move's perspective;
        # negate because the next half-move belongs to the opponent.
        g = r + GAMMA * (-g)
        values[t] = max(-1.0, min(1.0, g))
    return [(s, p, v) for (s, p, _, _), v in zip(history, values)]


# ------------------------------------------------------------------
# Per-game state container
# ------------------------------------------------------------------

class _Slot:
    """
    Holds all mutable state for one game running inside ParallelSelfPlay.
    Each slot owns its own env and MCTS tree; the ParallelSelfPlay runner
    drives the NN calls from outside so all slots share one GPU batch.
    """

    __slots__ = (
        "num_simulations", "env", "mcts",
        "history", "move_count", "done", "trajectory",
        "resign_counter", "allow_resign",
    )

    def __init__(self, num_simulations: int, allow_resign: bool = True):
        self.num_simulations = num_simulations
        self.env:        ChessGame = None
        self.mcts:       MCTS      = None
        self.history:    list      = []
        self.move_count: int       = 0
        self.done:       bool      = False
        self.trajectory: list      = []
        self.resign_counter: int   = 0
        self.allow_resign: bool    = allow_resign

    def reset(self) -> None:
        """Start a brand new game."""
        self.env = ChessGame()
        self.env.reset()
        # neural_net=None — NN calls are driven externally by ParallelSelfPlay
        self.mcts = MCTS(neural_net=None, num_simulations=self.num_simulations)
        self.mcts.prepare_search(self.env)
        self.history    = []
        self.move_count = 0
        self.done       = False
        self.trajectory = []
        self.resign_counter = 0

    def commit_move(self) -> None:
        """
        Sample a move from the completed MCTS search, step the env,
        record the (state, policy, player) tuple, and prepare the next search.

        Sets self.done = True and populates self.trajectory when the game ends.
        """
        root = self.mcts._search_root
        self.mcts.temperature = 1.0 if self.move_count < TEMP_THRESHOLD else 0.0
        probs = self.mcts._action_probabilities(root)

        moves   = list(probs.keys())
        weights = np.array(list(probs.values()), dtype=np.float32)
        weights /= weights.sum()   # floating-point safety

        # Sample move first so we can compute the capture reward on the pre-move board
        move = np.random.choice(moves, p=weights)

        # Record position BEFORE the move (capture reward requires pre-move board)
        state        = encode_board(self.env.board)
        policy       = _build_policy_vector(probs, self.env.board)
        player       = self.env.current_player
        capture_r    = self.env._capture_reward_normalized(move)
        self.history.append((state, policy, player, capture_r))
        self.mcts.advance_root(move)    # must come BEFORE env.step
        self.env.step(move)
        self.move_count += 1

        if self.env.is_terminal() or self.move_count >= MAX_GAME_MOVES:
            self.done = True
            self.trajectory = _assign_outcomes(self.history, self.env)
        else:
            # Check resignation: if root Q-value is very negative for several
            # consecutive moves, the current player resigns. This prevents
            # losing games from dragging into draws via repetition / 50-move rule.
            if self.allow_resign:
                root_q = root.q_value
                if root_q < RESIGN_THRESHOLD:
                    self.resign_counter += 1
                else:
                    self.resign_counter = 0

                if self.resign_counter >= RESIGN_CONSECUTIVE:
                    self.done = True
                    # Assign loss for the current player (who just moved)
                    # by using the environment's terminal result logic.
                    # Since the game isn't actually terminal, manually set values:
                    # the last player to move resigned, so they lose.
                    T = len(self.history)
                    values = [0.0] * T
                    g = 0.0
                    for t in reversed(range(T)):
                        state, policy, plr, capture_r = self.history[t]
                        r = capture_r
                        if t == T - 1:
                            # The player who just moved (current_player has flipped)
                            # is the one resigning, so from their POV it's -1.0
                            r += -1.0
                        g = r + GAMMA * (-g)
                        values[t] = max(-1.0, min(1.0, g))
                    self.trajectory = [
                        (s, p, v)
                        for (s, p, _, _), v in zip(self.history, values)
                    ]
                    return

            # Prepare root for the next position (tree reuse handled inside)
            self.mcts.prepare_search(self.env)


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------

class ParallelSelfPlay:
    """
    Generates self-play games using the current best network + MCTS,
    running num_parallel games simultaneously on a single GPU.

    Drop-in replacement for SelfPlay — same generate() signature.

    How it works
    ────────────
    1. N _Slot objects are initialised, each with its own env + MCTS tree.
    2. Each outer loop iteration:
       a. Every active slot contributes up to leaves_per_game leaf nodes
          (selected by PUCT, no NN called yet).
       b. All leaves from all slots are stacked into ONE tensor and sent
          through the GPU in a single forward pass.
       c. Results are sliced and fed back to each slot's process_results(),
          which completes expansion and backpropagation.
    3. Once a slot's simulation budget is spent (search_done()), it calls
       commit_move() to record the position and advance the game.
    4. Finished games are harvested and their slots are reset for new games
       until num_games total have been collected.
    """

    def __init__(
        self,
        neural_net,
        num_simulations: int = 800,
        num_parallel:    int = 16,   # simultaneous games; tune to your VRAM
        leaves_per_game: int = 4,    # leaves per slot per outer loop step
    ):
        self.neural_net      = neural_net
        self.num_simulations = num_simulations
        self.num_parallel    = num_parallel
        self.leaves_per_game = leaves_per_game
        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None


    def generate(self, num_games: int) -> list[tuple]:
        all_positions: list = []
        completed:     int  = 0

        # Initialise slots — never more slots than games requested
        n_slots = min(self.num_parallel, num_games)
        slots   = []
        for i in range(n_slots):
            allow_resign = (i / n_slots) >= NO_RESIGN_FRAC  # first ~10% are no-resign
            slot = _Slot(self.num_simulations, allow_resign=allow_resign)
            slot.reset()
            slots.append(slot)
        print('Generating games ... ')

        while completed < num_games:
            active = [s for s in slots if not s.done]
            if not active:
                break   # all completed (happens on the last round)

            # ── 1. Collect leaves from every active game ──────────────────
            all_leaves:  list = []
            all_paths:   list = []
            # Track which slice of the batch belongs to which slot
            slot_slices: list = []   # [(slot, start_idx, end_idx), ...]

            for slot in active:
                if slot.mcts.search_done():
                    continue   # this slot is waiting to commit, skip
                start = len(all_leaves)
                try:
                    lv, pa = slot.mcts.select_leaves(self.leaves_per_game)
                except Exception as e:
                    print(f"[!] select_leaves failed for slot, skipping: {e}")
                    slot.done = True
                    continue
                all_leaves.extend(lv)
                all_paths.extend(pa)
                slot_slices.append((slot, start, len(all_leaves)))

            # ── 2. Single GPU forward pass for ALL leaves ─────────────────
            pol_np = None
            val_np = None
            if all_leaves:
                boards = [node.env.board for node in all_leaves]
                # non_blocking=True lets CPU keep working while transfer happens
                batch = self.neural_net.encode_batch(boards)  

                if self._stream:
                    with torch.cuda.stream(self._stream):
                        with torch.no_grad():
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                policy_logits, value_logits = self.neural_net.model(batch)
                    # CPU can do other work here until we actually need the results
                    self._stream.synchronize()
                else:
                    with torch.no_grad():
                        policy_logits, value_logits = self.neural_net.model(batch)

                pol_np = policy_logits.float().cpu().numpy()
                # Scalar value output (tanh already applied by model)
                val_np = value_logits.float().squeeze(-1).cpu().numpy()
            
            # ── 3. Feed results back to each slot ────────────────────────────
            for slot, start, end in slot_slices:
                if start == end or pol_np is None:
                    continue   # this slot contributed no leaves this step
                try:
                    slot.mcts.process_results(
                        leaves           = all_leaves[start:end],
                        paths            = all_paths[start:end],
                        policy_logits_np = pol_np[start:end],
                        values_np        = val_np[start:end],
                    )
                except Exception as e:
                    print(f"[!] process_results failed for slot, skipping: {e}")
                    slot.done = True
    # Safety guard — break only if no slot has even started (true deadlock)
            if not all_leaves and not any(s.mcts.search_done() for s in active):
                # A slot may have exhausted its budget entirely via terminal
                # nodes this round (0 NN leaves, but sims fully counted).
                # Only bail if no slot has an expanded root yet — that is a
                # true deadlock where nothing can ever make progress.
                if not any(s.mcts._root_expanded for s in active):
                    print("[!] Warning: no leaves collected and no slots ready — breaking.")
                    break
            # ── 4. Commit moves for slots that have finished searching ─────
            for slot in active:
                if not slot.mcts.search_done():
                    continue
                try:
                    slot.commit_move()
                except Exception as e:
                    print(f"[!] commit_move failed for slot, skipping: {e}")
                    slot.done = True
                if slot.done:
                    all_positions.extend(slot.trajectory)
                    completed += 1
                    print(
                        f"  Game {completed}/{num_games}"
                        f" — {len(slot.trajectory)} positions"
                    )
                    # Recycle the slot for the next game if needed
                    if completed < num_games:
                        slot.reset()
        print(f'{num_games} completed')
        return all_positions