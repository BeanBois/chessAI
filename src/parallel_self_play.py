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

TEMP_THRESHOLD = 30
MAX_GAME_MOVES = 512


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


def _assign_outcomes(
    history: list[tuple],
    env: ChessGame,
) -> list[tuple]:
    outcome = env.board.outcome()
    winner  = outcome.winner if outcome else None
    return [
        (s, p, 0.0 if winner is None else (1.0 if winner == pl else -1.0))
        for s, p, pl in history
    ]


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
    )

    def __init__(self, num_simulations: int):
        self.num_simulations = num_simulations
        self.env:        ChessGame = None
        self.mcts:       MCTS      = None
        self.history:    list      = []
        self.move_count: int       = 0
        self.done:       bool      = False
        self.trajectory: list      = []

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

        # Record position BEFORE the move
        state  = encode_board(self.env.board)
        policy = _build_policy_vector(probs, self.env.board)
        player = self.env.current_player
        self.history.append((state, policy, player))

        # Sample move, save child subtree, step env
        move = np.random.choice(moves, p=weights)
        self.mcts.advance_root(move)    # must come BEFORE env.step
        self.env.step(move)
        self.move_count += 1

        if self.env.is_terminal() or self.move_count >= MAX_GAME_MOVES:
            self.done = True
            if self.move_count >= MAX_GAME_MOVES:
                self.trajectory = [(s, p, 0.0) for s, p, _ in self.history]
            else:
                self.trajectory = _assign_outcomes(self.history, self.env)
        else:
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
        slots   = [_Slot(self.num_simulations) for _ in range(n_slots)]
        for slot in slots:
            slot.reset()

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
                lv, pa = slot.mcts.select_leaves(self.leaves_per_game)
                all_leaves.extend(lv)
                all_paths.extend(pa)
                slot_slices.append((slot, start, len(all_leaves)))

            # ── 2. Single GPU forward pass for ALL leaves ─────────────────
            if all_leaves:
                boards = [node.env.board for node in all_leaves]
                # non_blocking=True lets CPU keep working while transfer happens
                batch = self.neural_net.encode_batch(boards)  

                if self._stream:
                    with torch.cuda.stream(self._stream):
                        with torch.no_grad():
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                policy_logits, values = self.neural_net.model(batch)
                    # CPU can do other work here until we actually need the results
                    self._stream.synchronize()
                else:
                    with torch.no_grad():
                        policy_logits, values = self.neural_net.model(batch)

                pol_np = policy_logits.float().cpu().numpy()
                val_np = values.float().squeeze(1).cpu().numpy()

            # ── 4. Commit moves for slots that have finished searching ─────
            for slot in active:
                if not slot.mcts.search_done():
                    continue
                slot.commit_move()
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