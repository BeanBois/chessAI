import os
import chess
import numpy as np
import multiprocessing as mp
from .env import ChessGame
from .cnn_model.board_encoder import encode_board
from .cnn_model.move_encoder import TOTAL_ACTIONS, move_to_index
from .mcts import MCTS

# Temperature schedule: τ=1 for opening moves, greedy after that
TEMP_THRESHOLD = 30    # ply at which we switch to greedy
MAX_GAME_MOVES = 512   # hard cap — adjudicated as draw


# ------------------------------------------------------------------
# Worker — must be a top-level function for multiprocessing to pickle it
# ------------------------------------------------------------------

def _play_game_worker(packed: tuple) -> list[tuple]:
    """
    Runs in a separate process.  Reconstructs the network from a plain
    state-dict (safe to pickle, works with CUDA parent process) and plays
    one complete game on CPU.
    """
    state_dict, num_simulations = packed

    # Import inside the worker to avoid pickling torch objects
    from src.cnn_model.network import ChessNet, NeuralNetwork
    from src.self_play import _play_one_game

    model = ChessNet()
    model.load_state_dict(state_dict)
    net = NeuralNetwork(model=model, device="cpu")   # workers always use CPU
    return _play_one_game(net, num_simulations)


def _play_one_game(neural_net, num_simulations: int) -> list[tuple]:
    """
    Standalone game loop — called by both the worker and SelfPlay.play_game()
    so there's a single source of truth.
    """
    env = ChessGame()
    env.reset()

    mcts = MCTS(neural_net=neural_net, num_simulations=num_simulations, temperature=1.0)
    history: list[tuple[np.ndarray, dict, chess.Color]] = []

    move_count = 0
    while not env.is_terminal() and move_count < MAX_GAME_MOVES:

        mcts.temperature = 1.0 if move_count < TEMP_THRESHOLD else 0.0

        action_probs = mcts.get_action_probs(env)

        # ── removed gc.collect() ─────────────────────────────────────────
        # Python already frees the old tree the moment the root goes out of
        # scope (refcount → 0).  Calling gc.collect() here triggers a full
        # generational sweep every single ply — ~60× per game — with zero
        # benefit because there are no reference cycles in the tree.

        state  = encode_board(env.board)
        policy = _build_policy_vector(action_probs, env.board)
        player = env.current_player
        history.append((state, policy, player))

        # Sample move
        moves  = list(action_probs.keys())
        probs  = np.array(list(action_probs.values()), dtype=np.float32)
        probs /= probs.sum()
        move   = np.random.choice(moves, p=probs)

        # ── Tree reuse ────────────────────────────────────────────────────
        # Tell MCTS which child will be the root BEFORE we step the env.
        # advance_root() materialises the child's env while the parent chain
        # is still intact, then detaches it so the rest of the tree is GC'd.
        mcts.advance_root(move)

        env.step(move)
        move_count += 1

    if move_count >= MAX_GAME_MOVES:
        return [(s, p, 0.0) for s, p, _ in history]

    return _assign_outcomes(history, env)


# ------------------------------------------------------------------
# Helpers (module-level so workers can import them cheaply)
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
    # Use env.get_result() so fivefold-repetition and 75-move-rule games
    # are correctly identified as draws (0.0) rather than falling through
    # as None when env.board.outcome() returns None for those terminal types.
    return [(s, p, env.get_result(pl)) for s, p, pl in history]


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------

class SelfPlay:
    """
    Generates self-play games using the current best network + MCTS.

    generate() can run games in parallel across CPU cores.  Each worker
    gets its own copy of the weights on CPU, so GPU memory is not wasted
    during data generation (the GPU is free for training).
    """

    def __init__(self, neural_net, num_simulations: int = 800):
        self.neural_net      = neural_net
        self.num_simulations = num_simulations

    def play_game(self) -> list[tuple]:
        """Play one game in the current process (used for quick tests)."""
        return _play_one_game(self.neural_net, self.num_simulations)

    def generate(
        self,
        num_games: int,
        num_workers: int = 0,   # 0 = auto (os.cpu_count()); 1 = no multiprocessing
    ) -> list[tuple]:
        """
        Generate `num_games` games, optionally in parallel.

        Workers are spawned fresh each call so stale weights are never reused.
        Spawn (not fork) is used so CUDA contexts in the parent process don't
        leak into workers.
        """
        workers = num_workers or os.cpu_count() or 1

        if workers == 1 or num_games == 1:
            # Single-process path — easier to debug and profile
            all_positions = []
            for idx in range(num_games):
                traj = _play_one_game(self.neural_net, self.num_simulations)
                all_positions.extend(traj)
                print(f"  Game {idx+1}/{num_games} — {len(traj)} positions")
            return all_positions

        # Parallel path
        # Pass the state-dict, not the model — state-dicts are plain dicts of
        # tensors and pickle cleanly across spawn boundaries even when the
        # parent loaded the model on CUDA.
        state_dict = {k: v.cpu() for k, v in self.neural_net.model.state_dict().items()}
        args = [(state_dict, self.num_simulations)] * num_games

        ctx = mp.get_context("spawn")   # safe with CUDA parent
        all_positions = []

        with ctx.Pool(processes=workers) as pool:
            for idx, traj in enumerate(pool.imap_unordered(_play_game_worker, args)):
                all_positions.extend(traj)
                print(f"  Game {idx+1}/{num_games} — {len(traj)} positions")

        return all_positions