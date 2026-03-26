import chess
import numpy as np
from functools import lru_cache

# AlphaZero move encoding: 73 planes × 64 squares = 4672
#
# Per source square, 73 planes:
#   0–55  : queen-like moves  (8 directions × 7 distances)
#   56–63 : knight moves      (8 offsets)
#   64–72 : underpromotions   (3 piece types × 3 directions)
#             pieces: knight=0, bishop=1, rook=2
#             dirs:   left diagonal=0, straight=1, right diagonal=2

NUM_PLANES    = 73
TOTAL_ACTIONS = 64 * NUM_PLANES  # 4672

# Queen direction: (delta_file, delta_rank)
QUEEN_DIRS = [
    (0,  1),   # N
    (1,  1),   # NE
    (1,  0),   # E
    (1, -1),   # SE
    (0, -1),   # S
    (-1,-1),   # SW
    (-1, 0),   # W
    (-1, 1),   # NW
]

# Knight offsets: (delta_file, delta_rank)
KNIGHT_OFFSETS = [
    (1,  2), (2,  1), (2, -1), (1, -2),
    (-1,-2), (-2,-1), (-2, 1), (-1, 2),
]

# Underpromotion piece order (queen promotion is handled by queen-plane)
UNDER_PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# Underpromotion directions (file delta from pawn's perspective)
UNDER_PROMO_DIRS = [-1, 0, 1]  # left capture, straight, right capture


@lru_cache(maxsize=None)
def _build_move_to_index() -> dict:
    """Build move → index mapping for all possible chess moves."""
    move_to_idx = {}

    for from_sq in range(64):
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)

        # Queen-like moves
        for dir_idx, (df, dr) in enumerate(QUEEN_DIRS):
            for dist in range(1, 8):
                to_file = from_file + df * dist
                to_rank = from_rank + dr * dist
                if not (0 <= to_file < 8 and 0 <= to_rank < 8):
                    break
                to_sq = chess.square(to_file, to_rank)
                plane = dir_idx * 7 + (dist - 1)
                idx = from_sq * NUM_PLANES + plane
                move = chess.Move(from_sq, to_sq)
                move_to_idx[move] = idx
                # Queen promotion (default — no explicit promo piece needed)
                if from_rank == 6 and to_rank == 7:
                    move_to_idx[chess.Move(from_sq, to_sq, chess.QUEEN)] = idx

        # Knight moves
        for knight_idx, (df, dr) in enumerate(KNIGHT_OFFSETS):
            to_file = from_file + df
            to_rank = from_rank + dr
            if not (0 <= to_file < 8 and 0 <= to_rank < 8):
                continue
            to_sq = chess.square(to_file, to_rank)
            plane = 56 + knight_idx
            idx = from_sq * NUM_PLANES + plane
            move_to_idx[chess.Move(from_sq, to_sq)] = idx

        # Underpromotions (only from rank 6 → rank 7 for white)
        if from_rank == 6:
            for promo_idx, promo_piece in enumerate(UNDER_PROMO_PIECES):
                for dir_offset_idx, df in enumerate(UNDER_PROMO_DIRS):
                    to_file = from_file + df
                    to_rank = 7
                    if not (0 <= to_file < 8):
                        continue
                    to_sq = chess.square(to_file, to_rank)
                    plane = 64 + promo_idx * 3 + dir_offset_idx
                    idx = from_sq * NUM_PLANES + plane
                    move_to_idx[chess.Move(from_sq, to_sq, promo_piece)] = idx

    return move_to_idx


def move_to_index(move: chess.Move, flip: bool = False) -> int:
    """
    Convert a chess.Move to a flat action index [0, 4672).
    Pass flip=True when encoding Black's moves (board was mirrored).
    """
    if flip:
        move = _mirror_move(move)
    return _build_move_to_index()[move]


# Add this alongside _build_move_to_index — cached once, reused forever
@lru_cache(maxsize=None)
def _build_index_to_move() -> dict:
    return {v: k for k, v in _build_move_to_index().items()}

def index_to_move(idx: int, flip: bool = False) -> chess.Move:
    move = _build_index_to_move()[idx]   # O(1) lookup, no rebuild
    if flip:
        move = _mirror_move(move)
    return move


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    Boolean mask of shape (4672,) — True for every legal move.
    Used to zero-out illegal moves from the policy logits.
    """
    flip = board.turn == chess.BLACK
    mask = np.zeros(TOTAL_ACTIONS, dtype=bool)
    move_map = _build_move_to_index()

    for move in board.legal_moves:
        m = _mirror_move(move) if flip else move
        if m in move_map:
            mask[move_map[m]] = True

    return mask


def policy_to_move_probs(
    policy_logits: np.ndarray,
    board: chess.Board,
) -> dict[chess.Move, float]:
    """
    Convert raw policy logits → probability dict over legal moves.
    Handles illegal move masking and perspective flip for Black.
    """
    flip = board.turn == chess.BLACK
    mask = legal_move_mask(board)

    # Mask illegal moves, softmax over legal ones
    logits = policy_logits.copy()
    logits[~mask] = -1e9
    logits -= logits.max()       # numerical stability
    exp = np.exp(logits)
    probs = exp / exp.sum()

    # Map back to chess.Move
    move_map = _build_move_to_index()
    idx_to_move_raw = {v: k for k, v in move_map.items()}
    result = {}
    for move in board.legal_moves:
        m = _mirror_move(move) if flip else move
        if m in move_map:
            result[move] = float(probs[move_map[m]])

    return result


def _mirror_move(move: chess.Move) -> chess.Move:
    """Mirror a move vertically (rank 0↔7) for Black's perspective."""
    def mirror_sq(sq):
        return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))
    promo = move.promotion
    return chess.Move(mirror_sq(move.from_square), mirror_sq(move.to_square), promo)