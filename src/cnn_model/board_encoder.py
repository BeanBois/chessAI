import chess
import numpy as np

# Plane layout per time-step (18 planes each, 8×8):
#   0–5   : white pieces  (P N B R Q K)
#   6–11  : black pieces  (P N B R Q K)
#   12    : side to move  (all-1 = white, all-0 = black)
#   13–16 : castling rights (WK WQ BK BQ)
#   17    : en passant file (1 on the file where e.p. is possible)
#
# With T=8 history: total input = 18 * 8 = 144 planes
# Ordered oldest → newest; most recent position occupies planes [126:144]

PLANES_PER_STEP = 18
HISTORY_LENGTH  = 8
TOTAL_PLANES    = PLANES_PER_STEP * HISTORY_LENGTH  # 144

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]


def _encode_single(board: chess.Board) -> np.ndarray:
    """
    Encode one board position into (18, 8, 8) float32 from current player's POV.
    Board is mirrored if it's Black's turn so the network always 'sees' itself as White.
    """
    planes = np.zeros((PLANES_PER_STEP, 8, 8), dtype=np.float32)

    b = board.mirror() if board.turn == chess.BLACK else board

    for plane_idx, piece_type in enumerate(PIECE_ORDER):
        squares = list(b.pieces(piece_type, chess.WHITE))
        if squares:
            rows, cols = np.divmod(squares, 8)
            planes[plane_idx, rows, cols] = 1.0

        squares = list(b.pieces(piece_type, chess.BLACK))
        if squares:
            rows, cols = np.divmod(squares, 8)
            planes[plane_idx + 6, rows, cols] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    if b.has_kingside_castling_rights(chess.WHITE):  planes[13, :, :] = 1.0
    if b.has_queenside_castling_rights(chess.WHITE): planes[14, :, :] = 1.0
    if b.has_kingside_castling_rights(chess.BLACK):  planes[15, :, :] = 1.0
    if b.has_queenside_castling_rights(chess.BLACK): planes[16, :, :] = 1.0

    if b.ep_square is not None:
        ep_file = chess.square_file(b.ep_square)
        planes[17, :, ep_file] = 1.0

    return planes


def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a board into a (144, 8, 8) float32 array using T=8 position history.

    The board's move_stack is used to reconstruct past positions.
    If fewer than T positions are available, earlier slots are padded with
    the starting position.

    Always from the perspective of the CURRENT player (board is flipped
    if it's Black's turn so the network always 'sees' itself as White).

    Returns: (144, 8, 8) = 8 time steps × 18 planes each, oldest first.
    """
    # Collect up to HISTORY_LENGTH past board states by replaying move stack
    moves = list(board.move_stack)
    # Replay: unwind to starting position
    b = board.copy()
    states = []
    # current state first, then undo moves
    for _ in range(min(HISTORY_LENGTH, len(moves) + 1)):
        states.append(_encode_single(b))
        if b.move_stack:
            b.pop()

    # states[0] = most recent, states[-1] = oldest available
    states.reverse()  # now oldest first

    # Pad to HISTORY_LENGTH with starting-position encoding if needed
    if len(states) < HISTORY_LENGTH:
        start_planes = _encode_single(chess.Board())
        padding = [start_planes] * (HISTORY_LENGTH - len(states))
        states = padding + states

    return np.concatenate(states, axis=0)  # (144, 8, 8)


def decode_board(planes: np.ndarray) -> chess.Board:
    """
    Reconstruct a chess.Board from encoded state planes.

    Accepts either (18, 8, 8) single-position or (144, 8, 8) history encoding.
    For history input, uses the most recent (last) 18 planes.

    Inverse of _encode_single() for the current position.

    NOTE: Move history and repetition data are not reconstructed.
    The result is sufficient for generating legal moves.
    """
    # Extract the most recent 18 planes regardless of input shape
    if planes.shape[0] == TOTAL_PLANES:
        planes = planes[-PLANES_PER_STEP:]  # last 18 planes = current position

    board = chess.Board(fen=None)
    board.clear()

    is_white_turn = planes[12, 0, 0] > 0.5

    for plane_idx, piece_type in enumerate(PIECE_ORDER):
        white_squares = np.argwhere(planes[plane_idx] > 0.5)
        for row, col in white_squares:
            sq = row * 8 + col
            board.set_piece_at(sq, chess.Piece(piece_type, chess.WHITE))

        black_squares = np.argwhere(planes[plane_idx + 6] > 0.5)
        for row, col in black_squares:
            sq = row * 8 + col
            board.set_piece_at(sq, chess.Piece(piece_type, chess.BLACK))

    board.castling_rights = chess.BB_EMPTY
    if planes[13, 0, 0] > 0.5: board.castling_rights |= chess.BB_H1  # WK
    if planes[14, 0, 0] > 0.5: board.castling_rights |= chess.BB_A1  # WQ
    if planes[15, 0, 0] > 0.5: board.castling_rights |= chess.BB_H8  # BK
    if planes[16, 0, 0] > 0.5: board.castling_rights |= chess.BB_A8  # BQ

    ep_cols = np.where(np.any(planes[17] > 0.5, axis=0))[0]
    if len(ep_cols) > 0:
        ep_file = int(ep_cols[0])
        board.ep_square = chess.square(ep_file, 5)

    board.turn = chess.WHITE
    if not is_white_turn:
        board = board.mirror()
        board.turn = chess.BLACK

    return board
