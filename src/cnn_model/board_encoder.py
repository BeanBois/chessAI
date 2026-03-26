import chess
import numpy as np

# Plane layout (18 planes total, each 8×8):
#   0–5   : white pieces  (P N B R Q K)
#   6–11  : black pieces  (P N B R Q K)
#   12    : side to move  (all-1 = white, all-0 = black)
#   13–16 : castling rights (WK WQ BK BQ)
#   17    : en passant file (1 on the file where e.p. is possible)

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a board into an (18, 8, 8) float32 array.
    Always from the perspective of the CURRENT player (board is flipped
    if it's Black's turn so the network always 'sees' itself as White).
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # Flip board if it's Black's turn — network always plays as White
    b = board.mirror() if board.turn == chess.BLACK else board

    # Piece planes
    for plane_idx, piece_type in enumerate(PIECE_ORDER):
        for sq in b.pieces(piece_type, chess.WHITE):
            row, col = divmod(sq, 8)
            planes[plane_idx, row, col] = 1.0

        for sq in b.pieces(piece_type, chess.BLACK):
            row, col = divmod(sq, 8)
            planes[plane_idx + 6, row, col] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    # Castling rights (from current player's perspective after mirror)
    if b.has_kingside_castling_rights(chess.WHITE):  planes[13, :, :] = 1.0
    if b.has_queenside_castling_rights(chess.WHITE): planes[14, :, :] = 1.0
    if b.has_kingside_castling_rights(chess.BLACK):  planes[15, :, :] = 1.0
    if b.has_queenside_castling_rights(chess.BLACK): planes[16, :, :] = 1.0

    # En passant
    if b.ep_square is not None:
        ep_file = chess.square_file(b.ep_square)
        planes[17, :, ep_file] = 1.0

    return planes