import chess
import numpy as np

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
MAX_CAPTURE_VALUE = 9.0
CAPTURE_WEIGHT = 0.5
MOBILITY_WEIGHT = 0.05


class ChessGame:
    def __init__(self, board: chess.Board = None):
        self.board = board if board is not None else chess.Board()
        self.t = 0

    def reset(self):
        self.board = chess.Board()
        self.t = 0
        return self._get_obs()

    # ------------------------------------------------------------------
    # Core MCTS interface
    # ------------------------------------------------------------------

    def clone(self) -> "ChessGame":
        """Deep copy the game state — essential for MCTS simulations."""
        cloned = ChessGame(board=self.board.copy())
        cloned.t = self.t
        return cloned

    def undo_move(self):
        """Pop the last move — for memory-efficient MCTS tree traversal."""
        self.board.pop()
        self.t -= 1

    def is_terminal(self) -> bool:
        """
        Covers all FIDE draw conditions that python-chess tracks:
        - Checkmate
        - Stalemate
        - Threefold repetition (can_claim, not just is_fivefold)
        - Fifty-move rule (can_claim)
        - Insufficient material
        """
        return (
            self.board.is_game_over()
            or self.board.can_claim_threefold_repetition()
            or self.board.can_claim_fifty_moves()
        )

    def get_result(self, player: chess.Color) -> float:
        assert self.is_terminal(), "get_result() called on non-terminal state"

        # Claim draws before checking outcome — outcome() returns None mid-game
        if (
            self.board.can_claim_threefold_repetition()
            or self.board.can_claim_fifty_moves()
            or self.board.is_insufficient_material()
        ):
            return 0.0

        outcome = self.board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == player else -1.0

    @property
    def current_player(self) -> chess.Color:
        """Whose turn is it? Used by MCTS for negamax backpropagation."""
        return self.board.turn

    def get_legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    # ------------------------------------------------------------------
    # Step (used for training signal, NOT for MCTS backprop value)
    # ------------------------------------------------------------------

    def step(self, action: chess.Move) -> dict:
        """
        Step the environment. Returns training rewards (capture + mobility).
        MCTS uses get_result() instead of this reward for backpropagation.
        """
        assert not self.is_terminal(), "Game is over — call reset()"
        assert action in self.board.legal_moves, f"Illegal move: {action}"

        reward = self._capture_reward(action)
        self.board.push(action)
        self.t += 1

        done = self.is_terminal()
        if done:
            # Terminal reward for training (not used by MCTS directly)
            outcome = self.board.outcome()
            if outcome is None:
                reward += 0.0
            else:
                just_moved = not self.board.turn
                reward += 1.0 if outcome.winner == just_moved else -1.0
        else:
            reward += self._mobility_reward()

        return {
            'reward': reward,
            'done': done,
            'legal_moves': self.get_legal_moves()
        }

    # ------------------------------------------------------------------
    # Reward components (for neural network training loss)
    # ------------------------------------------------------------------

    def _capture_reward(self, action: chess.Move) -> float:
        if not self.board.is_capture(action):
            return 0.0
        if self.board.is_en_passant(action):
            return CAPTURE_WEIGHT * (PIECE_VALUES[chess.PAWN] / MAX_CAPTURE_VALUE)
        captured = self.board.piece_at(action.to_square)
        if captured is None:
            return 0.0
        return CAPTURE_WEIGHT * (PIECE_VALUES.get(captured.piece_type, 0) / MAX_CAPTURE_VALUE)

    def _mobility_reward(self) -> float:
        opponent_mobility = self.board.legal_moves.count()
        # Toggle turn to count our legal moves without pushing an illegal null move
        self.board.turn = not self.board.turn
        our_mobility = self.board.legal_moves.count()
        self.board.turn = not self.board.turn
        diff = our_mobility - opponent_mobility
        return MOBILITY_WEIGHT * np.clip(diff / 218.0, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        return self.board, self.t, {'legal_moves': self.get_legal_moves()}

    def render(self):
        print(self.board)
        print(f"Turn: {'White' if self.board.turn else 'Black'} | Move: {self.t}")