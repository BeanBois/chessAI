import chess
import numpy as np
import copy
from .env import ChessGame
from .cnn_model.network import NeuralNetwork
from .mcts import MCTS


class Evaluator:
    """
    Pits the newly trained network against the current best network.
    Promotes the new network if it wins more than `win_threshold` of games.

    AlphaZero uses 400 evaluation games and a 55% win threshold.
    Reduce for faster iteration while prototyping.
    """

    def __init__(
        self,
        num_games: int      = 40,
        win_threshold: float = 0.55,
        num_simulations: int = 400,   # fewer sims than self-play for speed
    ):
        self.num_games       = num_games
        self.win_threshold   = win_threshold
        self.num_simulations = num_simulations

    def evaluate(
        self,
        candidate_net: NeuralNetwork,
        best_net: NeuralNetwork,
    ) -> tuple[bool, dict]:
        """
        Play num_games between candidate and best.
        Alternates who plays White to remove first-move advantage.

        Returns:
          promoted — True if candidate should become the new best
          stats    — dict with win/draw/loss counts and win rate
        """
        candidate_wins = 0
        draws          = 0
        best_wins      = 0

        for game_idx in range(self.num_games):
            # Alternate colours
            candidate_is_white = (game_idx % 2 == 0)
            result = self._play_one(candidate_net, best_net, candidate_is_white)

            if result == "candidate":
                candidate_wins += 1
            elif result == "draw":
                draws += 1
            else:
                best_wins += 1

            print(
                f"  Eval game {game_idx+1}/{self.num_games} | "
                f"Candidate {candidate_wins} — {draws} draws — {best_wins} Best"
            )

        total_decisive = candidate_wins + best_wins
        win_rate = candidate_wins / self.num_games

        # Win rate counts draws as 0.5 for the candidate (same as AlphaZero)
        adjusted_win_rate = (candidate_wins + 0.5 * draws) / self.num_games
        promoted = adjusted_win_rate >= self.win_threshold

        stats = {
            "candidate_wins": candidate_wins,
            "draws":          draws,
            "best_wins":      best_wins,
            "win_rate":       win_rate,
            "adjusted_win_rate": adjusted_win_rate,
            "promoted":       promoted,
        }
        return promoted, stats

    def _play_one(
        self,
        candidate_net: NeuralNetwork,
        best_net: NeuralNetwork,
        candidate_is_white: bool,
    ) -> str:
        """
        Play a single game. Returns 'candidate', 'best', or 'draw'.
        """
        env = ChessGame()
        env.reset()

        white_net = candidate_net if candidate_is_white else best_net
        black_net = best_net      if candidate_is_white else candidate_net

        white_mcts = MCTS(neural_net=white_net, num_simulations=self.num_simulations, temperature=0)
        black_mcts = MCTS(neural_net=black_net, num_simulations=self.num_simulations, temperature=0)

        while not env.is_terminal():
            if env.current_player == chess.WHITE:
                move = white_mcts.select_move(env)
            else:
                move = black_mcts.select_move(env)
            env.step(move)

        outcome = env.board.outcome()
        if outcome.winner is None:
            return "draw"
        winner_is_white = outcome.winner == chess.WHITE
        candidate_won   = winner_is_white == candidate_is_white
        return "candidate" if candidate_won else "best"