import chess
import numpy as np
import math
import torch
from typing import Optional
from .cnn_model.move_encoder import policy_to_move_probs


class MCTSNode:
    def __init__(
        self,
        env=None,
        parent: Optional["MCTSNode"] = None,
        move: Optional[chess.Move]   = None,
        prior: float                 = 0.0,
    ):
        self.parent        = parent
        self.move          = move
        self.prior         = prior
        self.visit_count   = 0
        self.value_sum     = 0.0
        self.children: dict[chess.Move, "MCTSNode"] = {}
        self.is_expanded   = False
        self._nn_value     = None
        self._virtual_loss = 0
        self._env          = env

    @property
    def env(self):
        if self._env is None:
            assert self.parent is not None and self.move is not None
            self._env = self.parent.env.clone()
            self._env.step(self.move)
        return self._env

    @property
    def q_value(self) -> float:
        n = self.visit_count + self._virtual_loss
        return self.value_sum / n if n > 0 else 0.0

    @property
    def is_leaf(self) -> bool:
        return not self.is_expanded

    @property
    def is_terminal(self) -> bool:
        return self.env.is_terminal()

    def puct_score(self, c_puct: float) -> float:
        assert self.parent is not None
        n_parent = self.parent.visit_count + self.parent._virtual_loss
        exploration = (
            c_puct
            * self.prior
            * math.sqrt(n_parent)
            / (1 + self.visit_count + self._virtual_loss)
        )
        return self.q_value + exploration

    def best_child(self, c_puct: float) -> "MCTSNode":
        return max(self.children.values(), key=lambda n: n.puct_score(c_puct))

    def apply_virtual_loss(self, path: list["MCTSNode"]):
        for node in path:
            node._virtual_loss += 1

    def revert_virtual_loss(self, path: list["MCTSNode"]):
        for node in path:
            node._virtual_loss -= 1


class MCTS:
    def __init__(
        self,
        neural_net,
        c_puct: float            = 1.4,
        num_simulations: int     = 800,
        batch_size: int          = 32,
        dirichlet_alpha: float   = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float       = 1.0,
    ):
        self.neural_net          = neural_net
        self.c_puct              = c_puct
        self.num_simulations     = num_simulations
        self.batch_size          = batch_size
        self.dirichlet_alpha     = dirichlet_alpha
        self.dirichlet_epsilon   = dirichlet_epsilon
        self.temperature         = temperature
        self._next_root: Optional[MCTSNode] = None

        # State for externally-driven parallel search (ParallelSelfPlay)
        self._search_root:   Optional[MCTSNode] = None
        self._search_sims:   int  = 0
        self._root_expanded: bool = False

    # ------------------------------------------------------------------
    # Parallel search API  (used by ParallelSelfPlay)
    # ------------------------------------------------------------------

    def prepare_search(self, env) -> None:
        """
        Initialise a root node ready for the next position.
        Must be called once per move before any select_leaves() calls.
        Handles tree reuse via _next_root automatically.

        Does NOT call the neural network — root expansion is deferred to
        the first select_leaves() call so it can be batched with other games.
        """
        if self._next_root is not None and self._next_root.is_expanded:
            # Reuse the subtree we saved from the previous move
            root = self._next_root
            self._next_root = None
            for child in root.children.values():
                child.parent = root
            self._root_expanded = True   # already has priors from last search
        else:
            self._next_root = None
            root = MCTSNode(env=env.clone())
            self._root_expanded = False  # needs NN eval before we can search

        self._search_root  = root
        self._current_root = root
        self._search_sims  = 0

    def select_leaves(self, n: int) -> tuple[list, list]:
        """
        Selection phase only — NO neural network call happens here.

        If the root is brand new (not yet expanded), the root itself is
        returned as the single leaf so the caller can evaluate it in a
        pooled batch alongside other games' leaves.

        Terminal nodes are resolved in-place (no NN needed) and counted
        against the simulation budget.

        Returns (leaves, paths) for the caller to pass to process_results()
        after a batched NN forward pass.
        """
        root = self._search_root

        # Fresh root — must be expanded before we can run PUCT
        if not self._root_expanded:
            return [root], [[root]]

        leaves: list = []
        paths:  list = []

        for _ in range(n):
            if self._search_sims + len(leaves) >= self.num_simulations:
                break
            node, path = self._select(root)
            if node is None:
                break
            # Terminals don't need NN — backprop game result immediately.
            # Mark as expanded so _select won't revisit this node.
            if node.is_terminal:
                value = node.env.get_result(node.env.current_player)
                self._backpropagate(path, value)
                node.is_expanded = True  # prevent re-visiting this terminal
                self._search_sims += 1
                continue
            node.apply_virtual_loss(path)
            leaves.append(node)
            paths.append(path)

        return leaves, paths

    def process_results(
        self,
        leaves:           list,
        paths:            list,
        policy_logits_np: np.ndarray,  # shape (len(leaves), 4672)
        values_np:        np.ndarray,  # shape (len(leaves),)
    ) -> None:
        """
        Expand leaves and backpropagate after an external batched NN call.

        The caller (ParallelSelfPlay) pools leaves from N games into one
        big batch, runs a single GPU forward pass, then calls this for each
        game with its slice of the results.

        Dirichlet noise is injected the first time the root is expanded.
        """
        for i, node in enumerate(leaves):
            node._nn_value = float(values_np[i])
            policy_dict    = policy_to_move_probs(policy_logits_np[i], node.env.board)
            for move in node.env.get_legal_moves():
                node.children[move] = MCTSNode(
                    env    = None,
                    parent = node,
                    move   = move,
                    prior  = policy_dict.get(move, 1e-8),
                )
            node.is_expanded = True

            # Root expanded for first time — add exploration noise now
            if node is self._search_root:
                self._add_dirichlet_noise(node)
                self._root_expanded = True

        for node, path in zip(leaves, paths):
            node.revert_virtual_loss(path)
            self._backpropagate(path, node._nn_value)

        self._search_sims += len(leaves)

    def search_done(self) -> bool:
        """True once the root is expanded and the simulation budget is spent."""
        return self._root_expanded and self._search_sims >= self.num_simulations

    # ------------------------------------------------------------------
    # Original single-game API  (Evaluator and SelfPlay.play_game use this)
    # ------------------------------------------------------------------

    def get_action_probs(self, env) -> dict[chess.Move, float]:
        if self.neural_net is None:
            raise RuntimeError(
                "get_action_probs() cannot be called when neural_net=None. "
                "This MCTS instance is in parallel mode — NN calls are driven "
                "externally by ParallelSelfPlay."
            )
        root = MCTSNode(env=env.clone())
        self._expand_batch([root])
        self._add_dirichlet_noise(root)

        sims_done = 0
        while sims_done < self.num_simulations:
            batch_paths:  list = []
            batch_leaves: list = []

            for _ in range(self.batch_size):
                if sims_done + len(batch_leaves) >= self.num_simulations:
                    break
                node, path = self._select(root)
                if node is None:
                    break
                node.apply_virtual_loss(path)
                batch_paths.append(path)
                batch_leaves.append(node)

            if not batch_leaves:
                break

            self._expand_batch(batch_leaves)

            for node, path in zip(batch_leaves, batch_paths):
                node.revert_virtual_loss(path)
                value = (
                    node.env.get_result(node.env.current_player)
                    if node.is_terminal else node._nn_value
                )
                self._backpropagate(path, value)

            sims_done += len(batch_leaves)

        self._current_root = root
        return self._action_probabilities(root)

    # ------------------------------------------------------------------
    # Tree reuse (shared by both APIs)
    # ------------------------------------------------------------------

    def advance_root(self, move: chess.Move):
        """
        Save the chosen move's child subtree for the next search.
        Call BEFORE env.step() so the parent env is still alive for
        lazy board cloning.
        """
        source = getattr(self, '_current_root', None) or self._search_root
        if source is None or move not in source.children:
            self._next_root = None
            return
        candidate = source.children[move]
        # Materialise the lazy env clone BEFORE detaching the parent.
        # MCTSNode.env lazily calls parent.env.clone() on first access, so the
        # entire ancestor chain must still be alive when we access it here.
        # After this line the candidate's _env is a fully independent object
        # and parent detachment is safe — the rest of the tree can be GC'd.
        _ = candidate.env
        candidate.parent = None  # detach — lets the rest of the tree be GC'd
        self._next_root = candidate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select(self, root: MCTSNode) -> tuple[Optional[MCTSNode], list]:
        node = root
        path = [node]
        while not node.is_leaf and not node.is_terminal:
            node = node.best_child(self.c_puct)
            path.append(node)
        return node, path

    def _expand_batch(self, leaves: list):
        to_evaluate = [n for n in leaves if not n.is_terminal and not n.is_expanded]
        if not to_evaluate:
            return
        boards = [n.env.board for n in to_evaluate]
        batch  = self.neural_net.encode_batch(boards)
        policy_logits_np, values_np = self.neural_net.evaluate_batch_infer(batch)
        for i, node in enumerate(to_evaluate):
            node._nn_value = float(values_np[i])
            policy_dict    = policy_to_move_probs(policy_logits_np[i], node.env.board)
            for move in node.env.get_legal_moves():
                node.children[move] = MCTSNode(
                    env=None, parent=node, move=move,
                    prior=policy_dict.get(move, 1e-8),
                )
            node.is_expanded = True

    def _backpropagate(self, path: list, value: float):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum   += value
            value = -value

    def _add_dirichlet_noise(self, root: MCTSNode):
        if not root.children:
            return
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (
                (1 - self.dirichlet_epsilon) * child.prior
                + self.dirichlet_epsilon * n
            )

    def _action_probabilities(self, root: MCTSNode) -> dict[chess.Move, float]:
        moves        = list(root.children.keys())
        visit_counts = np.array(
            [root.children[m].visit_count for m in moves], dtype=np.float32
        )
        if self.temperature == 0:
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            counts_temp = visit_counts ** (1.0 / self.temperature)
            probs       = counts_temp / counts_temp.sum()
        return dict(zip(moves, probs))

    def select_move(self, env) -> chess.Move:
        probs   = self.get_action_probs(env)
        moves   = list(probs.keys())
        weights = list(probs.values())
        return np.random.choice(moves, p=weights)