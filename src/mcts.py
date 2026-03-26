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
        """
        Lazy clone — board is only copied when this node is actually
        selected during tree traversal, not when it's created as a child.
        """
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
        batch_size: int          = 32,   # was 16 — larger batch = fewer NN calls
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

        # Tree reuse: persists between get_action_probs() calls within a game.
        # advance_root() sets this after each move is committed.
        self._next_root: Optional[MCTSNode] = None

    # ------------------------------------------------------------------
    # Tree reuse
    # ------------------------------------------------------------------

    def advance_root(self, move: chess.Move):
        """
        Call this immediately AFTER a move is chosen but BEFORE env.step().
        Saves the child corresponding to `move` so the next search reuses
        all visits already recorded in that subtree.

        Why this works: the child node already contains visit_count,
        value_sum, and expanded grandchildren from this move's search.
        The next search starts with those counts rather than zero,
        effectively getting them "for free".
        """
        if self._next_root is not None and move in self._next_root.children:
            # We already have a deeper saved root — walk one level further.
            candidate = self._next_root.children[move]
        elif hasattr(self, '_current_root') and move in self._current_root.children:
            candidate = self._current_root.children[move]
        else:
            # Move not in tree (shouldn't happen in normal play).
            self._next_root = None
            return

        # Materialise the env NOW while the parent is still alive
        # (lazy eval needs the parent chain; once parent=None it can't clone).
        _ = candidate.env          # triggers lazy clone + step if needed
        candidate.parent = None    # detach — allows GC of the rest of the tree
        self._next_root = candidate

    def get_action_probs(self, env) -> dict[chess.Move, float]:
        # ── Reuse or build root ──────────────────────────────────────────
        if self._next_root is not None and self._next_root.is_expanded:
            root = self._next_root
            self._next_root = None
            # Re-parent children so PUCT parent counts are correct
            for child in root.children.values():
                child.parent = root
        else:
            self._next_root = None
            root = MCTSNode(env=env.clone())
            self._expand_batch([root])

        # Always inject fresh Dirichlet noise at the root for exploration.
        # For reused roots the visits dominate the prior anyway for
        # well-explored children, so this only matters for rarely-visited ones.
        self._add_dirichlet_noise(root)

        # Save so advance_root() can walk into the chosen child
        self._current_root = root

        # ── Simulation loop ──────────────────────────────────────────────
        sims_done = 0
        while sims_done < self.num_simulations:
            batch_paths  = []
            batch_leaves = []

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
                if node.is_terminal:
                    value = node.env.get_result(node.env.current_player)
                else:
                    value = node._nn_value
                self._backpropagate(path, value)

            sims_done += len(batch_leaves)

        return self._action_probabilities(root)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select(self, root: MCTSNode) -> tuple[Optional[MCTSNode], list]:
        node = root
        path = [node]

        while not node.is_leaf and not node.is_terminal:
            node = node.best_child(self.c_puct)
            path.append(node)

        if node.is_terminal:
            return node, path

        return node, path

    # ------------------------------------------------------------------
    # Batched expansion
    # ------------------------------------------------------------------

    def _expand_batch(self, leaves: list[MCTSNode]):
        to_evaluate = [n for n in leaves if not n.is_terminal and not n.is_expanded]
        if not to_evaluate:
            return

        boards = [n.env.board for n in to_evaluate]
        batch  = self.neural_net.encode_batch(boards)

        with torch.no_grad():
            policy_logits, values = self.neural_net.model(batch)

        policy_logits_np = policy_logits.cpu().numpy()
        values_np        = values.squeeze(1).cpu().numpy()

        for i, node in enumerate(to_evaluate):
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

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, path: list[MCTSNode], value: float):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum   += value
            value = -value

    # ------------------------------------------------------------------
    # Dirichlet noise
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

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
            probs = counts_temp / counts_temp.sum()

        return dict(zip(moves, probs))

    def select_move(self, env) -> chess.Move:
        probs   = self.get_action_probs(env)
        moves   = list(probs.keys())
        weights = list(probs.values())
        return np.random.choice(moves, p=weights)