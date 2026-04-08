import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .cnn_model.network import ChessNet
from .cnn_model.board_encoder import decode_board
from .cnn_model.move_encoder import legal_move_mask
from .replay_buffer import ReplayBuffer


class Trainer:
    """
    Trains the network on batches sampled from the replay buffer.

    Loss = cross-entropy(policy) + MSE(value)

    AlphaZero uses SGD + momentum + L2 weight decay.
    Adam works too but SGD generalises better for self-play.
    """

    def __init__(
        self,
        model: ChessNet,
        device: str          = "cpu",
        lr: float            = 2e-3,
        momentum: float      = 0.9,
        weight_decay: float  = 1e-4,
        batch_size: int      = 512,
        epochs_per_update: int = 5,   # passes over sampled data each iteration
    ):
        self.model   = model.to(device)
        self.device  = torch.device(device)
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update

        self.optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        # LR schedule: cosine decay over training
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5
        )

    def train_step(self, buffer: ReplayBuffer, iteration: int = 0, num_iterations: int = 200) -> dict[str, float]:
        """
        Sample a batch from the buffer and do one round of gradient updates.
        Returns a dict of losses for logging.
        Caller must ensure len(buffer) >= batch_size before calling.
        """
        assert len(buffer) >= self.batch_size, (
            f"Buffer too small: {len(buffer)} < {self.batch_size}. "
            "Check MIN_BUFFER_SIZE guard in train.py."
        )

        # Decay weight decay from 1e-4 → 1e-5 over training
        progress = min(iteration / max(num_iterations, 1), 1.0)
        current_wd = 1e-4 * (1.0 - 0.9 * progress) + 1e-5 * 0.9 * progress
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = current_wd

        states, policies, values = buffer.sample(
            min(len(buffer), self.batch_size * self.epochs_per_update)
        )

        # Precompute legal move masks for all sampled positions.
        # Illegal indices get -inf so log_softmax zeroes them out; legal indices get 0.0.
        N = states.shape[0]
        mask_np = np.full((N, 4672), float('-inf'), dtype=np.float32)
        for i in range(N):
            board = decode_board(states[i])
            mask_np[i, legal_move_mask(board)] = 0.0
        masks = torch.from_numpy(mask_np)

        dataset = TensorDataset(
            torch.from_numpy(states),
            torch.from_numpy(policies),
            torch.from_numpy(values),
            masks,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        num_batches       = 0

        for s_batch, pi_batch, z_batch, mask_batch in loader:
            s_batch    = s_batch.to(self.device)
            pi_batch   = pi_batch.to(self.device)
            z_batch    = z_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)

            policy_logits, value_pred = self.model(s_batch)

            # Policy loss: cross-entropy between MCTS distribution and network output.
            # Mask illegal moves to -inf before log_softmax so the network only
            # competes over legal moves (same as inference in policy_to_move_probs).
            # nan_to_num guards against 0 * -inf = nan at zero-probability positions.
            masked_logits = policy_logits + mask_batch
            log_probs     = torch.log_softmax(masked_logits, dim=1)
            policy_loss   = -(torch.nan_to_num(pi_batch * log_probs, nan=0.0)).sum(dim=1).mean()

            # Value loss: cross-entropy over {loss, draw, win} classes
            # Target: z=-1.0 → class 0 (loss), z=0.0 → class 1 (draw), z=1.0 → class 2 (win)
            value_class = ((z_batch + 1.0) * 1.0).long().clamp(0, 2)  # -1→0, 0→1, 1→2
            value_loss = nn.functional.cross_entropy(
                value_pred, value_class
            )

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents exploding gradients early in training
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            num_batches       += 1

        self.scheduler.step()
        self.model.eval()

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss":  total_value_loss  / num_batches,
            "total_loss":  (total_policy_loss + total_value_loss) / num_batches,
            "lr":          self.scheduler.get_last_lr()[0],
        }