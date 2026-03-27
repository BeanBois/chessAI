import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .cnn_model.network import ChessNet
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

    def train_step(self, buffer: ReplayBuffer) -> dict[str, float]:
        """
        Sample a batch from the buffer and do one round of gradient updates.
        Returns a dict of losses for logging.
        Caller must ensure len(buffer) >= batch_size before calling.
        """
        assert len(buffer) >= self.batch_size, (
            f"Buffer too small: {len(buffer)} < {self.batch_size}. "
            "Check MIN_BUFFER_SIZE guard in train.py."
        )

        states, policies, values = buffer.sample(
            min(len(buffer), self.batch_size * self.epochs_per_update)
        )

        dataset = TensorDataset(
            torch.from_numpy(states),
            torch.from_numpy(policies),
            torch.from_numpy(values),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        num_batches       = 0

        for s_batch, pi_batch, z_batch in loader:
            s_batch  = s_batch.to(self.device)
            pi_batch = pi_batch.to(self.device)
            z_batch  = z_batch.to(self.device)

            policy_logits, value_pred = self.model(s_batch)

            # Policy loss: cross-entropy between MCTS distribution and network output
            # log_softmax + (target * log_prob) — more numerically stable than softmax first
            log_probs   = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -(pi_batch * log_probs).sum(dim=1).mean()

            # Value loss: MSE between network estimate and actual outcome
            value_loss = nn.functional.mse_loss(
                value_pred.squeeze(1), z_batch
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