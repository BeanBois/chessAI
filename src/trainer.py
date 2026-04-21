import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .cnn_model.board_encoder import decode_board
from .cnn_model.move_encoder import legal_move_mask
from .replay_buffer import ReplayBuffer


class Trainer:
    """
    Trains the network on batches sampled from the replay buffer.

    Loss = cross-entropy(policy) + MSE(value)
    Optimizer: AdamW with linear warmup + cosine decay.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str          = "cpu",
        lr: float            = 3e-4,
        weight_decay: float  = 1e-4,
        batch_size: int      = 512,
        epochs_per_update: int = 5,
        warmup_iterations: int = 10,
        num_iterations: int    = 200,
    ):
        self.model   = model.to(device)
        self.device  = torch.device(device)
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR schedule: linear warmup then cosine decay
        self._warmup_iters = warmup_iterations
        self._total_iters  = num_iterations
        self._base_lr      = lr
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(num_iterations - warmup_iterations, 1),
            eta_min=1e-5,
        )
        self._current_iter = 0

    def _warmup_lr(self, iteration: int):
        """Apply linear warmup for the first N iterations."""
        if iteration <= self._warmup_iters:
            warmup_factor = iteration / max(self._warmup_iters, 1)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self._base_lr * warmup_factor

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

        # Warmup LR for early iterations
        self._warmup_lr(iteration)

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
            # Compute legal move masks per-batch to avoid holding a full (N, 4672)
            # array in memory.
            s_np = s_batch.numpy()
            mask_np = np.full((s_np.shape[0], 4672), float('-inf'), dtype=np.float32)
            for i in range(s_np.shape[0]):
                board = decode_board(s_np[i])
                mask_np[i, legal_move_mask(board)] = 0.0
            mask_batch = torch.from_numpy(mask_np)

            s_batch    = s_batch.to(self.device)
            pi_batch   = pi_batch.to(self.device)
            z_batch    = z_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)

            policy_logits, value_pred = self.model(s_batch)

            # Policy loss: cross-entropy between MCTS distribution and network output.
            # Mask illegal moves to -inf before log_softmax so the network only
            # competes over legal moves (same as inference in policy_to_move_probs).
            masked_logits = policy_logits + mask_batch
            log_probs     = torch.log_softmax(masked_logits, dim=1)
            policy_loss   = -(torch.nan_to_num(pi_batch * log_probs, nan=0.0)).sum(dim=1).mean()

            # Value loss: MSE between predicted scalar and game outcome
            value_loss = nn.functional.mse_loss(value_pred.squeeze(-1), z_batch)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            num_batches       += 1

        # Step cosine schedule only after warmup
        if iteration > self._warmup_iters:
            self.scheduler.step()

        self.model.eval()

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss":  total_value_loss  / num_batches,
            "total_loss":  (total_policy_loss + total_value_loss) / num_batches,
            "lr":          self.optimizer.param_groups[0]['lr'],
        }
