import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

from .board_encoder import encode_board, TOTAL_PLANES
from .move_encoder import policy_to_move_probs, TOTAL_ACTIONS


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    """Standard AlphaZero residual block: two conv layers + skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


# ------------------------------------------------------------------
# Full network
# ------------------------------------------------------------------

class ChessNet(nn.Module):
    """
    AlphaZero-style network.

    Input:  (B, 18, 8, 8)  — encoded board planes
    Output: policy logits (B, 4672)  +  value scalar (B, 1)
    """

    def __init__(
        self,
        in_planes:    int = TOTAL_PLANES,  # 144 (8 history × 18 planes)
        num_channels: int = 256,   # residual tower width
        num_blocks:   int = 20,    # residual tower depth
        policy_size:  int = TOTAL_ACTIONS,  # 4672
    ):
        super().__init__()

        # Stem
        self.stem = ConvBnRelu(in_planes, num_channels, kernel=3, padding=1)

        # Residual tower
        self.tower = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_blocks)])

        # Policy head: 1×1 conv (reduce channels) → flatten → FC
        self.policy_conv = ConvBnRelu(num_channels, 2, kernel=1, padding=0)
        self.policy_fc   = nn.Linear(2 * 8 * 8, policy_size)

        # Value head: 1×1 conv → flatten → FC → 3-class categorical (loss/draw/win)
        self.value_conv = ConvBnRelu(num_channels, 1, kernel=1, padding=0)
        self.value_fc1  = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2  = nn.Linear(256, 3)  # logits for [loss, draw, win]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        # x: (B, 144, 8, 8)
        out = self.stem(x)    # (B, 256, 8, 8)
        out = self.tower(out) # (B, 256, 8, 8)

        # Policy head
        p = self.policy_conv(out)                   # (B, 2, 8, 8)
        p = p.view(p.size(0), -1)                   # (B, 128)
        policy_logits = self.policy_fc(p)            # (B, 4672)

        # Value head: categorical over {loss, draw, win}
        # Returns (B, 3) logits. Caller converts to scalar via softmax.
        v = self.value_conv(out)                    # (B, 1, 8, 8)
        v = F.relu(self.value_fc1(v.view(v.size(0), -1)), inplace=True)
        value_logits = self.value_fc2(v)            # (B, 3) ∈ [-∞, +∞]

        return policy_logits, value_logits


# ------------------------------------------------------------------
# Wrapper that MCTS calls directly
# ------------------------------------------------------------------

class NeuralNetwork:
    """
    Wraps ChessNet with numpy I/O so MCTS doesn't touch PyTorch.
    This is the object you pass into MCTS(neural_net=...).
    """

    def __init__(self, model: ChessNet = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = (model if model is not None else ChessNet()).to(self.device)
        self.model.eval()

    def evaluate(
        self,
        env,                             # ChessGame instance
        legal_moves: list[chess.Move],
    ) -> tuple[dict[chess.Move, float], float]:
        """
        Returns:
          policy — dict mapping each legal move to a prior probability
          value  — float ∈ [-1, 1], position eval from current player's POV
        """
        board_tensor = self._encode(env.board)

        with torch.no_grad():
            policy_logits, value_logits = self.model(board_tensor)

        policy_np = policy_logits.squeeze(0).cpu().numpy()
        # Convert categorical value logits → scalar in [-1, 1]
        # E[value] = P(win) - P(loss)
        value_probs = torch.softmax(value_logits.squeeze(0), dim=0)
        value_np = (value_probs[2] - value_probs[0]).item()  # win_prob - loss_prob

        policy = policy_to_move_probs(policy_np, env.board)

        # Fallback: if encoding missed any legal move, give it uniform prior
        missing = [m for m in legal_moves if m not in policy]
        if missing:
            uniform = 1.0 / len(legal_moves)
            for m in missing:
                policy[m] = uniform

        return policy, value_np

    def _encode(self, board: chess.Board) -> torch.Tensor:
        planes = encode_board(board)                        # (18, 8, 8)
        t = torch.from_numpy(planes).unsqueeze(0)          # (1, 18, 8, 8)
        return t.to(self.device)

    def encode_batch(self, boards: list) -> torch.Tensor:
        planes = np.stack([encode_board(b) for b in boards])
        t = torch.from_numpy(planes)
        if self.device.type == "cuda":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def evaluate_batch_infer(self, batch: torch.Tensor):
        """GPU inference for both self-play and training."""
        with torch.no_grad():
            policy_logits, value_logits = self.model(batch)
        # Convert categorical value logits → scalar per position
        value_probs = torch.softmax(value_logits, dim=1)  # (B, 3)
        values = (value_probs[:, 2] - value_probs[:, 0])  # win_prob - loss_prob
        return policy_logits.cpu().numpy(), values.cpu().numpy()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()