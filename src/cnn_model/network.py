import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import chess

from .board_encoder import encode_board, TOTAL_PLANES
from .move_encoder import policy_to_move_probs, TOTAL_ACTIONS


# ------------------------------------------------------------------
# Building blocks (CNN — kept for backward compatibility)
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


class ChessNet(nn.Module):
    """
    AlphaZero-style CNN (kept for loading old checkpoints).

    Input:  (B, 144, 8, 8)
    Output: policy logits (B, 4672)  +  value logits (B, 3)
    """

    def __init__(
        self,
        in_planes:    int = TOTAL_PLANES,
        num_channels: int = 256,
        num_blocks:   int = 20,
        policy_size:  int = TOTAL_ACTIONS,
    ):
        super().__init__()

        self.stem = ConvBnRelu(in_planes, num_channels, kernel=3, padding=1)
        self.tower = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_blocks)])

        self.policy_conv = ConvBnRelu(num_channels, 2, kernel=1, padding=0)
        self.policy_fc   = nn.Linear(2 * 8 * 8, policy_size)

        self.value_conv = ConvBnRelu(num_channels, 1, kernel=1, padding=0)
        self.value_fc1  = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2  = nn.Linear(256, 3)

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
        out = self.stem(x)
        out = self.tower(out)

        p = self.policy_conv(out)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = self.value_conv(out)
        v = F.relu(self.value_fc1(v.view(v.size(0), -1)), inplace=True)
        value_logits = self.value_fc2(v)

        return policy_logits, value_logits


# ------------------------------------------------------------------
# Transformer architecture
# ------------------------------------------------------------------

class ChessTransformer(nn.Module):
    """
    Vision Transformer for chess position evaluation.

    Treats the 8x8 board as 64 tokens, each with a feature vector derived
    from the input planes at that square. Self-attention allows pieces to
    "see" each other regardless of distance — better than CNN for long-range
    interactions (rooks, bishops, queens).

    Input:  (B, 144, 8, 8)
    Output: policy logits (B, 4672)  +  value scalar (B, 1)
    """

    def __init__(
        self,
        in_planes:       int = TOTAL_PLANES,   # 144
        d_model:         int = 256,
        nhead:           int = 8,
        num_layers:      int = 8,
        dim_feedforward: int = 1024,
        dropout:         float = 0.1,
        policy_planes:   int = 73,             # move planes per source square
    ):
        super().__init__()
        self.d_model       = d_model
        self.policy_planes = policy_planes
        self.num_squares   = 64

        # Per-square input projection: 144 features → d_model
        self.input_proj = nn.Linear(in_planes, d_model)

        # Learned positional embedding for each of the 64 squares
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_squares, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Pre-norm transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),  # final norm after last layer
        )

        # Policy head: per-token prediction of 73 move planes
        # Output shape: (B, 64, 73) → reshape to (B, 4672)
        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, policy_planes),
        )

        # Value head: mean-pool → MLP → scalar tanh
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        # x: (B, 144, 8, 8)
        B = x.size(0)

        # Reshape: (B, 144, 8, 8) → (B, 64, 144)
        # Each of the 64 squares gets its 144-channel feature vector
        tokens = x.view(B, x.size(1), 64).permute(0, 2, 1)  # (B, 64, 144)

        # Project to d_model and add positional embedding
        tokens = self.input_proj(tokens) + self.pos_embed  # (B, 64, 256)

        # Transformer encoder
        tokens = self.encoder(tokens)  # (B, 64, 256)

        # Policy head: per-square move prediction
        policy = self.policy_head(tokens)                # (B, 64, 73)
        policy_logits = policy.reshape(B, -1)            # (B, 4672)

        # Value head: global mean pool → scalar
        global_repr = tokens.mean(dim=1)                 # (B, 256)
        value = self.value_head(global_repr)             # (B, 1)

        return policy_logits, value


# ------------------------------------------------------------------
# Wrapper that MCTS calls directly
# ------------------------------------------------------------------

class NeuralNetwork:
    """
    Wraps a chess model (ChessTransformer or ChessNet) with numpy I/O
    so MCTS doesn't touch PyTorch.
    """

    def __init__(self, model: nn.Module = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = (model if model is not None else ChessTransformer()).to(self.device)
        self.model.eval()
        # Detect whether model outputs scalar value or categorical logits
        # Use getattr to handle torch.compile() wrapping (OptimizedModule)
        raw = getattr(self.model, "_orig_mod", self.model)
        self._scalar_value = isinstance(raw, ChessTransformer)

    def evaluate(
        self,
        env,                             # ChessGame instance
        legal_moves: list[chess.Move],
    ) -> tuple[dict[chess.Move, float], float]:
        """
        Returns:
          policy — dict mapping each legal move to a prior probability
          value  — float in [-1, 1], position eval from current player's POV
        """
        board_tensor = self._encode(env.board)

        with torch.no_grad():
            policy_logits, value_out = self.model(board_tensor)

        policy_np = policy_logits.squeeze(0).cpu().numpy()

        if self._scalar_value:
            value_np = value_out.squeeze().item()
        else:
            # Legacy ChessNet: categorical → scalar
            value_probs = torch.softmax(value_out.squeeze(0), dim=0)
            value_np = (value_probs[2] - value_probs[0]).item()

        policy = policy_to_move_probs(policy_np, env.board)

        # Fallback: if encoding missed any legal move, give it uniform prior
        missing = [m for m in legal_moves if m not in policy]
        if missing:
            uniform = 1.0 / len(legal_moves)
            for m in missing:
                policy[m] = uniform

        return policy, value_np

    def _encode(self, board: chess.Board) -> torch.Tensor:
        planes = encode_board(board)                        # (144, 8, 8)
        t = torch.from_numpy(planes).unsqueeze(0)          # (1, 144, 8, 8)
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
            policy_logits, value_out = self.model(batch)

        if self._scalar_value:
            values = value_out.squeeze(-1)  # (B,)
        else:
            # Legacy ChessNet: categorical → scalar
            value_probs = torch.softmax(value_out, dim=1)
            values = value_probs[:, 2] - value_probs[:, 0]

        return policy_logits.cpu().numpy(), values.cpu().numpy()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
