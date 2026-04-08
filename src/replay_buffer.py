import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Memory-efficient replay buffer.

    Savings vs naive float32 storage:
      State:  float32 (92,160 B for 144×8×8) → uint8 (23,040 B)  — 4× smaller
      Policy: float32 (18,688 B) → sparse dict (~240 B) — ~78× smaller

    Total per position: ~23,280 B instead of ~110,848 B
    500k capacity: ~11.6 GB instead of ~55 GB (still large — reduce capacity if needed)
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, policy: dict, value: float):
        """
        state  : (144, 8, 8) float32 — will be compressed to uint8
        policy : dict[int, float]   — sparse {action_index: probability}
        value  : float
        """
        # Compress: float32 binary planes → uint8 (values are always 0.0 or 1.0)
        state_u8 = state.astype(np.uint8)
        self.buffer.append((state_u8, policy, np.float32(value)))

    def push_game(self, trajectory: list[tuple]):
        for state, policy, value in trajectory:
            self.push(state, policy, value)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)

        # Decompress states back to float32 for the network
        state_arr = np.stack(states).astype(np.float32)          # (B, 144, 8, 8)

        # Reconstruct dense policy vectors from sparse dicts
        policy_arr = np.zeros((len(batch), 4672), dtype=np.float32)
        for i, p in enumerate(policies):
            for idx, prob in p.items():
                policy_arr[i, idx] = prob

        value_arr = np.array(values, dtype=np.float32)            # (B,)
        return state_arr, policy_arr, value_arr

    def __len__(self):
        return len(self.buffer)
