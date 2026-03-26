import os
import csv
import time
from collections import defaultdict
from typing import Optional


class MetricsLogger:
    """
    Lightweight CSV logger with optional TensorBoard and W&B backends.
    Works standalone even if neither is installed.
    """

    def __init__(
        self,
        log_dir: str            = "logs",
        run_name: Optional[str] = None,
        use_tensorboard: bool   = True,
        use_wandb: bool         = False,
        wandb_project: str      = "chess-alphazero",
    ):
        self.log_dir  = log_dir
        self.run_name = run_name or f"run_{int(time.time())}"
        self.run_dir  = os.path.join(log_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self._step_data: dict[int, dict] = defaultdict(dict)
        self._csv_path = os.path.join(self.run_dir, "metrics.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = None   # initialised on first write (unknown columns)

        # TensorBoard
        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=self.run_dir)
                print(f"TensorBoard logging → {self.run_dir}")
                print(f"  Run: tensorboard --logdir {log_dir}")
            except ImportError:
                print("TensorBoard not installed — pip install tensorboard")

        # W&B
        self._wb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=self.run_name)
                self._wb = wandb
                print(f"W&B logging → project: {wandb_project}")
            except ImportError:
                print("W&B not installed — pip install wandb")

    # ------------------------------------------------------------------
    # Core logging API
    # ------------------------------------------------------------------

    def log(self, step: int, **metrics):
        """
        Log any number of scalar metrics at a given step.

        Usage:
            logger.log(iteration,
                policy_loss=0.43, value_loss=0.12,
                win_rate=0.61, buffer_size=42000)
        """
        self._step_data[step].update(metrics)
        self._step_data[step]["step"] = step

        # TensorBoard
        if self._tb:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, global_step=step)

        # W&B
        if self._wb:
            self._wb.log({"step": step, **metrics})

        # CSV (write immediately so progress survives a crash)
        self._write_csv(step, metrics)

    def log_evaluation(self, step: int, stats: dict):
        """Convenience wrapper for evaluator stats dict."""
        self.log(
            step,
            eval_candidate_wins  = stats["candidate_wins"],
            eval_draws           = stats["draws"],
            eval_best_wins       = stats["best_wins"],
            eval_win_rate        = stats["win_rate"],
            eval_adj_win_rate    = stats["adjusted_win_rate"],
            eval_promoted        = int(stats["promoted"]),
        )

    def log_training(self, step: int, losses: dict):
        """Convenience wrapper for trainer losses dict."""
        self.log(
            step,
            loss_policy = losses.get("policy_loss", 0),
            loss_value  = losses.get("value_loss", 0),
            loss_total  = losses.get("total_loss", 0),
            lr          = losses.get("lr", 0),
        )

    def log_selfplay(self, step: int, num_positions: int, buffer_size: int):
        self.log(
            step,
            selfplay_positions_generated = num_positions,
            buffer_size                  = buffer_size,
        )

    def close(self):
        if self._tb:
            self._tb.close()
        if self._wb:
            self._wb.finish()
        self._csv_file.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_csv(self, step: int, metrics: dict):
        row = {"step": step, **metrics}

        # Initialise writer with columns from first row
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=list(row.keys()),
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()