import os
import copy
import signal
import sys
import torch
from src.cnn_model.network import ChessNet, NeuralNetwork
from src.replay_buffer import ReplayBuffer
from src.parallel_self_play import ParallelSelfPlay
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.logger import MetricsLogger


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR    = "checkpoints"
LOG_DIR           = "logs"
RUN_NAME          = "alphazero_v1"

NUM_ITERATIONS    = 100
GAMES_PER_ITER    = 10
SIMULATIONS       = 800
BUFFER_CAPACITY   = 500_000
MIN_BUFFER_SIZE   = 10_000
BATCH_SIZE        = 512
EPOCHS_PER_ITER   = 5

USE_TENSORBOARD   = True
USE_WANDB         = False


# ------------------------------------------------------------------
# Graceful shutdown
# ------------------------------------------------------------------

class _ShutdownRequested(Exception):
    """Raised by the signal handler to break the training loop cleanly."""
    pass


def _make_signal_handler():
    """Returns a signal handler that raises _ShutdownRequested once."""
    triggered = False

    def handler(sig, frame):
        nonlocal triggered
        if triggered:
            # Second Ctrl+C — user is impatient, hard exit
            print("\n[!] Force quit.")
            os._exit(1)
        triggered = True
        print("\n[!] Interrupt received — finishing current step then shutting down...")
        raise _ShutdownRequested()

    return handler


def _save_checkpoint(net: NeuralNetwork, iteration: int, label: str = "interrupt"):
    """Save a checkpoint with a clear label so interrupted runs aren't lost."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"best_iter_{iteration:04d}_{label}.pt")
    net.save(path)
    print(f"    Checkpoint saved → {path}")
    return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Register signal handler before anything else
    signal.signal(signal.SIGINT,  _make_signal_handler())
    signal.signal(signal.SIGTERM, _make_signal_handler())

    logger = MetricsLogger(
        log_dir         = LOG_DIR,
        run_name        = RUN_NAME,
        use_tensorboard = USE_TENSORBOARD,
        use_wandb       = USE_WANDB,
    )

    best_model      = ChessNet()
    best_net        = NeuralNetwork(model=best_model, device=DEVICE)
    candidate_model = copy.deepcopy(best_model)
    candidate_net   = NeuralNetwork(model=candidate_model, device=DEVICE)

    buffer    = ReplayBuffer(capacity=BUFFER_CAPACITY)
    trainer   = Trainer(
        model             = candidate_model,
        device            = DEVICE,
        batch_size        = BATCH_SIZE,
        epochs_per_update = EPOCHS_PER_ITER,
    )
    evaluator = Evaluator(num_games=40, win_threshold=0.55, num_simulations=400)

    last_promoted_iteration = 0

    try:
        for iteration in range(1, NUM_ITERATIONS + 1):
            print(f"\n{'='*52}")
            print(f"  Iteration {iteration}/{NUM_ITERATIONS}")
            print(f"{'='*52}")

            # 1. Self-play
            print(f"\n[1] Self-play ({GAMES_PER_ITER} games)...")
            self_play = ParallelSelfPlay(neural_net=best_net, num_simulations=SIMULATIONS, num_parallel=40, leaves_per_game=10)
            positions = self_play.generate(num_games=GAMES_PER_ITER)

            buffer.push_game(positions)

            logger.log_selfplay(
                step          = iteration,
                num_positions = len(positions),
                buffer_size   = len(buffer),
            )
            print(f"    Buffer: {len(buffer):,} positions")

            if len(buffer) < MIN_BUFFER_SIZE:
                print(f"    Waiting for buffer to fill ({len(buffer)}/{MIN_BUFFER_SIZE})...")
                continue

            # 2. Train
            print(f"\n[2] Training...")
            candidate_model.load_state_dict(copy.deepcopy(best_model.state_dict()))
            losses = trainer.train_step(buffer)
            logger.log_training(step=iteration, losses=losses)
            print(
                f"    policy={losses['policy_loss']:.4f}  "
                f"value={losses['value_loss']:.4f}  "
                f"lr={losses['lr']:.2e}"
            )

            # 3. Evaluate
            print(f"\n[3] Evaluating...")
            promoted, stats = evaluator.evaluate(candidate_net, best_net)
            logger.log_evaluation(step=iteration, stats=stats)
            print(
                f"    {stats['candidate_wins']}W / {stats['draws']}D / {stats['best_wins']}L"
                f"  →  adj. win rate {stats['adjusted_win_rate']:.1%}"
            )

            # 4. Promote
            if promoted:
                print(f"    ✓ New best network promoted!")
                best_model.load_state_dict(copy.deepcopy(candidate_model.state_dict()))
                best_net = NeuralNetwork(model=best_model, device=DEVICE)
                _save_checkpoint(best_net, iteration, label="best")
                last_promoted_iteration = iteration
            else:
                print(f"    ✗ Candidate not promoted.")

    except _ShutdownRequested:
        # Clean interrupted run — save whatever the best network is right now
        print(f"\n[~] Saving best network before exit...")
        _save_checkpoint(best_net, last_promoted_iteration or 0, label="interrupt")

    except Exception as e:
        # Unexpected crash — still try to save and close logger cleanly
        print(f"\n[!] Unexpected error: {e}")
        _save_checkpoint(best_net, last_promoted_iteration or 0, label="crash")
        raise

    finally:
        # Always runs — whether normal exit, Ctrl+C, or crash
        print("\n[~] Closing logger...")
        logger.close()
        print("[~] Done.")


if __name__ == "__main__":
    main()