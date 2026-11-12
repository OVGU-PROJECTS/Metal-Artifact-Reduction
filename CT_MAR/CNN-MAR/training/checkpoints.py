# checkpoints.py
"""Checkpoint and logging callbacks for the CNN-MAR trainer.

Extracted from trainer.py to keep the trainer lean. The core logic is unchanged:
- best.weights.h5     : best weights-only checkpoint (monitor val_loss).
- best_tf             : best full-model checkpoint (monitor val_loss).
- milestones/epoch_*  : full model saved every 50 epochs.
- tensorboard/tb      : TensorBoard logs.
- csv/history.csv     : per-epoch metrics CSV.
"""
from pathlib import Path
from typing import Tuple

import tensorflow as tf


def build_checkpoint_callbacks(
    save_dir: Path,
    steps_per_epoch: int,
) -> Tuple[
    tf.keras.callbacks.Callback,
    tf.keras.callbacks.Callback,
    tf.keras.callbacks.Callback,
    tf.keras.callbacks.Callback,
    tf.keras.callbacks.Callback,
]:
    """Create the checkpoint + logging callbacks used during training.

    Returns (ckpt_best, ckpt_best_full, ckpt_every50, tb, csv).
    """
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(weights_dir / "best.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )
    ckpt_best_full = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(save_dir / "best_tf"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
    )
    milestones_dir = save_dir / "milestones"
    milestones_dir.mkdir(parents=True, exist_ok=True)
    ckpt_every50 = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(milestones_dir / "epoch_{epoch:04d}"),
        save_freq=50 * steps_per_epoch,   # save_freq counts BATCHES, so ×steps = every 50 epochs
        save_weights_only=False,          # full model: weights + optimizer
    )

    tb_dir = save_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb = tf.keras.callbacks.TensorBoard(
            log_dir= str (tb_dir / "tb"),
            update_freq=100,
            profile_batch=0,
            histogram_freq=0,
            write_graph=False
        )

    csv_dir = save_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv = tf.keras.callbacks.CSVLogger(str(csv_dir/ "history.csv"), append=False)

    return ckpt_best, ckpt_best_full, ckpt_every50, tb, csv
