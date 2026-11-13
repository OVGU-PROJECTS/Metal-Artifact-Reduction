# trainer_cnnmar.py
"""
Minimal trainer for Zhang & Yu CNN-MAR (patch-based).

Paper-aligned essentials:
- 3-channel input (uncorrected, BHC, LI) and 1-channel target (metal-free/reference). citeturn8view0
- 64x64 patches. citeturn7view2
- MSE loss (Frobenius norm squared averaged over samples). citeturn9view0
- Paper reports training stopping after 2000 iterations. citeturn7view2

The thesis configuration uses Adam with lr=1.66e-3, batch size 16, and 200 epochs.
"""
import math, json
import os
import random
import argparse
import yaml
from types import SimpleNamespace
from typing import Optional, List
from pathlib import Path
from mar_common.cases import list_case_dirs

def _load_cfg(name):
    """Load configs/<name>.yaml from the repo root (or $MAR_CONFIG_DIR) as a namespace."""
    root = Path(os.environ.get(
        "MAR_CONFIG_DIR",
        Path(__file__).resolve().parents[1] / "configs",
    ))
    data = yaml.safe_load((root / f"{name}.yaml").read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{name}.yaml must contain a mapping")
    return SimpleNamespace(**data)

import numpy as np
import tensorflow as tf
from data_pipline.data_gen import gen_cnnmar
from models import ZhangCNNMAR

from mar_common.cases import list_case_dirs
from training.checkpoints import build_checkpoint_callbacks
from training.callbacks import FullImageSSIMCallback

@tf.keras.utils.register_keras_serializable(package="cnnmar")
def l2_loss(y_true, y_pred):
    """Per-example squared L2/MSE loss on mu patches, matching the original code."""
    y_true = y_true[..., :1]
    y_pred = y_pred[..., :1]
    per_ex = tf.reduce_mean(tf.square(y_pred - y_true), axis=[1,2,3])
    return per_ex  # shape [B]


@tf.keras.utils.register_keras_serializable(package="cnnmar")
def rmse_loss(y_true, y_pred):
    """Backward-compatible alias for older saved-model metadata."""
    return l2_loss(y_true, y_pred)


def set_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_dataset(
    *,
    max_examples: int = -1,
    split: str,
    batch_size: int,
    trainset: int = 13000,
    valset: int = 1000,
    seed: Optional[int] = None,
    shuffle: bool = True,
    reshuffle_each_epoch: bool = True,
    train_root: str,
    val_root: str,
    test_root: str,
    patch: int = 64,
    strong_ratio: float = 0.6,
    stride: int = 8,
) -> tf.data.Dataset:
    """
    Returns a tf.data.Dataset yielding (X,y) where:
      X: [B, patch, patch, 3]
      y: [B, patch, patch, 1]
    """
    def _gen():
        yield from gen_cnnmar(
            max_examples=max_examples,
            batch_size=batch_size,
            data_t=split,
            seed=seed,
            shuffle=shuffle,
            reshuffle_each_epoch=reshuffle_each_epoch,
            train_root=train_root,
            val_root=val_root,
            test_root=test_root,
            patches_per_cycle=trainset if split == "train" else valset,
            patch=patch,
            strong_ratio=strong_ratio,
            stride=stride,
        )
    # The leading dimension is variable because an exact MATLAB-style sampling
    # cycle can end in a partial batch.
    out_sig = (
        tf.TensorSpec(shape=(None, patch, patch, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, patch, patch, 1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(_gen, output_signature=out_sig)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(
    *,
    lr: float = 1.66e-3,
) -> tf.keras.Model:
    """Instantiate, build, and compile the CNN-MAR model."""
    model = ZhangCNNMAR(nMC=3, nFeature=32, nKernel=3)
    _ = model(tf.zeros((1, 64, 64, 3), tf.float32))  # one forward pass to create the weights
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=l2_loss, metrics = [model.s_similarity])
    return model


def train_cnnmar(
    *,
    lr: float = 1.66e-3,
    batch_size: int = 32,
    trainset: int = 13000,
    valset: int = 1000,
    epochs = 300,
    initial_epoch=0,
    resume_weights=None,
    resume_model=None,
    seed: Optional[int] = None,
    train_ssim_cases: int = 500,
    out_dir: str = "./cnnmar_runs",
    train_root: str,
    val_root: str,
    test_root: str,
    patch: int = 64,
    strong_ratio: float = 0.6,
    stride: int = 8,
    
) -> tf.keras.callbacks.History:
    """
    Trains for a fixed number of iterations (steps). This mirrors the paper's "iterations"
    notion more directly than "epochs".

    - steps_per_epoch = iterations
    - epochs = 1
    """
    os.makedirs(out_dir, exist_ok=True)
    if seed is not None:
        set_seeds(seed)

    set_memory_growth()

    visible_gpus = tf.config.list_logical_devices("GPU")
    required_gpus = int(os.environ.get("CNNMAR_REQUIRE_GPUS", "0"))
    if len(visible_gpus) < required_gpus:
        raise RuntimeError(
            f"CNN-MAR requires {required_gpus} GPUs for this run, but TensorFlow "
            f"only sees {len(visible_gpus)}"
        )
    if len(visible_gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    replicas = strategy.num_replicas_in_sync
    print(f"TensorFlow distribution: {type(strategy).__name__}, replicas={replicas}")

    if batch_size % replicas != 0:
        raise ValueError(
            f"Global batch_size={batch_size} must be divisible by replicas={replicas}"
        )
    for split_name, sample_count in (("trainset", trainset), ("valset", valset)):
        final_batch = sample_count % batch_size or batch_size
        if final_batch % replicas != 0:
            raise ValueError(
                f"Final {split_name} batch has {final_batch} samples, which is not "
                f"divisible by replicas={replicas}"
            )

    train_ds = make_dataset(
        max_examples=-1,  # infinite
        split="train",
        batch_size=batch_size,
        trainset=trainset,
        valset=valset,
        # Match dataGen.m/cnn_train.m: select patches once, then globally
        # reshuffle that fixed patch set at every training epoch.
        seed=seed,
        shuffle=True,
        reshuffle_each_epoch=True,
        train_root=train_root,
        val_root=val_root,
        test_root=test_root,
        patch=patch,
        strong_ratio=strong_ratio,
        stride=stride,
    )

    val_ds = make_dataset(
        max_examples=-1,            # infinite validation stream too, controlled by validation_steps
        split="val",
        batch_size=batch_size,
        trainset=trainset,
        valset=valset,
        seed=seed,               # keep deterministic if you set a seed
        # cnn_train.m does not permute the validation subset.
        shuffle=False,
        reshuffle_each_epoch=False,
        train_root=train_root,
        val_root=val_root,
        test_root=test_root,
        patch=patch,
        strong_ratio=strong_ratio,
        stride=stride,
    )
    
    steps_per_epoch = math.ceil(trainset/batch_size)
    val_steps = math.ceil(valset/batch_size)
    
    # Distributed variables and optimizer slots must be created in the strategy scope.
    with strategy.scope():
        if resume_model:
            model = tf.keras.models.load_model(resume_model)
        else:
            model = build_model(lr=lr)
            if resume_weights:
                model.load_weights(resume_weights)
    
    save_dir = Path(out_dir) / "saved"
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_best, ckpt_best_full, ckpt_every50, tb, csv = build_checkpoint_callbacks(
        save_dir, steps_per_epoch
    )

    if train_ssim_cases < 1:
      raise ValueError("train_ssim_cases must be at least 1")

    train_ssim_order = list_case_dirs(train_root)[:trainset]

    random.Random(
        (seed if seed is not None else 0) + 50_000
    ).shuffle(train_ssim_order)

    train_ssim_order = train_ssim_order[
        :min(train_ssim_cases, len(train_ssim_order))
    ]

    full_image_ssim = FullImageSSIMCallback(
        train_ssim_order,
        list_case_dirs(val_root)[:valset],
    )


    cbs = [ckpt_best, ckpt_best_full, ckpt_every50, full_image_ssim, tb, csv]
    
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=cbs,
        verbose=1,
    )
    
  
    last_ep_weights_dir = save_dir / "last_ep_weights"
    last_ep_weights_dir.mkdir(parents=True, exist_ok=True)
    # 2) Also save weights-only (optional, smaller)
    model.save_weights(last_ep_weights_dir/ "cnnmar_weights.weights.h5")
    model.save(str(save_dir / "last_tf"), save_format="tf")
    # 3) Save config you need for inference (patch size, HU window, etc.)
    run_cfg = {
        "lr": lr,
        "batch_size": batch_size,
        "trainset": trainset,
        "valset": valset,
        "seed_val": seed,
        "seed_train": seed,
        "train_root": train_root,
        "val_root": val_root,
        "patch": patch,
        "strong_ratio": strong_ratio,
        "stride": stride,
        "train_ssim_cases": train_ssim_cases,
        "num_replicas": replicas,
    }
    (save_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))
    return hist


def train_one(*,
              model: tf.keras.Model,
              train_ds: tf.data.Dataset,
              val_ds: tf.data.Dataset,
              epochs: int,
              steps_per_epoch: int,
              validation_steps: int,
              callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
             ) -> tf.keras.callbacks.History:
    
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks or [],
        verbose=1
    )
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cnnmar",
                    help="Name of configs/<config>.yaml to load (or $MAR_CONFIG_DIR/<config>.yaml).")
    cfg = vars(_load_cfg(ap.parse_args().config))

    # train_cnnmar owns these keys; bh_li_dir is preprocessing-only (BHC_LI.py).
    _TRAIN_KEYS = {
        "train_root", "val_root", "test_root", "out_dir",
        "lr", "batch_size", "epochs", "initial_epoch",
        "resume_weights", "resume_model", "seed",
        "trainset", "valset", "train_ssim_cases",
        "patch", "strong_ratio", "stride",
    }
    _SAMPLER_DEFAULTS = {"stride": 8}
    _PREP_ONLY = {"bh_li_dir"}

    if "topk" in cfg:
        cfg.pop("topk")
        print("Ignoring obsolete cnnmar.yaml key 'topk'; using the original grid sampler.")
    if "rotations" in cfg:
        cfg.pop("rotations")
        print("Ignoring obsolete cnnmar.yaml key 'rotations'; augmentation is disabled.")
    for key, value in _SAMPLER_DEFAULTS.items():
        cfg.setdefault(key, value)
    extra = set(cfg) - _TRAIN_KEYS - _PREP_ONLY
    missing = _TRAIN_KEYS - set(cfg)
    if extra or missing:
        raise ValueError(f"cnnmar.yaml key mismatch: extra={sorted(extra)} missing={sorted(missing)}")

    train_cnnmar(**{k: cfg[k] for k in _TRAIN_KEYS})
'''
conda activate myenv
cd /mnt/public/bahaa/last_CT_MAR/CNN-MAR/cnn_mar

mkdir -p /mnt/public/bahaa/last_CT_MAR/CNN-MAR/results/final_training
python -m training.trainer --config cnnmar
'''