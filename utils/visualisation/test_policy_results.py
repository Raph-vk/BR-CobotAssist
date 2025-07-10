#!/usr/bin/env python3
"""
Chunk‑prediction sanity check for **ACTPolicy** models (no CLI).

Directory layout the script expects ────────────────────────────────
```
~/tos_app_data/
    20250630_182048/              # ← latest training run (first‑level folder)
        Dataset/                  #   HDF5 episodes live here
            episode_*.hdf5
        Models/
            20250630_210815/      # ← most recent model snapshot
                config.json       #   saved Hydra/argparse config
                stats.pkl         #   normalisation statistics
                policy_best_epoch_485.ckpt   #   "best" checkpoint (pattern)
                policy_last.ckpt  #   etc…
```
The script automatically:
1. finds the *most‑recent* run under `~/tos_app_data`,
2. dives into the freshest `Models/*` subdir,
3. loads `config.json` + `stats.pkl`,
4. locates **the** checkpoint whose filename starts with
   `policy_best_epoch_`,
5. randomly samples a handful of episodes, and
6. visualises the model’s *chunk*‑length action predictions.

Edit the default arguments in `main()` to target a different base directory,
change the number of episodes, prediction horizon, or device.
"""

from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from modules.policy.act.detr.models import ACTPolicy  # Updated import path

# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────


def pre_process_joints(joints: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalise joint angles/positions using saved training statistics."""
    return (joints - stats["joint_pos_mean"]) / stats["joint_pos_std"]


def post_process_action(actions: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Undo normalisation to obtain real‑world joint commands."""
    print('stats', stats)
    return actions * stats["action_std"] + stats["action_mean"]


def most_recent_subdir(root: Path) -> Path:
    """Return the *newest* sub‑directory of *root* (by mtime)."""
    subs = [d for d in root.iterdir() if d.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No sub‑directories in {root}")
    return max(subs, key=os.path.getmtime)


def load_stats(stats_path: Path) -> Dict[str, np.ndarray]:
    with open(stats_path, "rb") as fp:
        return pickle.load(fp)


def load_policy(config_path: Path, device: torch.device) -> Tuple[ACTPolicy, dict]:
    """Instantiate **ACTPolicy** and load the *best* checkpoint in that folder."""
    with open(config_path, "r") as fp:
        cfg: dict = json.load(fp)

    policy_cfg = cfg["policy_config"]

    ckpt_dir = config_path.parent  # ← e.g. …/Models/20250630_210815
    best_ckpts = sorted(ckpt_dir.glob("policy_best_epoch_*.ckpt"))
    if not best_ckpts:
        raise FileNotFoundError(f"No policy_best_epoch_*.ckpt in {ckpt_dir}")
    ckpt_path = max(best_ckpts, key=os.path.getmtime)

    model = ACTPolicy(policy_cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval().to(device)
    print(f"Loaded checkpoint: {ckpt_path.relative_to(Path.home())}")
    return model, cfg

# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────


def _plot_episode(
    axes: np.ndarray,
    camera_names: List[str],
    frames: Dict[str, np.ndarray],
    t0: int,
    preds: np.ndarray,
    gt: np.ndarray,
    record_divisor: int = 4,
):
    """Plot camera frames (row 0) and per‑joint curves (rows 1‑7)."""
    n_cams = len(camera_names)
    chunk_size = preds.shape[0]

    # Pad ground-truth if it's shorter than chunk_size
    if gt.shape[0] < chunk_size:
        pad_width = chunk_size - gt.shape[0]
        gt = np.pad(gt, ((0, pad_width), (0, 0)), mode='edge')

    for cam_idx, cam in enumerate(camera_names):
        ax = axes[0, cam_idx] if n_cams > 1 else axes[0]
        ax.imshow(frames[cam][int(t0/record_divisor)].astype(np.uint8))
        ax.set_title(f"{cam}  |  t={t0}")
        ax.axis("off")

    x = np.arange(chunk_size)
    for j in range(7):
        for cam_idx in range(n_cams):
            ax = axes[j + 1, cam_idx] if n_cams > 1 else axes[j + 1]
            if cam_idx == 0:  # draw once; mirror to other columns for layout
                ax.plot(x, preds[:, j], "r.-", label="predicted")
                ax.plot(x, gt[:chunk_size, j], "b.--", label="ground‑truth")
                ax.legend(loc="best")
            ax.set_ylabel(f"joint {j}")
    for ax in axes[-1]:
        ax.set_xlabel("Δt in chunk")

# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_episode(
    h5: h5py.File,
    cam_names: List[str],
    policy: ACTPolicy,
    stats: Dict[str, np.ndarray],
    device: torch.device,
    chunk: int,
    record_divisor: int = 4,
):
    """Predict *chunk* future actions at a random t₀ inside one episode."""

    # ── find joint & action arrays (handles legacy vs new datasets) ──
    if "teachbot_positions" in h5:  # new
        robot_pos = h5["robot_positions"][()]
        master_pos = h5["teachbot_positions"][()]
        joints = robot_pos
        actions = master_pos

    

    if len(actions) < chunk:
        print("Episode too short → skip")
        return


    # Print the length of the saved images for each camera
    for cam in cam_names:
        if f"images/{cam}" in h5:
            num_frames = h5[f"images/{cam}/color"].shape[0]
            print(f"Camera {cam} has {num_frames} saved frames.")

    t0 = random.randint(0, num_frames - chunk - 1)

    record_divisor = 4  # e.g. 4 frames per joint state
    t0 = t0 * record_divisor

    print('joint positions at t₀:', joints[t0])

    # ── build model inputs ───────────────────────────────────────────
    joint_tensor = torch.tensor(
        pre_process_joints(joints[t0], stats), dtype=torch.float32, device=device
    )[None]

    stacked = []
    frame_cache: Dict[str, np.ndarray] = {}
    for cam in cam_names:
        if f"images/{cam}" in h5:  # new
            rgb_all = h5[f"images/{cam}/color"][()]
            d16_all = h5[f"images/{cam}/depth"][()]
            rgb = rgb_all[int(t0/record_divisor)]
            d16 = d16_all[int(t0/record_divisor)]
            # Store the full array for plotting
            frame_cache[cam] = rgb_all
        rgb_f = rgb.astype(np.float32) / 255.0
        d_f = d16.astype(np.float32) / 65535.0
        if d_f.ndim == 2:
            d_f = d_f[..., None]
        stacked.append(np.concatenate([rgb_f, d_f], axis=-1))

    img_tensor = (
        torch.from_numpy(np.stack(stacked))
        .permute(0, 3, 1, 2)  # (N, C, H, W)
        .unsqueeze(0)
        .to(device)
    )

    # ── forward pass ────────────────────────────────────────────────
    with torch.no_grad():
        out = policy(joint_tensor, img_tensor, None, None)
    if isinstance(out, (list, tuple)):
        out = out[0]
    
    print('first prediction pre process ', out[0])
    print('action_mean:', stats['action_mean'])
    print('action_std:', stats['action_std'])
    print('stats', stats)
    
    
    # preds = out.cpu().numpy().squeeze(0)
    preds = post_process_action(out.cpu().numpy().squeeze(0), stats)
    print('preds after pre process', preds[0])

    print(f"Predicted {len(preds)} actions for t₀={t0} in {h5.filename}")

    # ── compare & plot ─────────────────────────────────────────────
    gts = actions[t0 : t0 + chunk * record_divisor]
    n_cams = len(cam_names)
    fig, axes = plt.subplots(8, n_cams, figsize=(4 * n_cams, 20))
    plt.subplots_adjust(wspace=0.25, hspace=0.4)
    _plot_episode(axes, cam_names, frame_cache, t0, preds, gts, record_divisor)
    fig.suptitle(f"{Path(h5.filename).name}  |  t₀ = {t0}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



# ──────────────────────────────────────────────────────────────────────────────
# Main (edit defaults here)
# ──────────────────────────────────────────────────────────────────────────────


def main(
    data_root: Path = Path("~/tos_app_data").expanduser(),
    episodes: int = 5,
    chunk: int = 50,
    device_str: str = "cuda",
) -> None:
    device = torch.device(device_str)

    # Use hardcoded paths for specific model/dataset
    latest_run = Path("/home/teun/tos_app_data/converted_20250701_112712")
    latest_model_dir = Path("/home/teun/tos_app_data/converted_20250701_112712/20250701")

    config_path = latest_model_dir / "config.json"
    stats_path = latest_model_dir / "dataset_stats.pkl"  # Using stats.pkl instead of dataset_stats.pkl

    policy, cfg = load_policy(config_path, device)
    stats = load_stats(stats_path)

    cam_names: List[str] = cfg.get("camera_names", cfg["policy_config"]["camera_names"])
    dataset_dir = latest_run / "Dataset"

    episodes_files = sorted(Path(latest_run).glob("episode_*.hdf5"))
    if not episodes_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files in {latest_run}")

    sampled = random.sample(episodes_files, min(episodes, len(episodes_files)))
    for ep in sampled:
        with h5py.File(ep, "r") as h5:
            evaluate_episode(h5, cam_names, policy, stats, device, chunk, record_divisor=4)


if __name__ == "__main__":
    main()
