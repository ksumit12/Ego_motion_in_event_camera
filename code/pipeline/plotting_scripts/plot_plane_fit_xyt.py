#!/usr/bin/env python3
"""
3D visualization of local motion estimation in event cameras via spatio-temporal plane fitting.

Inputs:
  - --events: Path to .npy/.npz events array. Expected columns: [x, y, p, t, flag] (flag optional: 0=real, 1=pred)
  - --csv: (Optional) Tracker CSV with columns x,y,t (not required for plane fit but accepted per interface)

Outputs:
  - Saves plane_fit_xyt.png (or --out) at 300 dpi

Figure:
  - 3D scatter of (x, y, t_ms)
  - Least-squares plane fit: t_ms ≈ v_x x + v_y y + d
  - Render plane surface, annotate (v_x, v_y, d), draw normal
  - Optional insets: x–t and y–t projections
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def load_events(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        data = np.load(path)
        # try common keys
        for k in ("events", "real", "combined"):
            if k in data:
                arr = data[k]
                break
        else:
            # default to first array
            k = list(data.keys())[0]
            arr = data[k]
    else:
        arr = np.load(path)
    arr = np.asarray(arr)
    # Expect at least [x, y, p, t]; combined may have 5th flag column
    if arr.shape[1] < 4:
        raise ValueError("Events array must have at least 4 columns: x,y,p,t")
    return arr.astype(np.float64)


def load_csv_points(csv_path: str) -> np.ndarray:
    """Load (x,y,t) points from a CSV. Accepts either columns (x,y,t) or (center_x,center_y,timestamp)."""
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if all(k in cols for k in ("x", "y", "t")):
        x = df[cols["x"]].to_numpy(np.float64)
        y = df[cols["y"]].to_numpy(np.float64)
        t = df[cols["t"]].to_numpy(np.float64)
    elif all(k in cols for k in ("center_x", "center_y", "timestamp")):
        x = df[cols["center_x"]].to_numpy(np.float64)
        y = df[cols["center_y"]].to_numpy(np.float64)
        t = df[cols["timestamp"]].to_numpy(np.float64)
    else:
        raise ValueError("CSV must have (x,y,t) or (center_x,center_y,timestamp) columns")
    t_ms = to_ms(t)
    p = np.zeros_like(x)
    arr = np.column_stack([x, y, p, t_ms])
    return arr


def to_ms(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    # Heuristic: if t is in seconds (max < 1e3), convert to ms; if in microseconds (max > 1e6), convert to ms
    t_abs = np.nanmax(t) - np.nanmin(t)
    if t_abs > 1e6:  # microseconds
        return t * 1e-3
    if t_abs < 1e3:  # seconds range
        return t * 1e3
    return t  # already ms


def select_local_patch(events: np.ndarray, max_points: int = 15000, patch_px: float = 100.0) -> np.ndarray:
    """Pick a local spatial patch around a dense region for a clean plane.
    Strategy: pick a random seed among real events, then take neighbors within a radius.
    Fallback: random subsample if density is too low.
    """
    xy = events[:, :2]
    t = events[:, 3]
    # prefer real events if a 5th flag exists
    if events.shape[1] >= 5:
        real_mask = (events[:, 4] == 0.0)
    else:
        real_mask = np.ones(len(events), dtype=bool)
    idx = np.flatnonzero(real_mask)
    if idx.size == 0:
        idx = np.arange(len(events))
    # seed near median time to avoid edge effects
    seed = idx[np.random.randint(0, idx.size)]
    seed_xy = xy[seed]
    d2 = np.sum((xy - seed_xy) ** 2, axis=1)
    patch_mask = d2 <= (patch_px ** 2)
    cand = np.flatnonzero(patch_mask & real_mask)
    if cand.size < 500:  # fallback to time band around seed
        dt = np.abs(t - t[seed])
        cand = np.flatnonzero((dt <= 5.0) & real_mask)  # 5 ms band
        if cand.size < 500:
            cand = np.arange(len(events))
    if cand.size > max_points:
        sel = np.random.choice(cand, size=max_points, replace=False)
    else:
        sel = cand
    return events[sel]


def fit_plane(x: np.ndarray, y: np.ndarray, t_ms: np.ndarray) -> Tuple[float, float, float]:
    A = np.column_stack([x, y, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(A, t_ms, rcond=None)
    vx, vy, d = beta.tolist()
    return vx, vy, d


def plot_plane_fit(events: np.ndarray, out_path: str, title: str = "Local Motion via Spatio-Temporal Plane Fitting in (x, y, t)") -> None:
    # Prepare data
    x = events[:, 0]
    y = events[:, 1]
    t_ms = to_ms(events[:, 3])

    # Fit plane
    vx, vy, d = fit_plane(x, y, t_ms)

    # Figure style
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(8.2, 6.2), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter
    ax.scatter(x, y, t_ms, s=8, c="#1f77b4", alpha=0.60, depthshade=False)

    # Plane surface over bounding box
    x_min, x_max = np.percentile(x, [2, 98])
    y_min, y_max = np.percentile(y, [2, 98])
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 20),
        np.linspace(y_min, y_max, 20),
    )
    tt = vx * xx + vy * yy + d
    ax.plot_surface(xx, yy, tt, color="#ff7f0e", alpha=0.35, linewidth=0, antialiased=True)

    # Normal vector annotation
    # plane: n = (vx, vy, -1) in (x,y,t_ms) space (up to scale)
    cx, cy, ct = np.mean(x), np.mean(y), np.mean(t_ms)
    n = np.array([vx, vy, -1.0])
    n_norm = n / (np.linalg.norm(n) + 1e-12)
    scale = 50.0  # arrow length in data units
    ax.quiver(cx, cy, ct, n_norm[0], n_norm[1], n_norm[2], length=scale, color="#d62728", linewidth=2)

    # Text box with parameters
    ax.text2D(0.02, 0.98, f"v_x = {vx:.4f} ms/px\n"
                        f"v_y = {vy:.4f} ms/px\n"
                        f"d   = {d:.4f} ms",
              transform=ax.transAxes,
              fontsize=10,
              va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    # Axes labels
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_zlabel("t (ms)")
    ax.set_title(title)

    # Annotation about slope encoding velocity
    ax.text2D(0.55, 0.98, r"Slope encodes local image velocity:\n$(v_x, v_y) = (\Delta x / \Delta t,\; \Delta y / \Delta t)$",
              transform=ax.transAxes, fontsize=9, va='top', ha='left')

    # Optional insets: x–t and y–t projections
    try:
        axins1 = inset_axes(ax, width="32%", height="28%", loc='lower left', bbox_to_anchor=(0.00, 0.05, 0.5, 0.5), bbox_transform=ax.transAxes)
        axins1.scatter(x, t_ms, s=3, c="#1f77b4", alpha=0.5)
        axins1.set_xlabel("x (px)", fontsize=8)
        axins1.set_ylabel("t (ms)", fontsize=8)
        axins1.set_title("x–t projection", fontsize=8)
        axins1.grid(alpha=0.2, linestyle='--')

        axins2 = inset_axes(ax, width="32%", height="28%", loc='lower left', bbox_to_anchor=(0.98, 0.05, 0.5, 0.5), bbox_transform=ax.transAxes)
        axins2.scatter(y, t_ms, s=3, c="#2ca02c", alpha=0.5)
        axins2.set_xlabel("y (px)", fontsize=8)
        axins2.set_ylabel("t (ms)", fontsize=8)
        axins2.set_title("y–t projection", fontsize=8)
        axins2.grid(alpha=0.2, linestyle='--')
    except Exception:
        pass

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor='white')


def main():
    p = argparse.ArgumentParser(description="3D plane fit in (x,y,t) for event motion estimation")
    p.add_argument("--events", default="./combined_events_with_predictions.npy",
                   help="Path to events .npy/.npz (expects columns x,y,p,t[,flag])")
    p.add_argument("--csv", default="",
                   help="Optional tracker CSV with columns x,y,t (not used for plane fit)")
    p.add_argument("--out", default="plane_fit_xyt.png")
    p.add_argument("--max_points", type=int, default=15000)
    p.add_argument("--patch_px", type=float, default=100.0,
                   help="Select a local spatial patch radius (px)")
    args = p.parse_args()

    ev: np.ndarray
    try:
        if args.events and os.path.exists(args.events):
            ev = load_events(args.events)
        elif args.csv and os.path.exists(args.csv):
            ev = load_csv_points(args.csv)
        else:
            raise FileNotFoundError("Neither events file nor CSV found")
    except Exception:
        # Fallback to CSV if provided
        if args.csv and os.path.exists(args.csv):
            ev = load_csv_points(args.csv)
        else:
            raise
    # Select a local patch to visualize a coherent plane
    ev_sel = select_local_patch(ev, max_points=args.max_points, patch_px=args.patch_px)
    plot_plane_fit(ev_sel, out_path=args.out)


if __name__ == "__main__":
    main()

