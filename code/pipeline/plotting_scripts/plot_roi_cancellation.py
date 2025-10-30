#!/usr/bin/env python3
"""
ROI cancellation analysis plots using short-window predictions.

Generates a two-panel figure:
 (a) Cancellation rate vs Δt for inside vs outside a circular ROI
 (b) Efficiency gap (inside - outside) vs Δt

Inputs:
 - window_dir: directory with real_events.(npy|npz) and pred_events_dt_XX.Xms.npy
 - tracker_csv: to interpolate center (c_x, c_y) over time
 - roi_radius_px: radius of the circular ROI in pixels

Outputs:
 - PNG figure written to --out
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def load_window_real_events(window_dir: str) -> np.ndarray:
    for name in ("real_events.npy", "real_events.npz"):
        path = os.path.join(window_dir, name)
        if os.path.isfile(path):
            if name.endswith(".npz"):
                return np.load(path)["real"].astype(np.float32)
            return np.load(path).astype(np.float32)
    raise FileNotFoundError(f"real_events.(npy|npz) not found in {window_dir}")


def load_tracker_series(path: str):
    df = pd.read_csv(path, header=0)
    try:
        # With headers
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
            t_s = df["timestamp"].to_numpy(np.float64)
            cx_s = df["center_x"].to_numpy(np.float64)
            cy_s = df["center_y"].to_numpy(np.float64)
            # prefer explicit omega column names if present
            for col in ("omega_circlefit_rad_s", "omega_rad_s", "theta_dot_rad_s"):
                if col in df.columns:
                    om_s = df[col].to_numpy(np.float64)
                    break
            else:
                raise KeyError("omega column not found in tracker CSV")
            return t_s, cx_s, cy_s, om_s
        else:
            raise KeyError("no headers")
    except Exception:
        # Fallback: no headers, parse as raw numeric columns with mixed delimiters
        df = pd.read_csv(path, header=None, engine="python", sep=r"[ ,]+", comment=None)
        arr = df.to_numpy(np.float64)
        # Heuristic mapping: 0: time(s), 1: cx, 2: cy, 5: omega(rad/s)
        if arr.shape[1] < 6:
            raise ValueError("Tracker CSV without headers must have at least 6 columns: t, cx, cy, ..., omega")
        t_s = arr[:, 0]
        cx_s = arr[:, 1]
        cy_s = arr[:, 2]
        om_s = arr[:, 5]
        # Ensure sorted by time
        order = np.argsort(t_s)
        return t_s[order], cx_s[order], cy_s[order], om_s[order]


def interp1(tq: np.ndarray, tx: np.ndarray, vx: np.ndarray) -> np.ndarray:
    return np.interp(tq, tx, vx, left=vx[0], right=vx[-1])


def list_dt_prediction_files(window_dir: str) -> List[Tuple[float, str]]:
    files = []
    for fn in os.listdir(window_dir):
        if fn.startswith("pred_events_dt_") and fn.endswith("ms.npy"):
            # name format pred_events_dt_XX.Xms.npy
            try:
                core = fn[len("pred_events_dt_"):-len("ms.npy")]
                dt_ms = float(core)
                files.append((dt_ms, os.path.join(window_dir, fn)))
            except Exception:
                continue
    files.sort(key=lambda x: x[0])
    return files


def cancel_events_time_aware(real_events: np.ndarray,
                             predicted_events: np.ndarray,
                             dt_seconds: float,
                             temporal_tolerance_ms: float,
                             spatial_tolerance_pixels: float) -> Tuple[np.ndarray, np.ndarray, int]:
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    if num_real == 0 or num_predicted == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0

    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    matched_real = np.zeros(num_real, dtype=bool)
    matched_predicted = np.zeros(num_predicted, dtype=bool)
    total_matches = 0

    pred_tree = cKDTree(predicted_events[:, :2])
    chunk_size = min(50000, num_real)

    for chunk_start in range(0, num_real, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_real)
        chunk_real = real_events[chunk_start:chunk_end]
        chunk_target_times = chunk_real[:, 3] + dt_seconds
        spatial_candidates_list = pred_tree.query_ball_point(chunk_real[:, :2], spatial_tolerance_pixels)

        for i, (real_event, target_time, spatial_candidates) in enumerate(zip(chunk_real, chunk_target_times, spatial_candidates_list)):
            real_idx = chunk_start + i
            if matched_real[real_idx] or len(spatial_candidates) == 0:
                continue

            spatial_candidates = np.array(spatial_candidates)
            available_candidates = spatial_candidates[~matched_predicted[spatial_candidates]]
            if len(available_candidates) == 0:
                continue

            candidate_times = predicted_events[available_candidates, 3]
            temporal_mask = np.abs(candidate_times - target_time) <= temporal_tolerance_s
            if not np.any(temporal_mask):
                continue

            final_candidates = available_candidates[temporal_mask]
            candidate_events = predicted_events[final_candidates]

            real_polarity = real_event[2]
            pred_polarities = candidate_events[:, 2]
            polarity_matches = (pred_polarities != real_polarity)
            if np.any(polarity_matches):
                valid_candidates = final_candidates[polarity_matches]
                valid_events = candidate_events[polarity_matches]
                distances = np.sqrt(np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1))
                best_candidate = valid_candidates[np.argmin(distances)]
                matched_real[real_idx] = True
                matched_predicted[best_candidate] = True
                total_matches += 1

    return ~matched_real, ~matched_predicted, total_matches


def compute_roi_masks(real_events: np.ndarray, t_s, cx_s, cy_s, radius_px: float) -> np.ndarray:
    if len(real_events) == 0:
        return np.zeros(0, dtype=bool)
    tt = real_events[:, 3].astype(np.float64)
    cx = interp1(tt, t_s, cx_s)
    cy = interp1(tt, t_s, cy_s)
    dx = real_events[:, 0].astype(np.float64) - cx
    dy = real_events[:, 1].astype(np.float64) - cy
    r = np.sqrt(dx*dx + dy*dy)
    return (r <= radius_px)


def run(window_dir: str,
        tracker_csv: str,
        roi_radius_px: float,
        eps_t_ms: float,
        eps_xy_px: float,
        out_path: str,
        single_panel: bool = True,
        csv_out: str = "",
        latex_out: str = ""):

    real = load_window_real_events(window_dir)
    dt_files = list_dt_prediction_files(window_dir)
    if not dt_files:
        raise RuntimeError("No pred_events_dt_*.npy files found in window_dir")

    t_s, cx_s, cy_s, om_s = load_tracker_series(tracker_csv)

    inside_mask = compute_roi_masks(real, t_s, cx_s, cy_s, roi_radius_px)
    outside_mask = ~inside_mask

    dts_ms = []
    cr_inside = []
    cr_outside = []

    for dt_ms, pred_path in dt_files:
        preds = np.load(pred_path).astype(np.float32)
        dt_s = float(dt_ms) * 1e-3

        # Compute CR for inside
        if inside_mask.any():
            r_in = real[inside_mask]
            m_r_in, _, _ = cancel_events_time_aware(r_in, preds, dt_s, eps_t_ms, eps_xy_px)
            cr_in = 100.0 * (len(r_in) - int(m_r_in.sum())) / float(len(r_in))
        else:
            cr_in = np.nan

        # Compute CR for outside
        if outside_mask.any():
            r_out = real[outside_mask]
            m_r_out, _, _ = cancel_events_time_aware(r_out, preds, dt_s, eps_t_ms, eps_xy_px)
            cr_out = 100.0 * (len(r_out) - int(m_r_out.sum())) / float(len(r_out))
        else:
            cr_out = np.nan

        dts_ms.append(dt_ms)
        cr_inside.append(cr_in)
        cr_outside.append(cr_out)

    dts_ms = np.array(dts_ms, dtype=np.float64)
    cr_inside = np.array(cr_inside, dtype=np.float64)
    cr_outside = np.array(cr_outside, dtype=np.float64)
    gap = cr_inside - cr_outside

    # Optional CSV export
    if csv_out:
        import csv
        os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
        with open(csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dt_ms", "cr_inside", "cr_outside", "gap"])
            for a, b, c, d in zip(dts_ms, cr_inside, cr_outside, gap):
                w.writerow([f"{a:.3f}", f"{b:.3f}", f"{c:.3f}", f"{d:.3f}"])
        print(f"Saved CSV: {csv_out}")

    # Optional LaTeX table export
    if latex_out:
        os.makedirs(os.path.dirname(latex_out) or ".", exist_ok=True)
        header = (
            "% Auto-generated ROI cancellation table\n"
            "\\begin{tabular}{lccc}\n"
            "  \\toprule\n"
            "  \\textbf{$\\Delta t$ (ms)} & \\textbf{ROI CR (\\%)} & \\textbf{Background CR (\\%)} & \\textbf{Gap (\\%)} \\\\n"
            "  \\midrule\n"
        )
        rows = []
        for a, b, c, d in zip(dts_ms, cr_inside, cr_outside, gap):
            rows.append(f"  {a:.1f} & {b:.1f} & {c:.1f} & {d:.1f} \\\n")
        footer = "\n  \\bottomrule\n\\end{tabular}\n"
        with open(latex_out, "w") as f:
            f.write(header)
            f.write("\n".join(rows))
            f.write(footer)
        print(f"Saved LaTeX: {latex_out}")

    if single_panel:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.plot(dts_ms, cr_inside, label="Inside ROI", lw=2.5, marker="o", ms=5)
        ax.plot(dts_ms, cr_outside, label="Outside ROI", lw=2.5, marker="s", ms=5)
        # Shade the gap to emphasize progression
        ymin = np.minimum(cr_inside, cr_outside)
        ymax = np.maximum(cr_inside, cr_outside)
        ax.fill_between(dts_ms, ymin, ymax, color="tab:orange", alpha=0.15, label="Gap (inside − outside)")
        ax.set_xlabel("Prediction horizon Δt (ms)")
        ax.set_ylabel("Cancellation rate (%)")
        ax.set_ylim(0, 105)
        ax.set_xlim(min(dts_ms)-0.5, max(dts_ms)+0.5)
        ax.grid(alpha=0.35, ls="--")
        ax.set_title("ROI cancellation progression vs Δt")
        ax.legend(loc="upper right")
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        ax = axes[0]
        ax.plot(dts_ms, cr_inside, label="Inside ROI", lw=2)
        ax.plot(dts_ms, cr_outside, label="Outside ROI", lw=2)
        ax.set_xlabel("Prediction horizon Δt (ms)")
        ax.set_ylabel("Cancellation rate (%)")
        ax.set_title("(a) Inside vs outside circular ROI")
        ax.grid(alpha=0.3, ls="--")
        ax.legend()

        ax2 = axes[1]
        ax2.plot(dts_ms, gap, color="tab:orange", lw=2)
        ax2.axhline(0, color="k", lw=1)
        ax2.set_xlabel("Prediction horizon Δt (ms)")
        ax2.set_ylabel("Efficiency gap (inside − outside) (%)")
        ax2.set_title("(b) Gap vs Δt")
        ax2.grid(alpha=0.3, ls="--")
        fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="ROI cancellation analysis plots")
    p.add_argument("--window_dir", required=True)
    p.add_argument("--tracker_csv", required=True)
    p.add_argument("--roi_radius_px", type=float, required=True)
    p.add_argument("--eps_t_ms", type=float, default=0.5)
    p.add_argument("--eps_xy_px", type=float, default=2.5)
    p.add_argument("--out", default="roi_cancellation.png")
    p.add_argument("--csv_out", default="", help="Optional CSV output path for dt vs CR data")
    p.add_argument("--latex_out", default="", help="Optional LaTeX tabular output path")
    p.add_argument("--two_panel", action="store_true", help="Use two subplots instead of single progression panel")
    args = p.parse_args()

    run(window_dir=args.window_dir,
        tracker_csv=args.tracker_csv,
        roi_radius_px=args.roi_radius_px,
        eps_t_ms=args.eps_t_ms,
        eps_xy_px=args.eps_xy_px,
        out_path=args.out,
        single_panel=(not args.two_panel),
        csv_out=args.csv_out,
        latex_out=args.latex_out)


if __name__ == "__main__":
    main()


