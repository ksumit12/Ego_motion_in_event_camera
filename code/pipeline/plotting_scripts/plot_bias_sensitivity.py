#!/usr/bin/env python3
"""
Plot robustness to angular-velocity bias (delta_omega) using short-window data.

Inputs:
- A window directory produced by create_short_windows_and_predictions.py
  containing real_events.(npy|npz)
- Tracker CSV used to interpolate center (c_x, c_y) and omega over time

Procedure:
1) Load the window's real events
2) For a sweep of bias values (delta_omega), regenerate predicted events with
   omega' = omega + delta_omega at a fixed dt
3) Run true temporal-gate cancellation for several (eps_t, eps_xy) pairs
4) Aggregate mean and std of CR over parameter combinations for each bias
5) Plot mean CR vs delta_omega with ±1 std shaded region
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


# ---------------------- IO helpers ----------------------
def load_window_real_events(window_dir: str) -> np.ndarray:
    """Load real events from a window directory (npy or npz)."""
    for name in ("real_events.npy", "real_events.npz"):
        path = os.path.join(window_dir, name)
        if os.path.isfile(path):
            if name.endswith(".npz"):
                return np.load(path)["real"].astype(np.float32)
            return np.load(path).astype(np.float32)
    raise FileNotFoundError(f"real_events.(npy|npz) not found in {window_dir}")


def load_tracker_series(path: str, source: str = "circlefit"):
    df = pd.read_csv(path).sort_values("timestamp")
    t_s = df["timestamp"].to_numpy(np.float64)
    cx_s = df["center_x"].to_numpy(np.float64)
    cy_s = df["center_y"].to_numpy(np.float64)
    if source == "theta_dot":
        col = "theta_dot_rad_s" if "theta_dot_rad_s" in df.columns else "omega_rad_s"
    else:
        col = "omega_circlefit_rad_s" if "omega_circlefit_rad_s" in df.columns else "omega_rad_s"
    om_s = df[col].to_numpy(np.float64)
    return t_s, cx_s, cy_s, om_s


def interp1(tq: np.ndarray, tx: np.ndarray, vx: np.ndarray) -> np.ndarray:
    return np.interp(tq, tx, vx, left=vx[0], right=vx[-1])


# ---------------------- Motion model ----------------------
def apply_rotation(x, y, cx, cy, omega, dt):
    theta = omega * dt
    c = np.cos(theta)
    s = np.sin(theta)
    dx = x - cx
    dy = y - cy
    x_new = cx + c * dx - s * dy
    y_new = cy + s * dx + c * dy
    return x_new.astype(np.float32), y_new.astype(np.float32)


def predict_events_for_window(real_events: np.ndarray,
                              t_s: np.ndarray,
                              cx_s: np.ndarray,
                              cy_s: np.ndarray,
                              om_s: np.ndarray,
                              dt: float,
                              omega_bias: float) -> np.ndarray:
    if len(real_events) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x = real_events[:, 0].astype(np.float64)
    y = real_events[:, 1].astype(np.float64)
    p = real_events[:, 2].astype(np.float32)
    tt = real_events[:, 3].astype(np.float64)

    cx = interp1(tt, t_s, cx_s)
    cy = interp1(tt, t_s, cy_s)
    om = interp1(tt, t_s, om_s)
    if omega_bias != 0.0:
        om = om + omega_bias

    px, py = apply_rotation(x, y, cx, cy, om, dt)
    pt = (tt + dt).astype(np.float32)
    pp = (1.0 - p).astype(np.float32)  # flip polarity
    return np.column_stack([px, py, pp, pt])


# ---------------------- Cancellation (true temporal gate) ----------------------
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
            polarity_matches = (pred_polarities != real_polarity)  # opposite polarity
            if np.any(polarity_matches):
                valid_candidates = final_candidates[polarity_matches]
                valid_events = candidate_events[polarity_matches]
                distances = np.sqrt(np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1))
                best_candidate = valid_candidates[np.argmin(distances)]
                matched_real[real_idx] = True
                matched_predicted[best_candidate] = True
                total_matches += 1

    return ~matched_real, ~matched_predicted, total_matches


def compute_cr(real_events: np.ndarray,
               pred_events: np.ndarray,
               dt_s: float,
               eps_t_ms: float,
               eps_xy_px: float) -> float:
    mask_r, mask_p, total_matches = cancel_events_time_aware(real_events, pred_events, dt_s, eps_t_ms, eps_xy_px)
    if len(real_events) == 0:
        return 0.0
    cancelled = len(real_events) - int(mask_r.sum())
    return 100.0 * cancelled / float(len(real_events))


# ---------------------- Main plotting ----------------------
def run(window_dir: str,
        tracker_csv: str,
        omega_source: str,
        dt_ms: float,
        eps_t_list: List[float],
        eps_xy_list: List[float],
        bias_min: float,
        bias_max: float,
        bias_step: float,
        output_path: str):

    real = load_window_real_events(window_dir)
    t_s, cx_s, cy_s, om_s = load_tracker_series(tracker_csv, source=omega_source)

    dt_s = dt_ms * 1e-3
    biases = np.arange(bias_min, bias_max + 1e-9, bias_step, dtype=np.float64)

    mean_cr = []
    std_cr = []

    for b in biases:
        preds = predict_events_for_window(real, t_s, cx_s, cy_s, om_s, dt_s, omega_bias=b)

        cr_vals = []
        for et in eps_t_list:
            for exy in eps_xy_list:
                cr_vals.append(compute_cr(real, preds, dt_s, et, exy))
        cr_vals = np.array(cr_vals, dtype=np.float64)
        mean_cr.append(cr_vals.mean() if cr_vals.size else 0.0)
        std_cr.append(cr_vals.std(ddof=0) if cr_vals.size else 0.0)

    mean_cr = np.array(mean_cr)
    std_cr = np.array(std_cr)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(biases, mean_cr, color="#2E86AB", lw=2, label="mean CR")
    ax.fill_between(biases, mean_cr - std_cr, mean_cr + std_cr, color="#2E86AB", alpha=0.2, label="±1 std")
    ax.axhline(50.0, color="tab:red", lw=1.2, ls="--", alpha=0.8, label="50%")
    ax.axvspan(-0.05, 0.05, color="gray", alpha=0.08, label="linear regime (~±0.05 rad/s)")

    ax.set_xlabel("Angular velocity bias Δω (rad/s)")
    ax.set_ylabel("Cancellation rate (%)")
    ax.set_title("Robustness to motion estimation bias")
    ax.grid(alpha=0.3, ls="--")
    ax.legend()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Plot bias sensitivity using short-window prediction")
    p.add_argument("--window_dir", required=True, help="Path to window_X_t0s_to_t1s directory")
    p.add_argument("--tracker_csv", required=True, help="Tracker CSV used for circle fitting / omega")
    p.add_argument("--omega_source", default="circlefit", choices=["circlefit", "theta_dot"], help="Omega column source")
    p.add_argument("--dt_ms", type=float, default=2.0, help="Prediction horizon Δt in ms")
    p.add_argument("--eps_t_ms", type=float, nargs="*", default=[0.5, 1.0, 1.5], help="Temporal tolerance list (ms)")
    p.add_argument("--eps_xy_px", type=float, nargs="*", default=[2.0, 3.0], help="Spatial tolerance list (px)")
    p.add_argument("--bias_min", type=float, default=-0.08)
    p.add_argument("--bias_max", type=float, default=0.08)
    p.add_argument("--bias_step", type=float, default=0.01)
    p.add_argument("--out", default="bias_sensitivity.png")
    args = p.parse_args()

    run(window_dir=args.window_dir,
        tracker_csv=args.tracker_csv,
        omega_source=args.omega_source,
        dt_ms=args.dt_ms,
        eps_t_list=args.eps_t_ms,
        eps_xy_list=args.eps_xy_px,
        bias_min=args.bias_min,
        bias_max=args.bias_max,
        bias_step=args.bias_step,
        output_path=args.out)


if __name__ == "__main__":
    main()






