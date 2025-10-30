#!/usr/bin/env python3
"""
Auto-generate fine-resolution (0–3 ms) dt predictions over 5-second windows.

Differences vs create_short_windows_and_predictions.py:
- Window duration: 5.0 seconds (not 10 ms)
- dt sweep: fine resolution (0–2ms in 0.1ms, 2–3ms in 0.25ms)
- Output root: /media/sumit/New Volume/fine_resolution_window

Output layout
  /media/sumit/New Volume/fine_resolution_window/
    window_1_<t0>s_to_<t1>s/
      real_events.npy
      pred_events_dt_0.1ms.npy, pred_events_dt_0.2ms.npy, ...
      (optional) combined_events_dt_*.npy if --save-combined is set
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration (overridable via CLI)
# ------------------------------------------------------------
REAL_EVENTS_FILE = \
    "/media/sumit/New Volume/anu_research/recording/new_data/perlin_1280hz_hand_outframe.csv"
TRACKER_CSV_FILE = \
    "/media/sumit/New Volume/anu_research/ego_motion/results_csv/perlin_1280hz_hand_outframe_combined.csv"

OUTPUT_DIR = "/media/sumit/New Volume/fine_resolution_window"

WINDOW_DURATION_S = 5.0   # 5-second windows
NUM_WINDOWS = 1           # choose top-1 dense 5s window by default
MIN_GAP_S = 1.0           # spacing between selected windows

# Fine dt sweep
DT_VALUES_MS_PRIMARY = np.arange(0.0, 2.0 + 1e-9, 0.1)
DT_VALUES_MS_TAIL = np.arange(2.0, 3.0 + 1e-9, 0.25)

SAVE_COMPRESSED = False
SAVE_COMBINED = False
OMEGA_SOURCE = "circlefit"  # or "theta_dot"
OMEGA_BIAS = 0.0


# ------------------------------------------------------------
# Helpers: loading and interpolation
# ------------------------------------------------------------
def load_event_data_fast(path: str) -> np.ndarray:
    df = pd.read_csv(path, names=["x", "y", "p", "t"],
                     dtype={"x": np.float32, "y": np.float32, "p": np.float32, "t": np.float64})
    df["t"] = df["t"] * 1e-6  # µs → s
    ev = df[["x", "y", "p", "t"]].to_numpy(np.float32)
    if not np.all(ev[:-1, 3] <= ev[1:, 3]):
        ev = ev[np.argsort(ev[:, 3])]
    u = np.unique(ev[:, 2])
    if u.shape[0] == 2 and u.min() == -1.0 and u.max() == 1.0:
        ev[:, 2] = (ev[:, 2] + 1.0) * 0.5
    return ev


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


def extract_window_events(events: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    t0, t1 = window
    m = (events[:, 3] >= t0) & (events[:, 3] < t1)
    return events[m]


# ------------------------------------------------------------
# Motion model and prediction
# ------------------------------------------------------------
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
    pp = (1.0 - p).astype(np.float32)
    return np.column_stack([px, py, pp, pt])


def combine_window_events(real_events: np.ndarray, pred_events: np.ndarray) -> np.ndarray:
    if len(real_events) == 0 and len(pred_events) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    real_flag = np.zeros((len(real_events), 1), dtype=np.float32)
    pred_flag = np.ones((len(pred_events), 1), dtype=np.float32)
    if len(real_events) == 0:
        return np.column_stack([pred_events, pred_flag])
    if len(pred_events) == 0:
        return np.column_stack([real_events, real_flag])
    combined = np.vstack([
        np.column_stack([real_events, real_flag]),
        np.column_stack([pred_events, pred_flag])
    ])
    return combined[np.argsort(combined[:, 3])]


# ------------------------------------------------------------
# Window selection (dense 5s segments)
# ------------------------------------------------------------
def select_dense_windows(events: np.ndarray,
                         window_duration_s: float,
                         num_windows: int,
                         min_gap_s: float) -> List[Tuple[float, float]]:
    t = events[:, 3].astype(np.float64)
    if len(t) == 0:
        return []
    t0, t1 = float(t.min()), float(t.max())
    bin_dt = 0.010  # 10 ms bins for coarse density estimate over 5s windows
    nbins = int(np.ceil((t1 - t0) / bin_dt)) + 1
    hist, _ = np.histogram(t, bins=nbins, range=(t0, t0 + nbins * bin_dt))
    k = int(round(window_duration_s / bin_dt))
    if k <= 0:
        k = 1
    csum = np.cumsum(np.pad(hist, (1, 0)))
    window_counts = csum[k:] - csum[:-k]
    idxs = np.argsort(window_counts)[::-1]
    chosen = []
    chosen_times = []
    for idx in idxs:
        start_time = t0 + idx * bin_dt
        if all(abs(start_time - ct) >= min_gap_s for ct in chosen_times):
            chosen.append((start_time, start_time + window_duration_s))
            chosen_times.append(start_time)
            if len(chosen) >= num_windows:
                break
    return sorted(chosen)


# ------------------------------------------------------------
# Main processing
# ------------------------------------------------------------
def process_fine_windows(events_file: str,
                         tracker_file: str,
                         output_dir: str,
                         num_windows: int,
                         window_duration_s: float,
                         min_gap_s: float,
                         use_compressed: bool,
                         save_combined: bool):
    events = load_event_data_fast(events_file)
    t_s, cx_s, cy_s, om_s = load_tracker_series(tracker_file, source=OMEGA_SOURCE)

    os.makedirs(output_dir, exist_ok=True)

    windows = select_dense_windows(events, window_duration_s, num_windows, min_gap_s)
    if not windows:
        print("No windows selected - check data files.")
        return

    dt_values_ms = np.unique(np.concatenate([DT_VALUES_MS_PRIMARY, DT_VALUES_MS_TAIL]))

    for wi, (w0, w1) in enumerate(windows, 1):
        print(f"\nProcessing window {wi}/{len(windows)}: {w0:.3f}s to {w1:.3f}s")
        win_events = extract_window_events(events, (w0, w1))
        print(f"  Real events in window: {len(win_events):,}")
        if len(win_events) == 0:
            continue

        wdir = os.path.join(output_dir, f"window_{wi}_{w0:.3f}s_to_{w1:.3f}s")
        os.makedirs(wdir, exist_ok=True)

        real_path = os.path.join(wdir, "real_events." + ("npz" if use_compressed else "npy"))
        if use_compressed:
            np.savez_compressed(real_path, real=win_events)
        else:
            np.save(real_path, win_events)

        for dt_ms in dt_values_ms:
            dt_s = float(dt_ms) / 1000.0
            preds = predict_events_for_window(win_events, t_s, cx_s, cy_s, om_s, dt_s, OMEGA_BIAS)
            pred_path = os.path.join(wdir, f"pred_events_dt_{dt_ms:04.1f}ms." + ("npz" if use_compressed else "npy"))
            if use_compressed:
                np.savez_compressed(pred_path, pred=preds)
            else:
                np.save(pred_path, preds)
            if save_combined:
                comb = combine_window_events(win_events, preds)
                comb_path = os.path.join(wdir, f"combined_events_dt_{dt_ms:04.1f}ms." + ("npz" if use_compressed else "npy"))
                if use_compressed:
                    np.savez_compressed(comb_path, combined=comb)
                else:
                    np.save(comb_path, comb)

    print(f"\nDone. Fine-resolution windows saved to: {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Create fine-resolution (0–3ms) dt predictions over 5s windows")
    p.add_argument("--events", default=REAL_EVENTS_FILE)
    p.add_argument("--tracker", default=TRACKER_CSV_FILE)
    p.add_argument("--out", default=OUTPUT_DIR)
    p.add_argument("--num", type=int, default=NUM_WINDOWS)
    p.add_argument("--dur", type=float, default=WINDOW_DURATION_S)
    p.add_argument("--gap", type=float, default=MIN_GAP_S)
    p.add_argument("--compressed", action="store_true")
    p.add_argument("--save-combined", action="store_true")
    args = p.parse_args()

    process_fine_windows(events_file=args.events,
                         tracker_file=args.tracker,
                         output_dir=args.out,
                         num_windows=args.num,
                         window_duration_s=args.dur,
                         min_gap_s=args.gap,
                         use_compressed=args.compressed or SAVE_COMPRESSED,
                         save_combined=args.save_combined or SAVE_COMBINED)


if __name__ == "__main__":
    main()



