#!/usr/bin/env python3
"""
Batchwise fine resolution dt sweep analysis for one window.
Input: window directory of fine predictions.
Usage:
  python analyze_fine_resolution_dt.py /media/sumit/New\ Volume/fine_resolution_window/window_1_5.867s_to_10.867s
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Borrow the same run_cancellation_for_window and CR calcs from analyze_dt_cancellation
sys.path.append(str(Path(__file__).parent))
from analyze_dt_cancellation import run_cancellation_for_window, calculate_roi_cancellation_rate

BIN_MS = 5.0       # temporal tolerance (ms)
R_PIX = 2.0        # spatial tolerance (px)
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264

def analyze_fine_resolution_window(window_dir):
    window_dir = Path(window_dir)
    real_path = window_dir / "real_events.npy"
    if not real_path.exists():
        print(f"real_events.npy missing in {window_dir}")
        return
    real_array = np.load(real_path, mmap_mode='r')
    files = sorted(window_dir.glob("pred_events_dt_*.npy"))
    dt_values = []
    import re
    for f in files:
        m = re.search(r"dt_([0-9]+(?:\.[0-9]+)?)ms", f.name)
        if m:
            dt_values.append(float(m.group(1)))
    dt_values = sorted(set(dt_values))
    if not dt_values:
        print(f"No pred_events_dt_*.npy files found in {window_dir}")
        return
    results = []
    for dt_ms in dt_values:
        try:
            pred_file = window_dir / f"pred_events_dt_{dt_ms:04.1f}ms.npy"
            if not pred_file.exists():
                pred_file = window_dir / f"pred_events_dt_{dt_ms:02d}.0ms.npy"
            if not pred_file.exists():
                continue
            pred_array = np.load(str(pred_file), mmap_mode='r')
            rf = np.zeros((len(real_array), 1), dtype=np.float32)
            pf = np.ones((len(pred_array), 1), dtype=np.float32)
            combined_events = np.vstack([
                np.column_stack([real_array, rf]),
                np.column_stack([pred_array, pf])
            ])
            combined_events = combined_events[np.argsort(combined_events[:, 3])]
            residual_real, _, matched_pairs = run_cancellation_for_window(
                combined_events, BIN_MS, R_PIX)
            cr, total_roi_real, total_roi_cancelled = calculate_roi_cancellation_rate(
                combined_events, residual_real,
                (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS)
            print(f"dt={dt_ms:.3f} ms: CR={cr:.2f} matched_pairs={matched_pairs}")
            results.append({
                'dt_ms': float(dt_ms),
                'cancellation_rate': cr,
            })
            del pred_array, combined_events, residual_real
        except Exception as e:
            print(f"Failed at dt={dt_ms}: {repr(e)}")
    if not results:
        print("No valid dt results for fine window.")
        return
    # Sort and plot
    results.sort(key=lambda r: r['dt_ms'])
    xs = [r['dt_ms'] for r in results]
    ys = [r['cancellation_rate'] for r in results]
    plt.figure(figsize=(10,6))
    plt.plot(xs, ys, "o-", label="Cancellation Rate")
    plt.xlabel("dt (ms)")
    plt.ylabel("Cancellation Rate (%)")
    plt.title(f"Fine-Resolution Cancellation vs dt\n{window_dir.name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.legend()
    out_path = window_dir / "fine_resolution_dt_sweep.svg"
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_fine_resolution_dt.py <window_dir>")
        sys.exit(1)
    analyze_fine_resolution_window(sys.argv[1])

