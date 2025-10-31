#!/usr/bin/env python3
"""
Generate consolidated figures for the thesis from existing analysis outputs:

Outputs (PNG+SVG, 300 DPI):
  1) Multi-window cancellation vs dt (standard 0–20 ms sweep)
  2) Early drop-off (0–3 ms, fine resolution) with pixel-displacement axis
  3) Flow magnitude maps (dt ∈ {1.0, 3.0} ms)

Heavy whole-sequence analysis is intentionally skipped.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree

# Reuse helpers from the standard dt sweep implementation in this repository
# Make local package importable when run as a script
import sys
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analyze_dt_cancellation import (  # type: ignore
    discover_window_dirs,
    try_load_combined_or_split,
    run_cancellation_for_window,
    calculate_roi_cancellation_rate,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Primary data locations
WINDOW_PREDICTIONS_DIR = "/media/sumit/New Volume/short_window_predictions"  # dt=1..20ms
WINDOW_PREDICTIONS_DIR_FINE = "/media/sumit/New Volume/fine_resolution_window"  # HARDCODED per user

# Cancellation tolerances (match scripts used in thesis)
BIN_MS = 5.0       # temporal tolerance (ms)
R_PIX = 2.0        # spatial tolerance (px)

# Motion parameters for pixel-displacement overlays
OMEGA_RAD_S = 3.612     # mean angular velocity from tracker
MEAN_RADIUS_PX = 199    # mean event radius used for pixel axis

# Disc geometry (from experiments)
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264

# Output directories
OUT_DIR_ROOT = Path("code/thesis_figures")
OUT_DIR_SHORT = OUT_DIR_ROOT / "short_window_plots"
OUT_DIR_FINE = OUT_DIR_ROOT / "fine_resolution_plots"
OUT_DIR_SHORT.mkdir(parents=True, exist_ok=True)
OUT_DIR_FINE.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def create_windowwise_dt_sweep():
    """Create per-window figures and one combined figure (0–20 ms)."""
    dt_values_ms = np.arange(0, 21, 1)

    discovered = discover_window_dirs(WINDOW_PREDICTIONS_DIR)
    if not discovered:
        print(f"No window directories found under {WINDOW_PREDICTIONS_DIR}")
        return

    all_results = []
    for window_idx, (window_dir, label) in enumerate(discovered):
        print(f"Analyzing (standard sweep) {label}")
        results = []
        for dt_ms in dt_values_ms:
            # Prefer existing combined; otherwise synthesize from real+pred
            combined_events = None
            try:
                combined_events = try_load_combined_or_split(window_dir, dt_ms)
            except FileNotFoundError:
                real_path = Path(window_dir) / "real_events.npy"
                pred_path = Path(window_dir) / f"pred_events_dt_{dt_ms:02d}.0ms.npy"
                # Also try fractional naming (unlikely here, but safe)
                if not pred_path.exists():
                    pred_path = Path(window_dir) / f"pred_events_dt_{float(dt_ms):04.1f}ms.npy"
                if real_path.exists() and pred_path.exists():
                    real = np.load(real_path)
                    pred = np.load(pred_path)
                    rf = np.zeros((len(real), 1), dtype=np.float32)
                    pf = np.ones((len(pred), 1), dtype=np.float32)
                    combined_events = np.vstack([
                        np.column_stack([real, rf]),
                        np.column_stack([pred, pf])
                    ])
                    combined_events = combined_events[np.argsort(combined_events[:, 3])]
                else:
                    continue
            residual_real, _, matched_pairs = run_cancellation_for_window(
                combined_events, BIN_MS, R_PIX
            )
            cr, total_roi_real, total_roi_cancelled = calculate_roi_cancellation_rate(
                combined_events, residual_real, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS
            )
            results.append({
                "dt_ms": dt_ms,
                "cancellation_rate": cr,
                "total_roi_real": total_roi_real,
                "total_roi_cancelled": total_roi_cancelled,
                "total_matched_pairs": matched_pairs,
            })
        all_results.append((label, results))

        # Per-window figure
        if results:
            xs = [r['dt_ms'] for r in results]
            ys = [r['cancellation_rate'] for r in results]
            counts = [r['total_roi_real'] for r in results]

            fig_w, ax_w = plt.subplots(figsize=(8, 6))
            ax_w.plot(xs, ys, 'o-', linewidth=2, color='tab:blue')
            ax_w.set_xlabel('dt (ms)')
            ax_w.set_ylabel('Cancellation Rate (%)')
            ax_w.set_ylim(30, 101)
            ax_w.grid(True, alpha=0.3)
            ax2 = ax_w.twinx()
            ax2.bar(xs, counts, alpha=0.15, color='tab:blue')
            ax2.set_ylabel('Total ROI Events')
            ax_w.set_title(f"{label.replace('window_', 'Window ').replace('_', ' ')}")
            fig_w.tight_layout()
            out_name = f"dt_sweep_{label}.svg"
            fig_w.savefig(OUT_DIR_SHORT / out_name, dpi=300, bbox_inches='tight')
            plt.close(fig_w)

    # Combined figure (only the curves)
    if all_results:
        fig_c, axc = plt.subplots(figsize=(9, 7))
        # Define distinct colors for each window
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        for idx, (label, results) in enumerate(all_results):
            if not results:
                continue
            xs = [r['dt_ms'] for r in results]
            ys = [r['cancellation_rate'] for r in results]
            color = colors[idx % len(colors)]  # Cycle through colors if more windows than colors
            axc.plot(xs, ys, 'o-', linewidth=2, color=color, label=f"{label.split('_',1)[-1]}")
        axc.set_xlabel('dt (ms)')
        axc.set_ylabel('Cancellation Rate (%)')
        axc.set_ylim(30, 101)
        axc.grid(True, alpha=0.3)
        axc.set_title('Combined: All Windows')
        axc.legend()
        fig_c.tight_layout()
        fig_c.savefig(OUT_DIR_SHORT / 'windows_dt_sweep.svg', dpi=300, bbox_inches='tight')
        plt.close(fig_c)
        print(f"Saved: {OUT_DIR_SHORT / 'windows_dt_sweep.svg'}")


def early_dropoff_fine_resolution():
    """Create early drop-off plot (0–3 ms, fine resolution) + pixel axis."""
    discovered = discover_window_dirs(WINDOW_PREDICTIONS_DIR_FINE)
    if not discovered:
        print(f"No fine-resolution windows found under {WINDOW_PREDICTIONS_DIR_FINE}")
        return

    # Use the first available window directory (or iterate all)
    window_dir, label = discovered[0]

    # Auto-detect available fractional dt files inside the window folder
    candidates = []
    for ext in ("npy", "npz"):
        # Prefer combined; if absent, look for pred files
        candidates.extend(sorted(Path(window_dir).glob(f"combined_events_dt_*ms.{ext}")))
        candidates.extend(sorted(Path(window_dir).glob(f"pred_events_dt_*ms.{ext}")))
    import re
    dt_values_ms = []
    for p in candidates:
        m = re.search(r"combined_events_dt_([0-9]+(?:\.[0-9]+)?)ms\.(?:npy|npz)$", p.name)
        if m:
            val = float(m.group(1))
            if 0.0 <= val <= 3.0:
                dt_values_ms.append(val)
    dt_values_ms = sorted(set(dt_values_ms))

    results = []
    # Local loader that supports fractional dt file names (e.g., 0.1ms → 00.1)
    def try_load_combined_or_split_fine(window_dir: str, dt_ms_float: float):
        # Try fractional-named combined first
        for ext in ("npy", "npz"):
            p = os.path.join(window_dir, f"combined_events_dt_{dt_ms_float:04.1f}ms.{ext}")
            if os.path.exists(p):
                if p.endswith('.npz'):
                    with np.load(p) as z:
                        key = 'combined' if 'combined' in z.files else list(z.files)[0]
                        return z[key]
                return np.load(p, mmap_mode='r')
        # If no combined, synthesize from real+pred for fractional dt
        real_npy = os.path.join(window_dir, "real_events.npy")
        pred_npy = os.path.join(window_dir, f"pred_events_dt_{dt_ms_float:04.1f}ms.npy")
        if os.path.exists(real_npy) and os.path.exists(pred_npy):
            real = np.load(real_npy)
            pred = np.load(pred_npy)
            rf = np.zeros((len(real), 1), dtype=np.float32)
            pf = np.ones((len(pred), 1), dtype=np.float32)
            combined = np.vstack([
                np.column_stack([real, rf]),
                np.column_stack([pred, pf])
            ])
            return combined[np.argsort(combined[:, 3])]
        # Fall back to integer-named format from the standard helper
        return try_load_combined_or_split(window_dir, int(round(dt_ms_float)))

    for dt_ms in dt_values_ms:
        try:
            combined_events = try_load_combined_or_split_fine(window_dir, float(dt_ms))
        except FileNotFoundError:
            continue
        residual_real, _, matched_pairs = run_cancellation_for_window(
            combined_events, BIN_MS, R_PIX
        )
        cr, total_roi_real, total_roi_cancelled = calculate_roi_cancellation_rate(
            combined_events, residual_real, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS
        )
        results.append({
            'dt_ms': float(dt_ms),
            'cancellation_rate': cr,
            'total_roi_real': total_roi_real,
            'total_matched_pairs': matched_pairs,
        })

    if not results:
        print("No results for fine-resolution analysis")
        return

    results.sort(key=lambda r: r['dt_ms'])
    xs = [r['dt_ms'] for r in results]
    ys = [r['cancellation_rate'] for r in results]
    match_rate = [r['total_matched_pairs'] / r['total_roi_real'] * 100 if r['total_roi_real'] > 0 else 0 for r in results]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(xs, ys, 'o-', color='tab:blue', linewidth=2, markersize=4, label='Cancellation Rate')
    ax.axhspan(90, 100, alpha=0.1, color='green', label='≥90% Region')

    # Threshold lines (first crossing below 90% and 80%)
    arr = np.array(ys)
    idx90 = np.where(arr < 90)[0]
    t90 = xs[idx90[0]] if len(idx90) else None
    if t90 is not None:
        ax.axvline(t90, color='orange', linestyle='--', alpha=0.7, label=f'First dt < 90%: {t90:.1f}ms')

    ax2 = ax.twinx()
    ax2.plot(xs, match_rate, 'r--', alpha=0.6, linewidth=1, label='Match Rate (any match)')
    ax2.set_ylabel('Match Rate (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.set_xlim(0, 3)
    ax.set_ylim(70, 100)
    ax.set_xlabel('Prediction Horizon Δt (ms)')
    ax.set_ylabel('Cancellation Rate (%)')
    ax.set_title('Ego-Motion Cancellation: Early Drop-off (fine resolution)')
    ax.grid(True, alpha=0.3)

    # Secondary x-axis: pixel displacement
    ax3 = ax.twiny()
    ax3.set_xlim(0, 3)
    ax3.set_xlabel('Predicted Pixel Displacement (px)')
    dt_range = np.linspace(0, 3, 100)
    px_range = dt_range * OMEGA_RAD_S * MEAN_RADIUS_PX / 1000.0
    ax3.plot(px_range, [100]*len(px_range), alpha=0)

    if t90 is not None:
        t90_px = t90 * OMEGA_RAD_S * MEAN_RADIUS_PX / 1000.0
        ax.text(0.02, 0.98, f'First dt < 90%: {t90:.1f}ms ≈ {t90_px:.1f}px',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(loc='upper right')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    fig.savefig(OUT_DIR_FINE / 'early_dropoff_fine.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_DIR_FINE / 'early_dropoff_fine.svg'}")


def fine_resolution_dt_sweep():
    """Generate cancellation vs dt for fine-resolution directory (0–3ms, batch-wise)."""
    discovered = discover_window_dirs(WINDOW_PREDICTIONS_DIR_FINE)
    if not discovered:
        print(f"No fine-resolution windows found under {WINDOW_PREDICTIONS_DIR_FINE}")
        return

    window_dir, label = discovered[0]
    # Find dt values as before
    candidates = []
    for ext in ("npy", "npz"):
        candidates.extend(sorted(Path(window_dir).glob(f"pred_events_dt_*ms.{ext}")))
        candidates.extend(sorted(Path(window_dir).glob(f"combined_events_dt_*ms.{ext}")))
    import re
    dt_values_ms = []
    for p in candidates:
        m = re.search(r"_(?:events|pred)_dt_([0-9]+(?:\.[0-9]+)?)ms\.(?:npy|npz)$", p.name)
        if m:
            val = float(m.group(1))
            if 0.0 <= val <= 5.0:
                dt_values_ms.append(val)
    dt_values_ms = sorted(set(dt_values_ms))
    if not dt_values_ms:
        print("No dt candidates found for fine sweep")
        return

    # Load real events just once
    real_path_np = Path(window_dir) / "real_events.npy"
    if not real_path_np.exists():
        print(f"No real_events.npy found in {window_dir}")
        return
    real_array = np.load(real_path_np, mmap_mode='r')

    results = []
    for dt_ms in dt_values_ms:
        try:
            pred_file = Path(window_dir) / f"pred_events_dt_{dt_ms:04.1f}ms.npy"
            if not pred_file.exists():
                pred_file = Path(window_dir) / f"pred_events_dt_{dt_ms:02d}.0ms.npy"
            if not pred_file.exists():
                continue
            pred_array = np.load(str(pred_file), mmap_mode='r')
            # Synthesize combined array in memory, then immediate process and discard
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
                combined_events, residual_real, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS)
            results.append({
                'dt_ms': float(dt_ms),
                'cancellation_rate': cr,
            })
            # Explicitly clear to minimize memory
            del pred_array, combined_events, residual_real
        except Exception as e:
            print(f"Failed at dt={dt_ms}: {repr(e)}")
            continue

    if not results:
        print("No results for fine-resolution dt sweep")
        return

    results.sort(key=lambda r: r['dt_ms'])
    xs = [r['dt_ms'] for r in results]
    ys = [r['cancellation_rate'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, ys, 'o-', linewidth=2, color='tab:blue')
    ax.set_xlabel('dt (ms)')
    ax.set_ylabel('Cancellation Rate (%)')
    ax.set_title(f'Fine-Resolution Cancellation vs dt — {label}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR_FINE / 'fine_windows_dt_sweep.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_DIR_FINE / 'fine_windows_dt_sweep.svg'}")


def flow_magnitude_maps():
    """Create 2-panel flow magnitude maps for dt = 1.0, 3.0 ms with a central colorbar."""
    height, width = 720, 1280
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    distances = np.sqrt((x_coords - DISC_CENTER_X)**2 + (y_coords - DISC_CENTER_Y)**2)

    dt_values = [1.0, 3.0]
    disp_all = [distances * OMEGA_RAD_S * (d / 1000.0) for d in dt_values]
    vmax = max(np.max(d) for d in disp_all)

    # Layout: [image] [narrow cbar axis] [image]
    from matplotlib import gridspec
    fig = plt.figure(figsize=(18, 6), constrained_layout=False)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.04, 1], wspace=0.08, figure=fig)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    # Left image (dt=1.0)
    disp_left = disp_all[0]
    im_left = ax_left.imshow(disp_left, cmap='viridis', origin='upper', vmin=0, vmax=vmax)
    ax_left.set_title('Flow Magnitude Map (dt = 1.0ms)')
    ax_left.set_xlabel('X (pixels)')
    ax_left.set_ylabel('Y (pixels)')
    circle = plt.Circle((DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS, fill=False,
                        linestyle='--', color='white', linewidth=2, alpha=0.8)
    ax_left.add_patch(circle)
    ax_left.plot(DISC_CENTER_X, DISC_CENTER_Y, 'r+', markersize=10, markeredgewidth=2)
    ax_left.text(DISC_CENTER_X + 20, DISC_CENTER_Y + 20, 'Center', color='red', fontsize=10)

    # Right image (dt=3.0)
    disp_right = disp_all[1]
    im_right = ax_right.imshow(disp_right, cmap='viridis', origin='upper', vmin=0, vmax=vmax)
    ax_right.set_title('Flow Magnitude Map (dt = 3.0ms)')
    ax_right.set_xlabel('X (pixels)')
    ax_right.set_ylabel('Y (pixels)')
    circle2 = plt.Circle((DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS, fill=False,
                         linestyle='--', color='white', linewidth=2, alpha=0.8)
    ax_right.add_patch(circle2)
    ax_right.plot(DISC_CENTER_X, DISC_CENTER_Y, 'r+', markersize=10, markeredgewidth=2)
    ax_right.text(DISC_CENTER_X + 20, DISC_CENTER_Y + 20, 'Center', color='red', fontsize=10)

    # Central colorbar
    cbar = fig.colorbar(im_right, cax=ax_cbar)
    cbar.set_label('Pixel Displacement (px)')

    fig.savefig(OUT_DIR_SHORT / 'flow_magnitude_maps.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_DIR_SHORT / 'flow_magnitude_maps.svg'}")


def main():
    print("Generating consolidated thesis figures (images)...")
    create_windowwise_dt_sweep()
    flow_magnitude_maps()
    early_dropoff_fine_resolution()
    fine_resolution_dt_sweep()
    print(f"All outputs written to: {OUT_DIR_SHORT} and {OUT_DIR_FINE}")


if __name__ == "__main__":
    main()


