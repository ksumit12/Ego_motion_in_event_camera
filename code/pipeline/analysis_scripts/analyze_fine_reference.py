#!/usr/bin/env python3
"""
Analyze cancellation rates for fine-resolution data using REFERENCE SCRIPT LOGIC.
This is adapted from the working /media/sumit/New Volume/anu_research/ego_motion/analyze_dt_cancellation.py

Key settings matching reference:
- BIN_MS = 5.0 (temporal tolerance)
- R_PIX = 2.0 (spatial tolerance)
- SUBSET_SIZE = 50000
- Uses correct cancel_events_time_aware logic
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

# =============== Configuration ===============
WINDOW_PREDICTIONS_DIR = "/media/sumit/New Volume/fine_resolution_window"

# Cancellation parameters (MATCHING REFERENCE)
BIN_MS = 5.0          # Temporal tolerance (ms)
R_PIX = 2.0           # Spatial tolerance (pixels)
POLARITY_MODE = "opposite"

# Motion parameters
OMEGA_RAD_S = 3.6  # Angular velocity (rad/s)
DISC_RADIUS_PX = 264

# Disc center coordinates
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 250

# Output settings
OUTPUT_DIR = "./code/thesis_figures/fine_resolution_plots"
PLOT_DPI = 300

# Performance options (MATCHING REFERENCE)
USE_SUBSET_FOR_TESTING = True
SUBSET_SIZE = 50000
USE_MEMORY_MAPPING = True
CHUNK_SIZE = 10000

def load_npy_or_npz(path, key=None, use_mmap=True, subset_size=None):
    """Load .npy or .npz with memory mapping and optional subsetting."""
    if path.endswith('.npz'):
        with np.load(path) as z:
            if key is not None and key in z:
                arr = z[key]
            else:
                first_key = list(z.files)[0]
                arr = z[first_key]
    else:
        if use_mmap:
            arr = np.load(path, mmap_mode='r')
        else:
            arr = np.load(path)
    
    if subset_size is not None and len(arr) > subset_size:
        arr = arr[:subset_size]
    
    return arr

def combine_real_pred(real_events, pred_events):
    """Combine 4-col real and pred into 5-col with flags and sort by time."""
    if len(real_events) == 0 and len(pred_events) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    real_flag = np.zeros((len(real_events), 1), dtype=np.float32)
    pred_flag = np.ones((len(pred_events), 1), dtype=np.float32)
    stacks = []
    if len(real_events) > 0:
        stacks.append(np.column_stack([real_events, real_flag]))
    if len(pred_events) > 0:
        stacks.append(np.column_stack([pred_events, pred_flag]))
    combined = np.vstack(stacks)
    return combined[np.argsort(combined[:, 3])]

def discover_window_dirs(base_dir):
    """Return list of (path, label) for window_* subdirs, sorted by name."""
    try:
        entries = sorted([p for p in Path(base_dir).iterdir() if p.is_dir() and p.name.startswith('window_')], 
                        key=lambda p: p.name)
        return [(str(p), p.name) for p in entries]
    except FileNotFoundError:
        return []

def try_load_combined_or_split(window_dir, dt_ms):
    """Try loading combined_events for given dt. If not present, load split real/pred."""
    subset_size = SUBSET_SIZE if USE_SUBSET_FOR_TESTING else None
    
    # Try combined .npy or .npz
    combined_candidates = [
        os.path.join(window_dir, f"combined_events_dt_{dt_ms:04.1f}ms.npy"),
        os.path.join(window_dir, f"combined_events_dt_{dt_ms:04.1f}ms.npz"),
    ]
    for path in combined_candidates:
        if os.path.exists(path):
            arr = load_npy_or_npz(path, key='combined', use_mmap=USE_MEMORY_MAPPING, subset_size=subset_size)
            return arr
    
    # Try split
    real_candidates = [
        os.path.join(window_dir, 'real_events.npy'),
        os.path.join(window_dir, 'real_events.npz'),
    ]
    pred_candidates = [
        os.path.join(window_dir, f"pred_events_dt_{dt_ms:04.1f}ms.npy"),
        os.path.join(window_dir, f"pred_events_dt_{dt_ms:04.1f}ms.npz"),
    ]
    real = None
    for path in real_candidates:
        if os.path.exists(path):
            real = load_npy_or_npz(path, key='real', use_mmap=USE_MEMORY_MAPPING, subset_size=subset_size)
            break
    if real is None:
        raise FileNotFoundError(f"Missing real_events in {window_dir}")
    pred = None
    for path in pred_candidates:
        if os.path.exists(path):
            pred = load_npy_or_npz(path, key='pred', use_mmap=USE_MEMORY_MAPPING, subset_size=subset_size)
            break
    if pred is None:
        raise FileNotFoundError(f"Missing predictions for dt={dt_ms}ms in {window_dir}")
    return combine_real_pred(real, pred)

def circle_mask(x, y, cx, cy, r, scale=1.05):
    """Create boolean mask for points inside scaled circle"""
    return (x - cx)**2 + (y - cy)**2 <= (r * scale)**2

def cancel_events_time_aware(real_events, predicted_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels):
    """
    REFERENCE IMPLEMENTATION - Match real and predicted events using TRUE temporal gate.
    """
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    
    if num_real == 0 or num_predicted == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    matched_real = np.zeros(num_real, dtype=bool)
    matched_predicted = np.zeros(num_predicted, dtype=bool)
    total_matches = 0
    
    # Create spatial KDTree for predicted events
    pred_tree = cKDTree(predicted_events[:, :2])
    
    chunk_size = min(CHUNK_SIZE, num_real)
    
    pbar = tqdm(total=num_real, desc="  Cancelling events", unit="events", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for chunk_start in range(0, num_real, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_real)
        chunk_real = real_events[chunk_start:chunk_end]
        
        # KEY: t_i + Δt for all events in chunk
        chunk_target_times = chunk_real[:, 3] + dt_seconds
        
        # Query KDTree for spatial candidates
        spatial_candidates_list = pred_tree.query_ball_point(chunk_real[:, :2], spatial_tolerance_pixels)
        
        for i, (real_event, target_time, spatial_candidates) in enumerate(zip(chunk_real, chunk_target_times, spatial_candidates_list)):
            real_idx = chunk_start + i
            if matched_real[real_idx]:
                pbar.update(1)
                continue
            
            if len(spatial_candidates) == 0:
                pbar.update(1)
                continue
            
            spatial_candidates = np.array(spatial_candidates)
            available_candidates = spatial_candidates[~matched_predicted[spatial_candidates]]
            
            if len(available_candidates) == 0:
                pbar.update(1)
                continue
            
            # Temporal filter
            candidate_times = predicted_events[available_candidates, 3]
            temporal_mask = np.abs(candidate_times - target_time) <= temporal_tolerance_s
            
            if not np.any(temporal_mask):
                pbar.update(1)
                continue
            
            final_candidates = available_candidates[temporal_mask]
            candidate_events = predicted_events[final_candidates]
            
            # Polarity check
            real_polarity = real_event[2]
            pred_polarities = candidate_events[:, 2]
            
            if POLARITY_MODE == "opposite":
                polarity_matches = (pred_polarities != real_polarity)
            elif POLARITY_MODE == "equal":
                polarity_matches = (pred_polarities == real_polarity)
            else:
                polarity_matches = np.ones(len(candidate_events), dtype=bool)
            
            if np.any(polarity_matches):
                valid_candidates = final_candidates[polarity_matches]
                valid_events = candidate_events[polarity_matches]
                
                distances = np.sqrt(np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1))
                best_candidate = valid_candidates[np.argmin(distances)]
                
                matched_real[real_idx] = True
                matched_predicted[best_candidate] = True
                total_matches += 1
            
            pbar.update(1)
    
    pbar.close()
    
    unmatched_real = ~matched_real
    unmatched_predicted = ~matched_predicted
    
    return unmatched_real, unmatched_predicted, total_matches

def run_cancellation_for_window(combined_events, temporal_tolerance_ms, spatial_tolerance_pixels):
    """Run cancellation using reference logic."""
    real_events = combined_events[combined_events[:, 4] == 0.0]
    pred_events = combined_events[combined_events[:, 4] == 1.0]
    
    if len(real_events) == 0 or len(pred_events) == 0:
        return np.zeros((0, 5)), np.zeros((0, 5)), 0
    
    # Estimate dt from data
    sample_real_times = real_events[:min(1000, len(real_events)), 3]
    sample_pred_times = pred_events[:min(1000, len(pred_events)), 3]
    
    time_diffs = []
    for rt in sample_real_times:
        closest_pred_times = sample_pred_times[np.abs(sample_pred_times - rt) < 0.1]
        if len(closest_pred_times) > 0:
            closest_diff = np.min(closest_pred_times - rt)
            if closest_diff > 0:
                time_diffs.append(closest_diff)
    
    dt_seconds = np.median(time_diffs) if len(time_diffs) > 0 else 0.002
    
    unmatched_real_mask, unmatched_predicted_mask, total_matched_pairs = cancel_events_time_aware(
        real_events, pred_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels
    )
    
    residual_real_events = real_events[unmatched_real_mask]
    residual_predicted_events = pred_events[unmatched_predicted_mask]
    
    return residual_real_events, residual_predicted_events, total_matched_pairs

def calculate_roi_cancellation_rate(combined_events, residual_real_events, disc_center, disc_radius):
    """Calculate cancellation rate for ROI."""
    real_events = combined_events[combined_events[:, 4] == 0.0]
    
    if len(real_events) == 0:
        return 0.0, 0, 0
    
    cx, cy = disc_center
    roi_mask = circle_mask(real_events[:, 0], real_events[:, 1], cx, cy, disc_radius)
    roi_real_events = real_events[roi_mask]
    
    if len(roi_real_events) == 0:
        return 0.0, 0, 0
    
    if len(residual_real_events) > 0:
        roi_residual_mask = circle_mask(residual_real_events[:, 0], residual_real_events[:, 1], cx, cy, disc_radius)
        roi_residual_events = residual_real_events[roi_residual_mask]
    else:
        roi_residual_events = np.zeros((0, 5))
    
    total_roi_real = len(roi_real_events)
    total_roi_residual = len(roi_residual_events)
    total_roi_cancelled = total_roi_real - total_roi_residual
    cancellation_rate = (total_roi_cancelled / total_roi_real * 100) if total_roi_real > 0 else 0.0
    
    return cancellation_rate, total_roi_real, total_roi_cancelled

def analyze_window_dt_cancellation_dir(window_idx, window_dir, window_label, dt_values_ms):
    """Analyze all dt values for a window directory."""
    print(f"\nAnalyzing window {window_idx + 1}: {window_label}")
    
    if not os.path.exists(window_dir):
        print(f"  Warning: Directory {window_dir} not found")
        return None
    
    results = []
    dt_pbar = tqdm(dt_values_ms, desc=f"Window {window_idx + 1}", unit="dt")
    
    for dt_ms in dt_pbar:
        dt_pbar.set_postfix_str(f"dt={dt_ms}ms")
        try:
            combined_events = try_load_combined_or_split(window_dir, dt_ms)
            if USE_SUBSET_FOR_TESTING and len(combined_events) > SUBSET_SIZE:
                print(f"\n    Using subset ({SUBSET_SIZE:,}/{len(combined_events):,}) events")
        except FileNotFoundError as e:
            print(f"\n    Skip: {e}")
            continue
        
        # Run cancellation
        residual_real, residual_pred, matched_pairs = run_cancellation_for_window(
            combined_events, BIN_MS, R_PIX
        )
        
        # Calculate ROI rate
        roi_cancellation_rate, total_roi_real, total_roi_cancelled = calculate_roi_cancellation_rate(
            combined_events, residual_real, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS
        )
        
        results.append({
            'dt_ms': dt_ms,
            'cancellation_rate': roi_cancellation_rate,
            'total_roi_real': total_roi_real,
            'total_roi_cancelled': total_roi_cancelled,
            'total_matched_pairs': matched_pairs
        })
        
        print(f"ROI CR: {roi_cancellation_rate:.1f}% ({total_roi_cancelled}/{total_roi_real})")
    
    return results

def create_plots(all_results, output_dir):
    """Create fine-resolution plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    t90_ms = None
    t80_ms = None
    
    MEAN_RADIUS_PX = 199
    
    for window_idx, window_results in enumerate(all_results):
        if window_results is None:
            continue
        
        dt_values = [r['dt_ms'] for r in window_results]
        cancel_rates = [r['cancellation_rate'] for r in window_results]
        
        # Plot CR
        ax.plot(dt_values, cancel_rates, 'o-', linewidth=2.5, markersize=6, 
                color='#1f77b4', label='Cancellation Rate')
        
        # Shade ≥90% region
        ax.axhspan(90, 100, alpha=0.15, color='green', label='≥90% Region')
        
        # Find thresholds
        t90_idx = np.where(np.array(cancel_rates) < 90)[0]
        t80_idx = np.where(np.array(cancel_rates) < 80)[0]
        
        t90_ms = dt_values[t90_idx[0]] if len(t90_idx) > 0 else None
        t80_ms = dt_values[t80_idx[0]] if len(t80_idx) > 0 else None
        
        # Add threshold lines
        ax.axhline(y=90, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='90% threshold')
        ax.axhline(y=80, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='80% threshold')
        
        if t90_ms is not None:
            ax.axvline(x=t90_ms, color='orange', linestyle='--', alpha=0.7, linewidth=2)
            t90_px = t90_ms * OMEGA_RAD_S * MEAN_RADIUS_PX / 1000
            ax.plot(t90_ms, cancel_rates[t90_idx[0]], 'o', markersize=12, color='red', zorder=5)
            ax.annotate(f'Drop-off: {t90_ms:.1f}ms\n{cancel_rates[t90_idx[0]]:.1f}%',
                       xy=(t90_ms, cancel_rates[t90_idx[0]]),
                       xytext=(t90_ms + 0.4, cancel_rates[t90_idx[0]] - 6),
                       fontsize=12,
                       bbox=dict(boxstyle='round,pad=0.6', fc='yellow', alpha=0.9, 
                                edgecolor='orange', linewidth=2),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=2))
        
        # Secondary axis: pixel displacement
        ax2 = ax.twiny()
        pixel_displacement = [dt_ms * OMEGA_RAD_S * MEAN_RADIUS_PX / 1000 for dt_ms in dt_values]
        ax2.set_xlim(pixel_displacement[0], pixel_displacement[-1])
        ax2.set_xlabel(f'Predicted Pixel Displacement (px)\n(ω={OMEGA_RAD_S} rad/s, r≈{MEAN_RADIUS_PX}px mean)',
                      fontsize=12, color='#555', fontweight='bold')
        ax2.tick_params(axis='x', labelcolor='#555')
        
        # Save data
        results_df = pd.DataFrame({
            'dt_ms': dt_values,
            'cancellation_rate': cancel_rates,
            'pixel_displacement': pixel_displacement
        })
        csv_path = os.path.join(output_dir, "fine_resolution_data.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV: {csv_path}")
        
        # Save npz for other plots
        np.savez(os.path.join(output_dir, "fine_resolution_data.npz"),
                dt_values=np.array(dt_values),
                cancellation_rates=np.array(cancel_rates),
                pixel_displacement=np.array(pixel_displacement))
    
    # Labels and formatting
    ax.set_xlim(0, 3)
    ax.set_ylim(70, 101)
    ax.set_xlabel('Prediction Horizon Δt (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cancellation Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Early Drop-off Analysis: Fine Resolution (0.1ms steps)\n(Representative sample: 50,000 events, εt=5.0ms, εxy=2.0px)',
                fontsize=14, fontweight='bold', pad=65)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "fine_resolution_dropoff.svg")
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    
    plt.close()
    
    return t90_ms, t80_ms

def main():
    """Main execution."""
    start_time = time.time()
    
    print("=== Fine Resolution Analysis (Reference Logic) ===")
    print(f"Window predictions: {WINDOW_PREDICTIONS_DIR}")
    print(f"Parameters: εt={BIN_MS}ms, εxy={R_PIX}px")
    print(f"Polarity: {POLARITY_MODE}")
    print(f"ROI: ({DISC_CENTER_X:.1f}, {DISC_CENTER_Y:.1f}), r={DISC_RADIUS}px")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Subset mode: {SUBSET_SIZE:,} events per dt\n")
    
    # Fine dt values
    dt_values_ms_primary = np.arange(0.0, 2.0 + 1e-9, 0.1)
    dt_values_ms_tail = np.arange(2.0, 3.0 + 1e-9, 0.25)
    dt_values_ms = np.unique(np.concatenate([dt_values_ms_primary, dt_values_ms_tail]))
    
    print(f"Analyzing {len(dt_values_ms)} dt values: {dt_values_ms[0]:.1f} to {dt_values_ms[-1]:.1f}ms\n")
    
    # Discover windows
    discovered = discover_window_dirs(WINDOW_PREDICTIONS_DIR)
    all_results = []
    
    if discovered:
        print(f"Found {len(discovered)} window directories")
        for window_idx, (window_dir, label) in enumerate(discovered):
            window_results = analyze_window_dt_cancellation_dir(window_idx, window_dir, label, dt_values_ms)
            all_results.append(window_results)
    else:
        print("No windows found")
        return
    
    # Create plots
    print("\nCreating plots...")
    t90_ms, t80_ms = create_plots(all_results, OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if t90_ms:
        t90_px = t90_ms * OMEGA_RAD_S * 199 / 1000
        print(f"t90: {t90_ms:.1f}ms ≈ {t90_px:.1f}px")
    if t80_ms:
        t80_px = t80_ms * OMEGA_RAD_S * 199 / 1000
        print(f"t80: {t80_ms:.1f}ms ≈ {t80_px:.1f}px")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    print("="*60)

if __name__ == "__main__":
    main()

