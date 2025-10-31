#!/usr/bin/env python3
"""
Analyze cancellation rates for different dt values within circular ROI.
This script loads the window predictions and calculates cancellation rates
for each dt value, focusing on the circular region of interest.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

# =============== Configuration ===============
WINDOW_PREDICTIONS_DIR = "/media/sumit/New Volume1/window_predictions_5s"

# Cancellation parameters (same as visualize_time_window.py)
BIN_MS = 5.0          # Temporal tolerance (ms)
R_PIX = 2.0           # Spatial tolerance (pixels)
POLARITY_MODE = "opposite"  # "opposite" | "equal" | "ignore"

# Disc center coordinates and radius (same as visualize_time_window.py)
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264

# Time windows (same as in generate_window_predictions.py)
WINDOWS = [
    (5.000, 5.010),
    (8.200, 8.210),
    (9.000, 9.010),
]

# DT values to analyze
DT_RANGE_MS = (0, 20)
DT_STEP_MS = 1

# Output settings
OUTPUT_DIR = "./dt_analysis_results"
PLOT_DPI = 150

# Performance options for large datasets
USE_SUBSET_FOR_TESTING = False  # Process FULL 5-second dataset
SUBSET_SIZE = None  # No subsetting - process all events
USE_MEMORY_MAPPING = True  # Use memory mapping to avoid loading full arrays
CHUNK_SIZE = 1_000_000  # Process events in chunks of 1M for memory efficiency

def load_npy_or_npz(path, key=None, use_mmap=True, subset_size=None):
    """Load .npy or .npz with memory mapping and optional subsetting."""
    if path.endswith('.npz'):
        with np.load(path) as z:
            if key is not None and key in z:
                arr = z[key]
            else:
                # fallback to first array in archive
                first_key = list(z.files)[0]
                arr = z[first_key]
    else:
        if use_mmap:
            arr = np.load(path, mmap_mode='r')
        else:
            arr = np.load(path)
    
    # Apply subset if requested
    if subset_size is not None and len(arr) > subset_size:
        # Take a representative subset (first N events)
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
        entries = sorted([p for p in Path(base_dir).iterdir() if p.is_dir() and p.name.startswith('window_')], key=lambda p: p.name)
        return [(str(p), p.name) for p in entries]
    except FileNotFoundError:
        return []

def try_load_combined_or_split(window_dir, dt_ms):
    """Try loading combined_events for given dt. If not present, load split real/pred."""
    subset_size = SUBSET_SIZE if USE_SUBSET_FOR_TESTING else None
    
    # Try combined .npy or .npz
    combined_candidates = [
        os.path.join(window_dir, f"combined_events_dt_{dt_ms:02d}ms.npy"),
        os.path.join(window_dir, f"combined_events_dt_{dt_ms:02d}ms.npz"),
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
        os.path.join(window_dir, f"pred_events_dt_{dt_ms:02d}ms.npy"),
        os.path.join(window_dir, f"pred_events_dt_{dt_ms:02d}ms.npz"),
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

def check_polarity_match(real_polarity, predicted_polarity):
    """Check if two events should be matched based on polarity mode"""
    if POLARITY_MODE == "ignore":
        return True
    elif POLARITY_MODE == "equal":
        return real_polarity == predicted_polarity
    else:  # "opposite" mode
        return real_polarity != predicted_polarity

def cancel_events_time_aware(real_events, predicted_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels):
    """
    Match real and predicted events using TRUE temporal gate: |t_j-(t_i+Δt)|≤ε_t
    
    This implements the correct mathematical formulation from the thesis.
    HIGHLY OPTIMIZED version using spatial indexing for large datasets.
    """
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    
    # If either list is empty, nothing can be matched
    if num_real == 0 or num_predicted == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    matched_real = np.zeros(num_real, dtype=bool)
    matched_predicted = np.zeros(num_predicted, dtype=bool)
    total_matches = 0
    
    # Create spatial KDTree for predicted events (much faster spatial search)
    pred_tree = cKDTree(predicted_events[:, :2])  # Only x, y coordinates
    
    # Process real events in larger chunks for efficiency
    chunk_size = min(CHUNK_SIZE, num_real)  # Process in chunks of 1M events
    
    # Create progress bar for cancellation processing
    pbar = tqdm(total=num_real, desc="  Cancelling events", unit="events", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for chunk_start in range(0, num_real, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_real)
        chunk_real = real_events[chunk_start:chunk_end]
        
        # Vectorized processing of the entire chunk
        chunk_target_times = chunk_real[:, 3] + dt_seconds  # t_i + Δt for all events in chunk
        
        # For each real event in chunk, find spatial candidates first (much faster)
        # Query KDTree for spatial candidates within tolerance
        spatial_candidates_list = pred_tree.query_ball_point(chunk_real[:, :2], spatial_tolerance_pixels)
        
        # Process each event in the chunk
        for i, (real_event, target_time, spatial_candidates) in enumerate(zip(chunk_real, chunk_target_times, spatial_candidates_list)):
            real_idx = chunk_start + i
            if matched_real[real_idx]:
                pbar.update(1)
                continue
            
            if len(spatial_candidates) == 0:
                pbar.update(1)
                continue
            
            # Convert to numpy array and filter out already matched
            spatial_candidates = np.array(spatial_candidates)
            available_candidates = spatial_candidates[~matched_predicted[spatial_candidates]]
            
            if len(available_candidates) == 0:
                pbar.update(1)
                continue
            
            # Among spatial candidates, find temporal candidates
            candidate_times = predicted_events[available_candidates, 3]
            temporal_mask = np.abs(candidate_times - target_time) <= temporal_tolerance_s
            
            if not np.any(temporal_mask):
                pbar.update(1)
                continue
            
            # Get final candidates (both spatial and temporal)
            final_candidates = available_candidates[temporal_mask]
            candidate_events = predicted_events[final_candidates]
            
            # Vectorized polarity check
            real_polarity = real_event[2]
            pred_polarities = candidate_events[:, 2]
            
            if POLARITY_MODE == "ignore":
                polarity_matches = np.ones(len(candidate_events), dtype=bool)
            elif POLARITY_MODE == "equal":
                polarity_matches = (pred_polarities == real_polarity)
            else:  # "opposite"
                polarity_matches = (pred_polarities != real_polarity)
            
            if np.any(polarity_matches):
                # Find closest among valid candidates
                valid_candidates = final_candidates[polarity_matches]
                valid_events = candidate_events[polarity_matches]
                
                # Calculate distances to valid candidates
                distances = np.sqrt(np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1))
                best_candidate = valid_candidates[np.argmin(distances)]
                
                # Make the match
                matched_real[real_idx] = True
                matched_predicted[best_candidate] = True
                total_matches += 1
            
            pbar.update(1)
    
    pbar.close()
    
    # Return unmatched events (inverse of matched)
    unmatched_real = ~matched_real
    unmatched_predicted = ~matched_predicted
    
    return unmatched_real, unmatched_predicted, total_matches


def run_cancellation_for_window(combined_events, temporal_tolerance_ms, spatial_tolerance_pixels):
    """
    Run ego-motion cancellation using TRUE temporal gate: |t_j-(t_i+Δt)|≤ε_t
    
    This implements the correct mathematical formulation from the thesis.
    No binning - direct temporal relationship per event.
    """
    # Split events
    real_events = combined_events[combined_events[:, 4] == 0.0]
    pred_events = combined_events[combined_events[:, 4] == 1.0]
    
    total_real_events = len(real_events)
    total_predicted_events = len(pred_events)
    
    if total_real_events == 0 or total_predicted_events == 0:
        return np.zeros((0, 5), dtype=combined_events.dtype), np.zeros((0, 5), dtype=combined_events.dtype), 0
    
    # Use the correct temporal gate approach
    # We need dt_seconds - estimate from the data
    if len(real_events) > 0 and len(pred_events) > 0:
        # Estimate dt from the time difference between real and predicted events
        # This is a heuristic - in practice dt should be known from the prediction process
        sample_real_times = real_events[:min(1000, len(real_events)), 3]
        sample_pred_times = pred_events[:min(1000, len(pred_events)), 3]
        
        # Find the most common time difference (this should be dt)
        time_diffs = []
        for rt in sample_real_times:
            closest_pred_times = sample_pred_times[np.abs(sample_pred_times - rt) < 0.1]  # Within 100ms
            if len(closest_pred_times) > 0:
                closest_diff = np.min(closest_pred_times - rt)
                if closest_diff > 0:  # Predicted events should be later
                    time_diffs.append(closest_diff)
        
        if len(time_diffs) > 0:
            dt_seconds = np.median(time_diffs)  # Use median as robust estimate
        else:
            dt_seconds = 0.002  # Default 2ms if estimation fails
    else:
        dt_seconds = 0.002  # Default 2ms
    
    # Run time-aware cancellation
    unmatched_real_mask, unmatched_predicted_mask, total_matched_pairs = cancel_events_time_aware(
        real_events, pred_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels
    )
    
    # Get residual events
    residual_real_events = real_events[unmatched_real_mask]
    residual_predicted_events = pred_events[unmatched_predicted_mask]
    
    return residual_real_events, residual_predicted_events, total_matched_pairs

def calculate_roi_cancellation_rate(combined_events, residual_real_events, disc_center, disc_radius):
    """Calculate cancellation rate specifically for the circular ROI"""
    # Extract real events from combined data
    real_events = combined_events[combined_events[:, 4] == 0.0]
    
    if len(real_events) == 0:
        return 0.0, 0, 0
    
    # Filter real events to ROI
    cx, cy = disc_center
    roi_mask = circle_mask(real_events[:, 0], real_events[:, 1], cx, cy, disc_radius)
    roi_real_events = real_events[roi_mask]
    
    if len(roi_real_events) == 0:
        return 0.0, 0, 0
    
    # Filter residual events to ROI
    if len(residual_real_events) > 0:
        roi_residual_mask = circle_mask(residual_real_events[:, 0], residual_real_events[:, 1], cx, cy, disc_radius)
        roi_residual_events = residual_real_events[roi_residual_mask]
    else:
        roi_residual_events = np.zeros((0, 5), dtype=combined_events.dtype)
    
    # Calculate cancellation rate
    total_roi_real = len(roi_real_events)
    total_roi_residual = len(roi_residual_events)
    total_roi_cancelled = total_roi_real - total_roi_residual
    cancellation_rate = (total_roi_cancelled / total_roi_real * 100) if total_roi_real > 0 else 0.0
    
    return cancellation_rate, total_roi_real, total_roi_cancelled

def calculate_outside_roi_cancellation_rate(combined_events, residual_real_events, disc_center, disc_radius):
    """Calculate cancellation rate specifically for the region OUTSIDE the circular ROI"""
    # Extract real events from combined data
    real_events = combined_events[combined_events[:, 4] == 0.0]
    
    if len(real_events) == 0:
        return 0.0, 0, 0
    
    # Build inside mask then invert for outside
    cx, cy = disc_center
    inside_mask = circle_mask(real_events[:, 0], real_events[:, 1], cx, cy, disc_radius)
    outside_real_events = real_events[~inside_mask]
    
    if len(outside_real_events) == 0:
        return 0.0, 0, 0
    
    # Filter residual events to OUTSIDE ROI
    if len(residual_real_events) > 0:
        residual_inside_mask = circle_mask(residual_real_events[:, 0], residual_real_events[:, 1], cx, cy, disc_radius)
        outside_residual_events = residual_real_events[~residual_inside_mask]
    else:
        outside_residual_events = np.zeros((0, 5), dtype=combined_events.dtype)
    
    total_out_real = len(outside_real_events)
    total_out_residual = len(outside_residual_events)
    total_out_cancelled = total_out_real - total_out_residual
    cancellation_rate_out = (total_out_cancelled / total_out_real * 100) if total_out_real > 0 else 0.0
    
    return cancellation_rate_out, total_out_real, total_out_cancelled

def analyze_window_dt_cancellation_dir(window_idx, window_dir, window_label, dt_values_ms):
    """Analyze cancellation rates for all dt values for a given window directory"""
    print(f"\nAnalyzing window {window_idx + 1}: {window_label}")
    
    if not os.path.exists(window_dir):
        print(f"  Warning: Directory {window_dir} not found, skipping...")
        return None
    
    results = []
    
    # Create progress bar for dt analysis
    dt_pbar = tqdm(dt_values_ms, desc=f"Window {window_idx + 1}", unit="dt")
    
    for dt_ms in dt_pbar:
        # Load combined or split (real+pred) for this dt
        dt_pbar.set_postfix_str(f"dt={dt_ms}ms")
        try:
            combined_events = try_load_combined_or_split(window_dir, dt_ms)
            if USE_SUBSET_FOR_TESTING and len(combined_events) > SUBSET_SIZE:
                print(f"\n    Using subset ({SUBSET_SIZE:,}/{len(combined_events):,}) events")
            else:
                print(f"\n    Loaded {len(combined_events):,} events")
        except FileNotFoundError as e:
            print(f"\n    Skip: {e}")
            continue
        
        # Run cancellation
        residual_real, residual_pred, matched_pairs = run_cancellation_for_window(
            combined_events, BIN_MS, R_PIX
        )
        
        # Calculate ROI cancellation rate (inside)
        roi_cancellation_rate, total_roi_real, total_roi_cancelled = calculate_roi_cancellation_rate(
            combined_events, residual_real, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS
        )
        # Calculate OUTSIDE ROI cancellation rate
        out_cancellation_rate, total_out_real, total_out_cancelled = calculate_outside_roi_cancellation_rate(
            combined_events, residual_real, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS
        )
        
        results.append({
            'dt_ms': dt_ms,
            'cancellation_rate': roi_cancellation_rate,
            'total_roi_real': total_roi_real,
            'total_roi_cancelled': total_roi_cancelled,
            'cancellation_rate_outside': out_cancellation_rate,
            'total_outside_real': total_out_real,
            'total_outside_cancelled': total_out_cancelled,
            'total_matched_pairs': matched_pairs
        })
        
        print(f"ROI cancel: {roi_cancellation_rate:.1f}% ({total_roi_cancelled}/{total_roi_real}) | OUTSIDE: {out_cancellation_rate:.1f}% ({total_out_cancelled}/{total_out_real})")
    
    return results

def analyze_window_dt_cancellation(window_idx, window, dt_values_ms):
    """Analyze cancellation rates for all dt values for a specific window"""
    t0, t1 = window
    print(f"\nAnalyzing window {window_idx + 1}: {t0:.3f}s to {t1:.3f}s")
    
    # Load window predictions directory
    window_dir = os.path.join(WINDOW_PREDICTIONS_DIR, f"window_{window_idx + 1}_{t0:.3f}s_to_{t1:.3f}s")
    
    if not os.path.exists(window_dir):
        print(f"  Warning: Directory {window_dir} not found, skipping...")
        return None
    
    results = []
    
    for dt_ms in dt_values_ms:
        # Load combined events for this dt
        filename = f"combined_events_dt_{dt_ms:02d}ms.npy"
        filepath = os.path.join(window_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  Warning: File {filename} not found, skipping dt={dt_ms}ms")
            continue
        
        print(f"  Processing dt={dt_ms:2d}ms...", end=" ")
        
        # Load data
        combined_events = np.load(filepath)
        
        # Run cancellation
        residual_real, residual_pred, matched_pairs = run_cancellation_for_window(
            combined_events, BIN_MS, R_PIX
        )
        
        # Calculate ROI cancellation rate
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
        
        print(f"ROI cancel rate: {roi_cancellation_rate:.1f}% ({total_roi_cancelled}/{total_roi_real})")
    
    return results

def create_cancellation_plots(all_results, output_dir):
    """Create plots showing cancellation rate vs dt for all windows"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots for each window
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['tab:blue', 'tab:red', 'tab:green']
    
    for window_idx, window_results in enumerate(all_results):
        if window_results is None:
            continue
        
        ax = axes[window_idx]
        
        # Extract data
        dt_values = [r['dt_ms'] for r in window_results]
        cancel_rates = [r['cancellation_rate'] for r in window_results]
        total_real = [r['total_roi_real'] for r in window_results]
        
        # Plot cancellation rate
        ax.plot(dt_values, cancel_rates, 'o-', color=colors[window_idx], 
                linewidth=2, markersize=6, label=f'Window {window_idx + 1}')
        ax.set_xlabel('dt (ms)')
        ax.set_ylabel('Cancellation Rate (%)')
        ax.set_title(f'Window {window_idx + 1}: {WINDOWS[window_idx][0]:.3f}s to {WINDOWS[window_idx][1]:.3f}s')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add secondary y-axis for event counts
        ax2 = ax.twinx()
        ax2.bar(dt_values, total_real, alpha=0.3, color=colors[window_idx], 
                label=f'ROI Events (Window {window_idx + 1})')
        ax2.set_ylabel('Total ROI Events')
        ax2.legend(loc='upper right')
    
    # Combined plot
    ax_combined = axes[3]
    for window_idx, window_results in enumerate(all_results):
        if window_results is None:
            continue
        
        dt_values = [r['dt_ms'] for r in window_results]
        cancel_rates = [r['cancellation_rate'] for r in window_results]
        
        ax_combined.plot(dt_values, cancel_rates, 'o-', color=colors[window_idx], 
                        linewidth=2, markersize=6, label=f'Window {window_idx + 1}')
    
    ax_combined.set_xlabel('dt (ms)')
    ax_combined.set_ylabel('Cancellation Rate (%)')
    ax_combined.set_title('Combined: All Windows')
    ax_combined.grid(True, alpha=0.3)
    ax_combined.legend()
    
    plt.tight_layout()
    
    # Save plot (INSIDE ROI)
    plot_filename = os.path.join(output_dir, "dt_cancellation_analysis.png")
    plt.savefig(plot_filename, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved plot: {plot_filename}")
    
    # Save data to CSV (INSIDE ROI)
    csv_filename = os.path.join(output_dir, "dt_cancellation_data.csv")
    with open(csv_filename, 'w') as f:
        f.write("window,dt_ms,cancellation_rate,total_roi_real,total_roi_cancelled,total_matched_pairs\n")
        for window_idx, window_results in enumerate(all_results):
            if window_results is None:
                continue
            for result in window_results:
                f.write(f"{window_idx + 1},{result['dt_ms']},{result['cancellation_rate']:.2f},"
                       f"{result['total_roi_real']},{result['total_roi_cancelled']},{result['total_matched_pairs']}\n")
    
    print(f"Saved data: {csv_filename}")

    # --------- OUTSIDE ROI PLOTS ---------
    fig_out, axes_out = plt.subplots(2, 2, figsize=(15, 12))
    axes_out = axes_out.flatten()

    for window_idx, window_results in enumerate(all_results):
        if window_results is None:
            continue
        ax = axes_out[window_idx]
        dt_values = [r['dt_ms'] for r in window_results]
        cancel_rates_out = [r.get('cancellation_rate_outside', np.nan) for r in window_results]
        total_out = [r.get('total_outside_real', 0) for r in window_results]

        ax.plot(dt_values, cancel_rates_out, 'o-', color=colors[window_idx],
                linewidth=2, markersize=6, label=f'Window {window_idx + 1} (outside)')
        ax.set_xlabel('dt (ms)')
        ax.set_ylabel('Cancellation Rate Outside ROI (%)')
        ax.set_title(f'Outside ROI • Window {window_idx + 1}: {WINDOWS[window_idx][0]:.3f}s to {WINDOWS[window_idx][1]:.3f}s')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax2 = ax.twinx()
        ax2.bar(dt_values, total_out, alpha=0.3, color=colors[window_idx],
                label=f'Outside ROI Events (Window {window_idx + 1})')
        ax2.set_ylabel('Total Outside-ROI Events')
        ax2.legend(loc='upper right')

    ax_combined_out = axes_out[3]
    for window_idx, window_results in enumerate(all_results):
        if window_results is None:
            continue
        dt_values = [r['dt_ms'] for r in window_results]
        cancel_rates_out = [r.get('cancellation_rate_outside', np.nan) for r in window_results]
        ax_combined_out.plot(dt_values, cancel_rates_out, 'o-', color=colors[window_idx],
                             linewidth=2, markersize=6, label=f'Window {window_idx + 1}')

    ax_combined_out.set_xlabel('dt (ms)')
    ax_combined_out.set_ylabel('Cancellation Rate Outside ROI (%)')
    ax_combined_out.set_title('Combined: All Windows (Outside ROI)')
    ax_combined_out.grid(True, alpha=0.3)
    ax_combined_out.legend()

    plt.tight_layout()

    plot_filename_out = os.path.join(output_dir, "dt_cancellation_analysis_outside.png")
    fig_out.savefig(plot_filename_out, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved plot: {plot_filename_out}")

    csv_filename_out = os.path.join(output_dir, "dt_cancellation_data_outside.csv")
    with open(csv_filename_out, 'w') as f:
        f.write("window,dt_ms,cancellation_rate_outside,total_outside_real,total_outside_cancelled,total_matched_pairs\n")
        for window_idx, window_results in enumerate(all_results):
            if window_results is None:
                continue
            for result in window_results:
                val_out = result.get('cancellation_rate_outside', float('nan'))
                tot_out = result.get('total_outside_real', 0)
                can_out = result.get('total_outside_cancelled', 0)
                f.write(f"{window_idx + 1},{result['dt_ms']},{val_out:.2f},{tot_out},{can_out},{result['total_matched_pairs']}\n")

    print(f"Saved data: {csv_filename_out}")

    return fig

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("=== DT Cancellation Analysis ===")
    print(f"Window predictions directory: {WINDOW_PREDICTIONS_DIR}")
    print(f"Cancellation parameters: {BIN_MS}ms temporal, {R_PIX}px spatial")
    print(f"Polarity mode: {POLARITY_MODE}")
    print(f"ROI: Circle center ({DISC_CENTER_X:.1f}, {DISC_CENTER_Y:.1f}), radius {DISC_RADIUS:.0f}px")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Generate dt values
    dt_values_ms = np.arange(DT_RANGE_MS[0], DT_RANGE_MS[1] + DT_STEP_MS, DT_STEP_MS)
    print(f"Analyzing {len(dt_values_ms)} dt values: {dt_values_ms[0]}ms to {dt_values_ms[-1]}ms")
    
    # Determine window directories (auto-discover if available)
    discovered = discover_window_dirs(WINDOW_PREDICTIONS_DIR)
    all_results = []
    
    if discovered:
        print(f"Found {len(discovered)} window directories")
        if USE_SUBSET_FOR_TESTING:
            print(f"⚠️  LARGE DATASET DETECTED - Using subset mode ({SUBSET_SIZE:,} events per dt)")
            print(f"   Set USE_SUBSET_FOR_TESTING=False for full analysis")
        for window_idx, (window_dir, label) in enumerate(discovered):
            window_results = analyze_window_dt_cancellation_dir(window_idx, window_dir, label, dt_values_ms)
            all_results.append(window_results)
    else:
        print("No window directories found, using fallback window analysis")
        for window_idx, window in enumerate(WINDOWS):
            window_results = analyze_window_dt_cancellation(window_idx, window, dt_values_ms)
            all_results.append(window_results)
    
    # Create plots and save results
    print("\nCreating plots and saving results...")
    fig = create_cancellation_plots(all_results, OUTPUT_DIR)
    
    # Print summary
    print("\n=== Summary ===")
    for window_idx, window_results in enumerate(all_results):
        if window_results is None:
            continue
        label = discovered[window_idx][1] if discovered else f"{WINDOWS[window_idx][0]:.3f}s to {WINDOWS[window_idx][1]:.3f}s"
        print(f"\nWindow {window_idx + 1} ({label}):")
        best_dt = max(window_results, key=lambda x: x['cancellation_rate'])
        worst_dt = min(window_results, key=lambda x: x['cancellation_rate'])
        
        print(f"  Best dt: {best_dt['dt_ms']}ms (cancellation rate: {best_dt['cancellation_rate']:.1f}%)")
        print(f"  Worst dt: {worst_dt['dt_ms']}ms (cancellation rate: {worst_dt['cancellation_rate']:.1f}%)")
        print(f"  ROI events: {best_dt['total_roi_real']}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed_time:.1f}s")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()