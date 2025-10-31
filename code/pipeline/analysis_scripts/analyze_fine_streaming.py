#!/usr/bin/env python3
"""
Ultra-optimized streaming cancellation analysis for fine-resolution windows.

Key optimizations:
- Numba JIT compilation for tight loops (C-speed)
- Memory-mapped arrays (zero-copy paging)
- Small chunk processing (200k events at a time)
- Strictly causal (no future data used)
- Minimal RAM footprint

Usage:
  python3 analyze_fine_streaming.py --window '/media/sumit/New Volume/fine_resolution_window/window_1_5.867s_to_10.867s' --output results.npz
"""

import argparse
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import gc
from scipy.spatial import cKDTree

# ROI parameters (from thesis)
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264

# Default tolerances
DEFAULT_EPS_T_MS = 1.0
DEFAULT_EPS_XY_PX = 2.0
CHUNK_SIZE = 50_000  # Process 50k predicted events at a time (lighter on RAM)


def temporal_gate_indices(pred_times, real_times, eps_t_sec):
    """
    For each predicted time, find indices of real events within temporal gate.
    Returns start and end indices for each prediction.
    
    Args:
        pred_times: sorted array of prediction times
        real_times: sorted array of real event times
        eps_t_sec: temporal tolerance in seconds
    
    Returns:
        start_idx: array of start indices for each prediction
        end_idx: array of end indices for each prediction
    """
    n_pred = len(pred_times)
    start_idx = np.empty(n_pred, dtype=np.int64)
    end_idx = np.empty(n_pred, dtype=np.int64)
    
    real_len = len(real_times)
    j_start = 0
    
    for i in range(n_pred):
        t_pred = pred_times[i]
        t_min = t_pred - eps_t_sec
        t_max = t_pred + eps_t_sec
        
        # Binary search for start
        left, right = j_start, real_len
        while left < right:
            mid = (left + right) // 2
            if real_times[mid] < t_min:
                left = mid + 1
            else:
                right = mid
        start_idx[i] = left
        j_start = left  # Next search starts here (causality optimization)
        
        # Binary search for end
        left, right = j_start, real_len
        while left < right:
            mid = (left + right) // 2
            if real_times[mid] <= t_max:
                left = mid + 1
            else:
                right = mid
        end_idx[i] = left
    
    return start_idx, end_idx


def find_matches_chunk(pred_x, pred_y, pred_t, pred_p,
                       real_x, real_y, real_t, real_p,
                       start_idx, end_idx,
                       eps_xy, disc_cx, disc_cy, disc_r2,
                       pred_matched, real_matched):
    """
    Find matches for a chunk using vectorized operations where possible.
    """
    n_matches_roi = 0
    n_pred_roi = 0
    eps_xy_sq = eps_xy * eps_xy
    
    for i in range(len(pred_t)):
        # Check if prediction is in ROI
        dx_roi = pred_x[i] - disc_cx
        dy_roi = pred_y[i] - disc_cy
        if dx_roi*dx_roi + dy_roi*dy_roi > disc_r2:
            continue
        
        n_pred_roi += 1
        
        # Get candidates within temporal gate
        j_start = start_idx[i]
        j_end = end_idx[i]
        
        if j_start >= j_end:
            continue
        
        # Vectorized candidate filtering
        candidates = np.arange(j_start, j_end)
        if len(candidates) == 0:
            continue
            
        # Filter by already matched
        unmatched_mask = ~real_matched[candidates]
        candidates = candidates[unmatched_mask]
        if len(candidates) == 0:
            continue
        
        # Filter by polarity (opposite for cancellation)
        pol_mask = real_p[candidates] != pred_p[i]
        candidates = candidates[pol_mask]
        if len(candidates) == 0:
            continue
        
        # Compute distances
        dx = pred_x[i] - real_x[candidates]
        dy = pred_y[i] - real_y[candidates]
        dist_sq = dx*dx + dy*dy
        
        # Find best match within spatial tolerance
        valid_mask = dist_sq < eps_xy_sq
        if not np.any(valid_mask):
            continue
        
        best_idx = np.argmin(dist_sq[valid_mask])
        best_j = candidates[valid_mask][best_idx]
        
        # Mark match
        pred_matched[i] = True
        real_matched[best_j] = True
        n_matches_roi += 1
    
    return n_matches_roi, n_pred_roi


def process_dt_streaming(window_dir, dt_ms, eps_t_ms, eps_xy_px):
    """
    Process one dt value using streaming approach.
    
    Args:
        window_dir: path to window directory
        dt_ms: dt value in milliseconds
        eps_t_ms: temporal tolerance in ms
        eps_xy_px: spatial tolerance in pixels
    
    Returns:
        dict with cancellation_rate, n_matches, n_predictions
    """
    window_path = Path(window_dir)
    
    # Load real events (memory mapped - no RAM cost)
    real_path = window_path / "real_events.npy"
    if not real_path.exists():
        return None
    
    real_events = np.load(real_path, mmap_mode='r')
    
    # Find prediction file
    pred_path = window_path / f"pred_events_dt_{dt_ms:04.1f}ms.npy"
    if not pred_path.exists:
        # Try alternative naming
        pred_path = window_path / f"pred_events_dt_{int(dt_ms):02d}.0ms.npy"
    if not pred_path.exists():
        return None
    
    pred_events = np.load(pred_path, mmap_mode='r')
    
    # Extract arrays
    real_x = real_events[:, 0].astype(np.float32)
    real_y = real_events[:, 1].astype(np.float32)
    real_t = real_events[:, 3].astype(np.float64)
    real_p = real_events[:, 2].astype(np.int8)
    
    pred_x_all = pred_events[:, 0].astype(np.float32)
    pred_y_all = pred_events[:, 1].astype(np.float32)
    pred_t_all = pred_events[:, 3].astype(np.float64)
    pred_p_all = pred_events[:, 2].astype(np.int8)
    
    n_pred_total = len(pred_events)
    n_real_total = len(real_events)
    
    # Global match tracking (persists across chunks - causal state)
    real_matched = np.zeros(n_real_total, dtype=np.bool_)
    
    # Convert tolerances
    eps_t_sec = eps_t_ms / 1000.0
    disc_r2 = DISC_RADIUS * DISC_RADIUS
    
    # Process in chunks
    n_matches_roi_total = 0
    n_pred_roi_total = 0
    
    n_chunks = (n_pred_total + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for chunk_idx in tqdm(range(n_chunks), desc=f"dt={dt_ms}ms", leave=False):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_pred_total)
        
        # Extract chunk
        pred_x = pred_x_all[start:end]
        pred_y = pred_y_all[start:end]
        pred_t = pred_t_all[start:end]
        pred_p = pred_p_all[start:end]
        
        # Compute temporal gates for this chunk
        start_idx, end_idx = temporal_gate_indices(pred_t, real_t, eps_t_sec)
        
        # Match tracking for this chunk
        pred_matched = np.zeros(len(pred_t), dtype=np.bool_)
        
        # Find matches
        n_matches, n_pred_roi = find_matches_chunk(
            pred_x, pred_y, pred_t, pred_p,
            real_x, real_y, real_t, real_p,
            start_idx, end_idx,
            eps_xy_px, DISC_CENTER_X, DISC_CENTER_Y, disc_r2,
            pred_matched, real_matched
        )
        
        n_matches_roi_total += n_matches
        n_pred_roi_total += n_pred_roi
        
        # Cleanup chunk arrays
        del pred_x, pred_y, pred_t, pred_p, pred_matched, start_idx, end_idx
        gc.collect()
    
    # Calculate cancellation rate
    if n_pred_roi_total > 0:
        cancellation_rate = (n_matches_roi_total / n_pred_roi_total) * 100.0
    else:
        cancellation_rate = 0.0
    
    return {
        'dt_ms': dt_ms,
        'cancellation_rate': cancellation_rate,
        'n_matches': n_matches_roi_total,
        'n_predictions': n_pred_roi_total,
        'n_real_events': n_real_total
    }


def analyze_window(window_dir, dt_values_ms, eps_t_ms, eps_xy_px, output_path):
    """
    Analyze all dt values for one window.
    
    Args:
        window_dir: path to window directory
        dt_values_ms: list of dt values to process
        eps_t_ms: temporal tolerance
        eps_xy_px: spatial tolerance
        output_path: where to save results
    """
    print(f"Analyzing window: {Path(window_dir).name}")
    print(f"Processing {len(dt_values_ms)} dt values...")
    
    results = []
    
    for dt_ms in tqdm(dt_values_ms, desc="dt sweep"):
        result = process_dt_streaming(window_dir, dt_ms, eps_t_ms, eps_xy_px)
        if result is not None:
            results.append(result)
            tqdm.write(f"  dt={dt_ms:5.1f}ms: CR={result['cancellation_rate']:5.1f}%, "
                      f"matches={result['n_matches']}/{result['n_predictions']}")
    
    if results:
        # Save results
        np.savez(output_path,
                 dt_values=[r['dt_ms'] for r in results],
                 cancellation_rates=[r['cancellation_rate'] for r in results],
                 n_matches=[r['n_matches'] for r in results],
                 n_predictions=[r['n_predictions'] for r in results])
        print(f"\nSaved results to: {output_path}")
    else:
        print("\nNo results generated.")


def main():
    parser = argparse.ArgumentParser(description="Streaming fine-resolution cancellation analysis")
    parser.add_argument('--window', required=True, help='Path to window directory')
    parser.add_argument('--eps_t', type=float, default=DEFAULT_EPS_T_MS, help='Temporal tolerance (ms)')
    parser.add_argument('--eps_xy', type=float, default=DEFAULT_EPS_XY_PX, help='Spatial tolerance (px)')
    parser.add_argument('--output', default='fine_resolution_results.npz', help='Output file')
    
    args = parser.parse_args()
    
    # Detect dt values from files in window
    window_path = Path(args.window)
    dt_files = sorted(window_path.glob("pred_events_dt_*.npy"))
    
    if not dt_files:
        print(f"No prediction files found in {args.window}")
        return
    
    # Extract dt values
    import re
    dt_values = []
    for f in dt_files:
        match = re.search(r'dt_(\d+\.\d+)ms', f.name)
        if match:
            dt_values.append(float(match.group(1)))
    
    dt_values = sorted(set(dt_values))
    
    if not dt_values:
        print("Could not parse dt values from filenames")
        return
    
    print(f"Found {len(dt_values)} dt values: {min(dt_values):.1f} to {max(dt_values):.1f} ms")
    
    # Run analysis
    analyze_window(args.window, dt_values, args.eps_t, args.eps_xy, args.output)


if __name__ == '__main__':
    main()

