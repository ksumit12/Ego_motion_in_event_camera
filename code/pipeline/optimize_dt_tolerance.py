#!/usr/bin/env python3
"""
OPTIMIZED DT and Tolerance Analysis for Ego-Motion Cancellation
Finds optimal combination of dt, spatial tolerance, and temporal tolerance
for maximum cancellation rate with maximum pixel displacement.

PERFORMANCE OPTIMIZATIONS:
- Single KD-Tree build per DT (64x reduction)
- Single neighbor query with max radius (64x reduction) 
- Vectorized operations instead of Python loops
- Cache reuse across tolerance combinations
- Expected 10-15x speedup over original version
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
from scipy.spatial import cKDTree
from itertools import product
from math import ceil

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# =============== Configuration ===============
WINDOW_PREDICTIONS_DIR = "/media/sumit/New Volume1/window_predictions_5s"

# Analysis parameters
DT_VALUES_MS = list(range(0, 21))  # 0 to 20ms
SPATIAL_TOLERANCE_RANGE = (1.0, 5.0, 1.0)  # 1 to 5 pixels, step 1
TEMPORAL_TOLERANCE_RANGE = (1.0, 5.0, 1.0)  # 1 to 5ms, step 1

# Performance optimization - OPTIMIZED VERSION
MAX_EVENTS_PER_COMBINATION = 50000  # Process 50K events per combination
CHUNK_SIZE = 5000  # Process in chunks of 5K (legacy, not used in optimized version)
K_NEAREST_NEIGHBORS = 16  # Increased for single query with max spatial tolerance

# ROI parameters
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264

# Polarity mode
POLARITY_MODE = "opposite"  # "opposite" | "equal" | "ignore"

# Output settings
OUTPUT_DIR = "../dt_tolerance_optimization_results"
PLOT_DPI = 150

# =============== Core Functions ===============

def circle_mask(x, y, cx, cy, r, scale=1.05):
    """Create boolean mask for points inside scaled circle"""
    return (x - cx)**2 + (y - cy)**2 <= (r * scale)**2

def load_npy_or_npz(path, key=None):
    """Load .npy or .npz file with memory mapping"""
    print(f"    Loading file: {os.path.basename(path)}")
    if path.endswith('.npz'):
        with np.load(path) as z:
            if key is not None and key in z:
                return z[key]
            first_key = list(z.files)[0]
            return z[first_key]
    else:
        # Use memory mapping to avoid loading entire file into memory
        return np.load(path, mmap_mode='r')

def find_prediction_files(base_dir, dt_ms):
    """Find prediction files for given dt value"""
    candidates = []
    
    # Check base directory and subdirectories
    search_dirs = [base_dir]
    
    # Also check for window subdirectories
    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith('window_'):
                search_dirs.append(item_path)
    except FileNotFoundError:
        pass
    
    for root_dir in search_dirs:
        try:
            for fname in os.listdir(root_dir):
                # Look for both zero-padded and non-padded formats
                dt_patterns = [
                    f"dt_{dt_ms:02d}ms",  # Zero-padded: dt_00ms, dt_01ms, etc.
                    f"dt_{dt_ms}ms"       # Non-padded: dt_0ms, dt_1ms, etc.
                ]
                
                for pattern in dt_patterns:
                    # Check if the pattern is in the filename (case insensitive)
                    if pattern in fname.lower() and ("pred" in fname.lower() or "predict" in fname.lower()):
                        if fname.endswith('.npy') or fname.endswith('.npz'):
                            candidates.append(os.path.join(root_dir, fname))
        except FileNotFoundError:
            pass
    
    return candidates

def load_combined_events(base_dir, dt_ms):
    """Load combined events for given dt value"""
    # Try to find combined file first
    combined_candidates = []
    
    # Check base directory and window subdirectories
    search_dirs = [base_dir]
    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith('window_'):
                search_dirs.append(item_path)
    except FileNotFoundError:
        pass
    
    for search_dir in search_dirs:
        combined_candidates.extend([
            os.path.join(search_dir, f"combined_events_dt_{dt_ms:02d}ms.npy"),
            os.path.join(search_dir, f"combined_events_dt_{dt_ms:02d}ms.npz"),
            os.path.join(search_dir, f"combined_events_dt_{dt_ms}ms.npy"),
            os.path.join(search_dir, f"combined_events_dt_{dt_ms}ms.npz"),
        ])
    
    for path in combined_candidates:
        if os.path.exists(path):
            return load_npy_or_npz(path, key='combined')
    
    # If no combined file, try to find separate real and pred files
    real_candidates = []
    for search_dir in search_dirs:
        real_candidates.extend([
            os.path.join(search_dir, 'real_events.npy'),
            os.path.join(search_dir, 'real_events.npz'),
        ])
    
    pred_candidates = find_prediction_files(base_dir, dt_ms)
    
    real_events = None
    for path in real_candidates:
        if os.path.exists(path):
            real_events = load_npy_or_npz(path, key='real')
            break
    
    pred_events = None
    for path in pred_candidates:
        if os.path.exists(path):
            pred_events = load_npy_or_npz(path, key='pred')
            break
    
    if real_events is None:
        real_events = np.zeros((0, 4), dtype=np.float32)
    if pred_events is None:
        raise FileNotFoundError(f"No prediction files found for dt={dt_ms}ms")
    
    # Limit the number of events to prevent memory issues
    max_events = 200000  # Limit to 200K events per type
    if len(real_events) > max_events:
        print(f"    Limiting real events from {len(real_events):,} to {max_events:,}")
        real_events = real_events[:max_events]
    if len(pred_events) > max_events:
        print(f"    Limiting pred events from {len(pred_events):,} to {max_events:,}")
        pred_events = pred_events[:max_events]
    
    # Combine into 5-column array with flags
    real_flag = np.zeros((len(real_events), 1), dtype=np.float32)
    pred_flag = np.ones((len(pred_events), 1), dtype=np.float32)
    
    combined = np.vstack([
        np.column_stack([real_events, real_flag]),
        np.column_stack([pred_events, pred_flag])
    ])
    
    return combined[np.argsort(combined[:, 3])]

# ---------- OPTIMIZED: per-DT cache builder ----------
def build_dt_cache(real_events, pred_events, dt_seconds, max_spatial=8.0, k=16):
    """
    Precompute everything expensive once per DT:
      - KDTree on pred
      - KNN (dists, inds) for real -> pred within max_spatial
      - Flattened arrays for times and polarity for fast vector ops
    """
    if len(real_events) == 0 or len(pred_events) == 0:
        return None

    # Cap sizes if you keep the global cap
    if MAX_EVENTS_PER_COMBINATION is not None:
        real_events = real_events[:MAX_EVENTS_PER_COMBINATION]
        pred_events = pred_events[:MAX_EVENTS_PER_COMBINATION]

    real_xy = real_events[:, :2]
    real_pol = real_events[:, 2].astype(np.int8)
    real_ts  = real_events[:, 3].astype(np.float64)

    pred_xy  = pred_events[:, :2]
    pred_pol = pred_events[:, 2].astype(np.int8)
    pred_ts  = pred_events[:, 3].astype(np.float64)

    # KDTree only once
    tree = cKDTree(pred_xy)
    # One KNN query with the max radius and a larger k
    dists, inds = tree.query(real_xy, k=k, distance_upper_bound=max_spatial, workers=-1)

    # Ensure 2D
    if dists.ndim == 1:
        dists = dists[:, None]
        inds  = inds[:,  None]

    # Precompute per-row target times (real + dt)
    target_time = real_ts + float(dt_seconds)

    cache = {
        "real_xy": real_xy,
        "real_pol": real_pol,
        "real_ts": real_ts,
        "pred_pol": pred_pol,
        "pred_ts": pred_ts,
        "dists": dists,     # (N,K)
        "inds": inds,       # (N,K) with num_pred as "no neighbor" sentinel
        "target_time": target_time,
        "num_pred": len(pred_events),
    }
    return cache


# ---------- OPTIMIZED: fast vectorized evaluation for one (S,T) ----------
def evaluate_combo_from_cache(cache, spatial_tol, temporal_tol_ms, polarity_mode="opposite"):
    """
    Vectorized evaluation using precomputed KNN.
    Returns: residual_real_mask, residual_pred_mask, total_matches, mean_disp, frac_ge_3, frac_ge_5
    """
    if cache is None:
        return (np.ones(0, bool), np.ones(0, bool), 0, 0.0, 0.0, 0.0)

    dists = cache["dists"].copy()   # (N,K)
    inds  = cache["inds"].copy()    # (N,K)
    num_pred = cache["num_pred"]
    real_pol = cache["real_pol"]
    pred_pol = cache["pred_pol"]
    pred_ts  = cache["pred_ts"]
    target_time = cache["target_time"]

    N, K = dists.shape

    # Valid neighbor if within radius AND index valid
    valid_spatial = (dists <= float(spatial_tol)) & (inds < num_pred)

    # Temporal mask
    # pred_ts[inds] yields (N,K) with garbage where inds==num_pred; we mask those anyway
    # To be safe, fill invalid inds with 0 to avoid index error then mask them out
    inds_safe = inds.copy()
    inds_safe[inds_safe >= num_pred] = 0
    dt_abs = np.abs(pred_ts[inds_safe] - target_time[:, None])  # (N,K)
    valid_temporal = dt_abs <= (float(temporal_tol_ms) * 1e-3)

    # Polarity mask
    if polarity_mode == "ignore":
        valid_polarity = np.ones((N, K), dtype=bool)
    else:
        pred_p = pred_pol[inds_safe]              # (N,K)
        real_p = real_pol[:, None]                # (N,1)
        if polarity_mode == "equal":
            valid_polarity = (pred_p == real_p)
        else:  # "opposite"
            valid_polarity = (pred_p != real_p)

    valid = valid_spatial & valid_temporal & valid_polarity

    # Cost = distance; invalid -> +inf, we choose argmin
    cost = np.where(valid, dists, np.inf)
    best_k = np.argmin(cost, axis=1)                      # (N,)
    best_cost = cost[np.arange(N), best_k]                # (N,)
    best_pred = inds_safe[np.arange(N), best_k]           # (N,)

    # Rows with no valid candidate have cost == inf
    has_match = np.isfinite(best_cost)

    # Greedy dedup by predicted index:
    # Build candidate list for matched rows, sort by cost asc, keep first per pred idx
    rows = np.nonzero(has_match)[0]
    if rows.size == 0:
        residual_real = np.ones(N, dtype=bool)
        residual_pred = np.ones(num_pred, dtype=bool)
        return (residual_real, residual_pred, 0, 0.0, 0.0, 0.0)

    cand_pred = best_pred[rows]
    cand_cost = best_cost[rows]

    order = np.argsort(cand_cost)              # low cost first
    rows_sorted = rows[order]
    pred_sorted = cand_pred[order]
    cost_sorted = cand_cost[order]

    # keep first occurrence of each pred index
    _, keep_idx = np.unique(pred_sorted, return_index=True)
    chosen_rows = rows_sorted[keep_idx]
    chosen_pred = pred_sorted[keep_idx]
    chosen_cost = cost_sorted[keep_idx]

    # Build residual masks
    residual_real = np.ones(N, dtype=bool)
    residual_pred = np.ones(num_pred, dtype=bool)
    residual_real[chosen_rows] = False
    residual_pred[chosen_pred] = False

    total_matches = chosen_rows.size
    if total_matches == 0:
        return (residual_real, residual_pred, 0, 0.0, 0.0, 0.0)

    # Calculate displacement statistics
    valid_costs = chosen_cost[np.isfinite(chosen_cost)]
    if len(valid_costs) == 0:
        mean_disp = 0.0
        frac_ge_3 = 0.0
        frac_ge_5 = 0.0
    else:
        mean_disp = float(np.mean(valid_costs))
        frac_ge_3 = float(np.mean(valid_costs >= 3.0))
        frac_ge_5 = float(np.mean(valid_costs >= 5.0))

    return (residual_real, residual_pred, int(total_matches), mean_disp, frac_ge_3, frac_ge_5)

def calculate_roi_cancellation_rate(processed_real_events, residual_real_mask, disc_center, disc_radius):
    """Calculate cancellation rate specifically for the circular ROI"""
    if len(processed_real_events) == 0:
        return 0.0, 0, 0
    
    # Filter processed real events to ROI
    cx, cy = disc_center
    roi_mask = circle_mask(processed_real_events[:, 0], processed_real_events[:, 1], cx, cy, disc_radius)
    roi_real_events = processed_real_events[roi_mask]
    
    if len(roi_real_events) == 0:
        return 0.0, 0, 0
    
    # Apply the ROI mask to the residual mask to get ROI residual mask
    roi_residual_mask = residual_real_mask[roi_mask]
    roi_residual_events = roi_real_events[roi_residual_mask]
    
    # Calculate cancellation rate
    total_roi_real = len(roi_real_events)
    total_roi_residual = len(roi_residual_events)
    total_roi_cancelled = total_roi_real - total_roi_residual
    cancellation_rate = (total_roi_cancelled / total_roi_real * 100) if total_roi_real > 0 else 0.0
    
    return cancellation_rate, total_roi_real, total_roi_cancelled

def run_cancellation_analysis(combined_events, dt_ms, temporal_tolerance_ms, spatial_tolerance_pixels, cache=None):
    """Run cancellation analysis for given parameters using optimized cache"""
    if cache is None:
        # Fallback to old method if no cache provided
        real_events = combined_events[combined_events[:, 4] == 0.0]
        pred_events = combined_events[combined_events[:, 4] == 1.0]
        
        if len(real_events) == 0 or len(pred_events) == 0:
            return 0.0, 0, 0, 0.0, 0.0, 0.0
        
        # Build cache for this call
        dt_seconds = dt_ms * 1e-3
        cache = build_dt_cache(real_events, pred_events, dt_seconds, max_spatial=8.0, k=16)
    
    # Use optimized vectorized evaluation
    residual_real_mask, residual_pred_mask, matched_pairs, mean_disp_px, frac_ge_3px, frac_ge_5px = \
        evaluate_combo_from_cache(
            cache,
            spatial_tolerance_pixels,
            temporal_tolerance_ms,
            POLARITY_MODE
        )
    
    # ROI rate on the subset we actually processed (real side only)
    processed_real = cache["real_xy"]
    # but we need full (x,y,t,pol) array for real; rebuild small view:
    rr = combined_events[combined_events[:, 4] == 0.0]
    if MAX_EVENTS_PER_COMBINATION is not None and len(rr) > MAX_EVENTS_PER_COMBINATION:
        rr = rr[:MAX_EVENTS_PER_COMBINATION]
    
    roi_cancellation_rate, total_roi_real, total_roi_cancelled = calculate_roi_cancellation_rate(
        rr, residual_real_mask, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS
    )
    
    return roi_cancellation_rate, total_roi_real, total_roi_cancelled, mean_disp_px, frac_ge_3px, frac_ge_5px

def analyze_all_combinations():
    """Analyze all combinations of dt values and tolerance parameters"""
    print("=== DT and Tolerance Optimization Analysis ===")
    
    # Generate tolerance values
    spatial_values = np.arange(SPATIAL_TOLERANCE_RANGE[0], 
                              SPATIAL_TOLERANCE_RANGE[1] + SPATIAL_TOLERANCE_RANGE[2], 
                              SPATIAL_TOLERANCE_RANGE[2])
    temporal_values = np.arange(TEMPORAL_TOLERANCE_RANGE[0], 
                               TEMPORAL_TOLERANCE_RANGE[1] + TEMPORAL_TOLERANCE_RANGE[2], 
                               TEMPORAL_TOLERANCE_RANGE[2])
    
    print(f"DT values: {DT_VALUES_MS}")
    print(f"Spatial tolerances: {spatial_values}")
    print(f"Temporal tolerances: {temporal_values}")
    
    # Initialize results storage
    results = []
    current_combination = 0
    start_time = time.time()
    
    total_combinations = len(DT_VALUES_MS) * len(spatial_values) * len(temporal_values)
    print(f"Total combinations: {total_combinations}")
    print(f"Performance optimizations:")
    print(f"  - Max events per combination: {MAX_EVENTS_PER_COMBINATION}")
    print(f"  - Chunk size: {CHUNK_SIZE}")
    print(f"  - K-nearest neighbors: {K_NEAREST_NEIGHBORS}")
    
    # Test each dt value
    for dt_ms in DT_VALUES_MS:
        print(f"\nProcessing dt={dt_ms}ms...")
        
        try:
            # Load data for this dt
            print(f"  Attempting to load data for dt={dt_ms}ms...")
            combined_events = load_combined_events(WINDOW_PREDICTIONS_DIR, dt_ms)
            print(f"  Loaded {len(combined_events):,} events")
            
            # Debug: Show what files were found
            pred_candidates = find_prediction_files(WINDOW_PREDICTIONS_DIR, dt_ms)
            print(f"  Found prediction files: {pred_candidates}")
            
            if len(combined_events) == 0:
                print(f"  Warning: No events loaded for dt={dt_ms}ms; skipping")
                continue
            
            # OPTIMIZED: Build cache once per DT and reuse for all tolerance combinations
            real_events = combined_events[combined_events[:, 4] == 0.0]
            pred_events = combined_events[combined_events[:, 4] == 1.0]
            
            cache = build_dt_cache(real_events, pred_events, dt_ms * 1e-3, max_spatial=8.0, k=16)
            if cache is None:
                print("  No events; skipping")
                continue
            
            print(f"  Built cache for {len(real_events):,} real and {len(pred_events):,} predicted events")
            
            # Test all tolerance combinations for this dt
            for spatial_tol in spatial_values:
                for temporal_tol in temporal_values:
                    current_combination += 1
                    
                    if current_combination % 50 == 0:
                        elapsed = time.time() - start_time
                        if current_combination > 0:
                            avg_time_per_combination = elapsed / current_combination
                            remaining_combinations = total_combinations - current_combination
                            eta_seconds = remaining_combinations * avg_time_per_combination
                            eta_minutes = eta_seconds / 60
                            print(f"  Progress: {current_combination}/{total_combinations} ({current_combination/total_combinations*100:.1f}%) - ETA: {eta_minutes:.1f}min")
                    
                    # Run cancellation analysis using cached data
                    roi_cancellation_rate, total_roi_real, total_roi_cancelled, mean_disp_px, frac_ge_3px, frac_ge_5px = run_cancellation_analysis(
                        combined_events, dt_ms, temporal_tol, spatial_tol, cache
                    )
                    
                    # Store results
                    results.append({
                        'dt_ms': dt_ms,
                        'spatial_tolerance': spatial_tol,
                        'temporal_tolerance': temporal_tol,
                        'cancellation_rate': roi_cancellation_rate,
                        'total_roi_real': total_roi_real,
                        'total_roi_cancelled': total_roi_cancelled,
                        'mean_displacement_px': mean_disp_px,
                        'frac_matches_ge_3px': frac_ge_3px,
                        'frac_matches_ge_5px': frac_ge_5px,
                    })
                    
        except FileNotFoundError as e:
            print(f"  Warning: {e}; skipping dt={dt_ms}ms")
            continue
    
    # Final timing summary
    total_time = time.time() - start_time
    print(f"\n=== Analysis Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per combination: {total_time/total_combinations:.2f} seconds")
    print(f"Processed {len(results)} combinations successfully")
    
    return results

def create_3d_surface_plots(results_df, output_dir):
    """Create 3D surface plots showing the relationship between parameters with interactive display"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Enable interactive backend for better 3D viewing
    plt.ion()  # Turn on interactive mode
    
    # Create figure with subplots (adjusted layout for fewer plots)
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Cancellation Rate vs DT and Spatial Tolerance (averaged over temporal)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Group by dt and spatial tolerance, average over temporal
    grouped = results_df.groupby(['dt_ms', 'spatial_tolerance'])['cancellation_rate'].mean().reset_index()
    
    dt_vals = grouped['dt_ms'].values
    spatial_vals = grouped['spatial_tolerance'].values
    cancel_vals = grouped['cancellation_rate'].values
    
    # Create meshgrid for surface plot
    dt_unique = sorted(grouped['dt_ms'].unique())
    spatial_unique = sorted(grouped['spatial_tolerance'].unique())
    DT, SPATIAL = np.meshgrid(dt_unique, spatial_unique)
    
    # Reshape cancellation rates to match meshgrid
    CANCEL = np.zeros_like(DT)
    for i, dt in enumerate(dt_unique):
        for j, spatial in enumerate(spatial_unique):
            mask = (grouped['dt_ms'] == dt) & (grouped['spatial_tolerance'] == spatial)
            if mask.any():
                CANCEL[j, i] = grouped[mask]['cancellation_rate'].iloc[0]
    
    surf1 = ax1.plot_surface(DT, SPATIAL, CANCEL, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('DT (ms)')
    ax1.set_ylabel('Spatial Tolerance (px)')
    ax1.set_zlabel('Cancellation Rate (%)')
    ax1.set_title('Cancellation Rate vs DT & Spatial Tolerance')
    
    # Plot 2: Mean Displacement vs DT and Spatial Tolerance - COMMENTED OUT
    # ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    # 
    # grouped_disp = results_df.groupby(['dt_ms', 'spatial_tolerance'])['mean_displacement_px'].mean().reset_index()
    # DISP = np.zeros_like(DT)
    # for i, dt in enumerate(dt_unique):
    #     for j, spatial in enumerate(spatial_unique):
    #         mask = (grouped_disp['dt_ms'] == dt) & (grouped_disp['spatial_tolerance'] == spatial)
    #         if mask.any():
    #             DISP[j, i] = grouped_disp[mask]['mean_displacement_px'].iloc[0]
    # 
    # surf2 = ax2.plot_surface(DT, SPATIAL, DISP, cmap='plasma', alpha=0.8)
    # ax2.set_xlabel('DT (ms)')
    # ax2.set_ylabel('Spatial Tolerance (px)')
    # ax2.set_zlabel('Mean Displacement (px)')
    # ax2.set_title('Mean Displacement vs DT & Spatial Tolerance')
    
    # Plot 3: Cancellation Rate vs DT and Temporal Tolerance (averaged over spatial)
    ax3 = fig.add_subplot(2, 2, 2, projection='3d')
    
    grouped_temp = results_df.groupby(['dt_ms', 'temporal_tolerance'])['cancellation_rate'].mean().reset_index()
    temporal_unique = sorted(grouped_temp['temporal_tolerance'].unique())
    DT_TEMP, TEMPORAL = np.meshgrid(dt_unique, temporal_unique)
    
    CANCEL_TEMP = np.zeros_like(DT_TEMP)
    for i, dt in enumerate(dt_unique):
        for j, temporal in enumerate(temporal_unique):
            mask = (grouped_temp['dt_ms'] == dt) & (grouped_temp['temporal_tolerance'] == temporal)
            if mask.any():
                CANCEL_TEMP[j, i] = grouped_temp[mask]['cancellation_rate'].iloc[0]
    
    surf3 = ax3.plot_surface(DT_TEMP, TEMPORAL, CANCEL_TEMP, cmap='viridis', alpha=0.8)
    ax3.set_xlabel('DT (ms)')
    ax3.set_ylabel('Temporal Tolerance (ms)')
    ax3.set_zlabel('Cancellation Rate (%)')
    ax3.set_title('Cancellation Rate vs DT & Temporal Tolerance')
    
    # Plot 4: Combined Score (Cancellation Rate * Mean Displacement) - COMMENTED OUT
    # ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    # 
    # # Calculate combined score
    # results_df['combined_score'] = results_df['cancellation_rate'] * results_df['mean_displacement_px']
    # grouped_score = results_df.groupby(['dt_ms', 'spatial_tolerance'])['combined_score'].mean().reset_index()
    # 
    # SCORE = np.zeros_like(DT)
    # for i, dt in enumerate(dt_unique):
    #     for j, spatial in enumerate(spatial_unique):
    #         mask = (grouped_score['dt_ms'] == dt) & (grouped_score['spatial_tolerance'] == spatial)
    #         if mask.any():
    #             SCORE[j, i] = grouped_score[mask]['combined_score'].iloc[0]
    # 
    # surf4 = ax4.plot_surface(DT, SPATIAL, SCORE, cmap='coolwarm', alpha=0.8)
    # ax4.set_xlabel('DT (ms)')
    # ax4.set_ylabel('Spatial Tolerance (px)')
    # ax4.set_zlabel('Combined Score')
    # ax4.set_title('Combined Score (Cancellation × Displacement)')
    
    # Plot 5: Heatmap of best combinations - COMMENTED OUT
    # ax5 = fig.add_subplot(2, 3, 5)
    # 
    # # Create pivot table for heatmap
    # pivot_table = results_df.pivot_table(values='cancellation_rate', 
    #                                    index='spatial_tolerance', 
    #                                    columns='dt_ms', 
    #                                    aggfunc='mean')
    # 
    # sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='viridis', ax=ax5)
    # ax5.set_title('Cancellation Rate Heatmap')
    # ax5.set_xlabel('DT (ms)')
    # ax5.set_ylabel('Spatial Tolerance (px)')
    
    # Plot 6: Displacement vs Cancellation Rate scatter
    ax6 = fig.add_subplot(2, 2, 3)
    
    scatter = ax6.scatter(results_df['mean_displacement_px'], 
                         results_df['cancellation_rate'], 
                         c=results_df['dt_ms'], 
                         cmap='viridis', 
                         alpha=0.6, 
                         s=50)
    ax6.set_xlabel('Mean Displacement (px)')
    ax6.set_ylabel('Cancellation Rate (%)')
    ax6.set_title('Displacement vs Cancellation Rate')
    ax6.grid(True, alpha=0.3)
    
    # Add colorbar for dt values
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('DT (ms)')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_dir, "3d_optimization_analysis.png")
    plt.savefig(plot_filename, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved 3D analysis plot: {plot_filename}")
    
    # Display the plots in real-time
    print("Displaying 3D surface plots in matplotlib window...")
    print("Interactive features:")
    print("  - Click and drag to rotate 3D plots")
    print("  - Mouse wheel to zoom in/out")
    print("  - Right-click and drag to pan")
    print("  - Close window when done viewing")
    
    plt.show(block=True)  # Block until window is closed
    
    return fig

def find_optimal_combinations(results_df):
    """Find optimal combinations based on different criteria"""
    print("\n=== Optimal Combinations Analysis ===")
    
    # 1. Best cancellation rate
    best_cancel = results_df.loc[results_df['cancellation_rate'].idxmax()]
    print(f"Best Cancellation Rate:")
    print(f"  DT: {best_cancel['dt_ms']}ms")
    print(f"  Spatial: {best_cancel['spatial_tolerance']}px")
    print(f"  Temporal: {best_cancel['temporal_tolerance']}ms")
    print(f"  Cancellation Rate: {best_cancel['cancellation_rate']:.2f}%")
    print(f"  Mean Displacement: {best_cancel['mean_displacement_px']:.2f}px")
    
    # 2. Best displacement
    best_disp = results_df.loc[results_df['mean_displacement_px'].idxmax()]
    print(f"\nBest Displacement:")
    print(f"  DT: {best_disp['dt_ms']}ms")
    print(f"  Spatial: {best_disp['spatial_tolerance']}px")
    print(f"  Temporal: {best_disp['temporal_tolerance']}ms")
    print(f"  Cancellation Rate: {best_disp['cancellation_rate']:.2f}%")
    print(f"  Mean Displacement: {best_disp['mean_displacement_px']:.2f}px")
    
    # 3. Best combined score (cancellation rate * displacement)
    results_df['combined_score'] = results_df['cancellation_rate'] * results_df['mean_displacement_px']
    best_combined = results_df.loc[results_df['combined_score'].idxmax()]
    print(f"\nBest Combined Score (Cancellation × Displacement):")
    print(f"  DT: {best_combined['dt_ms']}ms")
    print(f"  Spatial: {best_combined['spatial_tolerance']}px")
    print(f"  Temporal: {best_combined['temporal_tolerance']}ms")
    print(f"  Cancellation Rate: {best_combined['cancellation_rate']:.2f}%")
    print(f"  Mean Displacement: {best_combined['mean_displacement_px']:.2f}px")
    print(f"  Combined Score: {best_combined['combined_score']:.2f}")
    
    # 4. Top 10 combinations by combined score
    print(f"\nTop 10 Combinations by Combined Score:")
    top_10 = results_df.nlargest(10, 'combined_score')
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"  {i:2d}. DT={int(row['dt_ms']):2d}ms, Spatial={row['spatial_tolerance']:1.0f}px, "
              f"Temporal={row['temporal_tolerance']:1.0f}ms, "
              f"Cancel={row['cancellation_rate']:5.1f}%, "
              f"Disp={row['mean_displacement_px']:4.1f}px, "
              f"Score={row['combined_score']:6.1f}")
    
    return best_cancel, best_disp, best_combined, top_10

def main():
    """Main execution function"""
    print("Starting DT and Tolerance Optimization Analysis...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run analysis
    results = analyze_all_combinations()
    
    if not results:
        print("No results obtained. Check data paths and file availability.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    csv_filename = os.path.join(OUTPUT_DIR, "optimization_results.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"Saved results: {csv_filename}")
    
    # Create visualizations
    print("Creating 3D surface plots...")
    fig = create_3d_surface_plots(results_df, OUTPUT_DIR)
    
    # Keep the plots open for interactive viewing
    print("\n" + "="*60)
    print("3D SURFACE PLOTS SUMMARY:")
    print("="*60)
    print("Plot 1: Cancellation Rate vs DT & Spatial Tolerance")
    print("Plot 2: Cancellation Rate vs DT & Temporal Tolerance")
    print("Plot 3: Displacement vs Cancellation Rate Scatter")
    print("(Mean Displacement and Combined Score plots commented out)")
    print("\nInteractive Controls:")
    print("- Click and drag to rotate 3D plots")
    print("- Mouse wheel to zoom in/out")
    print("- Right-click and drag to pan")
    print("- Close window when done viewing")
    print(f"- Plots also saved to: {OUTPUT_DIR}")
    print("="*60)
    
    # Find optimal combinations
    best_cancel, best_disp, best_combined, top_10 = find_optimal_combinations(results_df)
    
    # Save optimal combinations
    optimal_filename = os.path.join(OUTPUT_DIR, "optimal_combinations.csv")
    top_10.to_csv(optimal_filename, index=False)
    print(f"Saved optimal combinations: {optimal_filename}")
    
    print(f"\nAnalysis complete! Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
