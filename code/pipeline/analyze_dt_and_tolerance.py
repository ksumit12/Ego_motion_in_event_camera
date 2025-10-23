#!/usr/bin/env python3
"""
Comprehensive analysis of both dt values and tolerance combinations.
This script tests different dt values AND different spatial/temporal tolerance combinations
to find the optimal parameters for maximum cancellation rate.

Interactive 3D Visualization Dependencies (optional):
    pip install pyvista          # Recommended - creates VTK, PLY, and HTML formats
    pip install mayavi           # Alternative - creates Mayavi scene files
    
The script will work without these packages, but won't save interactive 3D formats.
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
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# No external 3D backends (PyVista/Mayavi) — using Matplotlib only

# =============== Configuration ===============
# Point to the 5s prediction set on the Windows partition by default
WINDOW_PREDICTIONS_DIR = "/media/sumit/New Volume1/window_predictions_5s"
# Optional: a single, shared real events file used for ALL dt/window combinations.
# If None, the script will auto-detect the first available real_events.(npy|npz) under WINDOW_PREDICTIONS_DIR.
REAL_EVENTS_GLOBAL_PATH = None  # e.g., "/media/.../window_predictions_5s/real_events.npy"

# DT values to test. Include 0..20ms if you have predictions for all.
DT_VALUES_MS = list(range(0, 21))

VERBOSE = True
USE_TQDM = True

# Tolerance ranges to test - REDUCED for faster computation
SPATIAL_TOLERANCE_RANGE = (1.0, 3.0, 1.0)  # (min, max, step) in pixels - REDUCED from 10 to 3 values
TEMPORAL_TOLERANCE_RANGE = (0.5, 6.0, 2.0)  # (min, max, step) in milliseconds - Start with 0.5ms for dt=0

# Performance optimization: limit event count per combination
MAX_EVENTS_PER_COMBINATION = 10000  # Process only first 10K events per combinaticon

# Early termination: stop processing if cancellation rate is too low
MIN_CANCELLATION_RATE_THRESHOLD = 5.0  # Stop if cancellation rate < 5%
MIN_EVENTS_FOR_EARLY_TERMINATION = 1000  # Need at least 1K events to make decision

# Polarity mode
POLARITY_MODE = "opposite"  # "opposite" | "equal" | "ignore"

# Disc center coordinates and radius
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264

# Time windows
WINDOWS = [
    (5.000, 5.010),
    (8.200, 8.210),
    (9.000, 9.010),
]

# Output settings
OUTPUT_DIR = "./dt_tolerance_analysis_results"
PLOT_DPI = 150
INTERACTIVE_3D = False  # Matplotlib saves figures; set True to show windows
# If you need to cap processing duration per combination, set this to a float (seconds); None means no cap
MAX_DURATION_S = None

# Flat-folder helper: if no window_* dirs, auto-detect a single window [tmin, tmax]
AUTO_SINGLE_WINDOW = True
# Prefer flat mode if base folder contains global real and pred files
PREFER_FLAT_IF_AVAILABLE = True

# =============== Utility Functions ===============
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

def load_npy_or_npz(path, key=None, use_mmap=True):
    """Load .npy or .npz. If .npz, use provided key or the first array.
    Use memory mapping for large files to reduce memory usage."""
    if path.endswith('.npz'):
        with np.load(path) as z:
            if key is not None and key in z:
                return z[key]
            first_key = list(z.files)[0]
            return z[first_key]
    else:
        # Use memory mapping for large files to reduce memory usage
        if use_mmap:
            return np.load(path, mmap_mode='r')
        else:
            return np.load(path)

def _discover_window_dirs(base_dir: str):
    try:
        entries = sorted([p for p in Path(base_dir).iterdir() if p.is_dir() and p.name.startswith('window_')], key=lambda p: p.name)
        return [(str(p), p.name) for p in entries]
    except FileNotFoundError:
        return []

def _detect_time_range_from_any(base_dir: str):
    """Try to infer [tmin, tmax] from real or any pred/combined in base_dir."""
    # Prefer global real
    for name, key in (("real_events.npy", 'real'), ("real_events.npz", 'real')):
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            arr = load_npy_or_npz(p, key=key)
            if arr is not None and arr.shape[1] >= 4:
                return float(np.min(arr[:,3])), float(np.max(arr[:,3]))
    # Try any combined
    for dt_ms in range(0, 21):
        for name, key in ((f"combined_events_dt_{dt_ms:02d}ms.npy", 'combined'), (f"combined_events_dt_{dt_ms:02d}ms.npz", 'combined')):
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                arr = load_npy_or_npz(p, key=key)
                if arr is not None and arr.shape[1] >= 4:
                    return float(np.min(arr[:,3])), float(np.max(arr[:,3]))
    # Try any pred
    for dt_ms in range(0, 21):
        for name, key in ((f"pred_events_dt_{dt_ms:02d}ms.npy", 'pred'), (f"pred_events_dt_{dt_ms:02d}ms.npz", 'pred')):
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                arr = load_npy_or_npz(p, key=key)
                if arr is not None and arr.shape[1] >= 4:
                    return float(np.min(arr[:,3])), float(np.max(arr[:,3]))
    return None

def _has_global_real(base_dir: str) -> bool:
    return os.path.exists(os.path.join(base_dir, "real_events.npy")) or os.path.exists(os.path.join(base_dir, "real_events.npz"))

def _has_any_base_pred(base_dir: str, dt_values: list) -> bool:
    for dt_ms in dt_values:
        for name in (f"pred_events_dt_{dt_ms:02d}ms.npy", f"pred_events_dt_{dt_ms:02d}ms.npz",
                     f"combined_events_dt_{dt_ms:02d}ms.npy", f"combined_events_dt_{dt_ms:02d}ms.npz"):
            if os.path.exists(os.path.join(base_dir, name)):
                return True
    return False

def _available_dt_in_base(base_dir: str, dt_values: list) -> list:
    found = []
    for dt_ms in dt_values:
        has_dt = False
        for name in (f"pred_events_dt_{dt_ms:02d}ms.npy", f"pred_events_dt_{dt_ms:02d}ms.npz",
                     f"combined_events_dt_{dt_ms:02d}ms.npy", f"combined_events_dt_{dt_ms:02d}ms.npz"):
            if os.path.exists(os.path.join(base_dir, name)):
                has_dt = True
                break
        if has_dt:
            found.append(dt_ms)
    return found

def _available_dt_in_dir(dir_path: str, dt_values: list) -> list:
    found = []
    for dt_ms in dt_values:
        has_dt = False
        for name in (f"pred_events_dt_{dt_ms:02d}ms.npy", f"pred_events_dt_{dt_ms:02d}ms.npz",
                     f"pred_events_dt_{dt_ms}ms.npy", f"pred_events_dt_{dt_ms}ms.npz",
                     f"combined_events_dt_{dt_ms:02d}ms.npy", f"combined_events_dt_{dt_ms:02d}ms.npz",
                     f"combined_events_dt_{dt_ms}ms.npy", f"combined_events_dt_{dt_ms}ms.npz"):
            if os.path.exists(os.path.join(dir_path, name)):
                has_dt = True
                break
        if has_dt:
            found.append(dt_ms)
    return found

def _auto_detect_global_real_events(base_dir: str):
    """Search recursively for a single real_events.(npy|npz) file and return its path or None."""
    base = Path(base_dir)
    for name in ("real_events.npy", "real_events.npz"):
        # Prefer a file at the root
        root_path = base / name
        if root_path.exists():
            return str(root_path)
    # Otherwise, search recursively and take the first match
    for p in base.rglob("real_events.npy"):
        return str(p)
    for p in base.rglob("real_events.npz"):
        return str(p)
    return None

def try_load_combined_or_split(window_dir, dt_ms):
    """Try combined; else use real+pred split and combine on the fly."""
    # Look for combined files in window_dir, or fallback to base dir
    combined_candidates = [
        # zero-padded
        os.path.join(window_dir, f"combined_events_dt_{dt_ms:02d}ms.npy"),
        os.path.join(window_dir, f"combined_events_dt_{dt_ms:02d}ms.npz"),
        os.path.join(WINDOW_PREDICTIONS_DIR, f"combined_events_dt_{dt_ms:02d}ms.npy"),
        os.path.join(WINDOW_PREDICTIONS_DIR, f"combined_events_dt_{dt_ms:02d}ms.npz"),
        # non-padded
        os.path.join(window_dir, f"combined_events_dt_{dt_ms}ms.npy"),
        os.path.join(window_dir, f"combined_events_dt_{dt_ms}ms.npz"),
        os.path.join(WINDOW_PREDICTIONS_DIR, f"combined_events_dt_{dt_ms}ms.npy"),
        os.path.join(WINDOW_PREDICTIONS_DIR, f"combined_events_dt_{dt_ms}ms.npz"),
    ]
    for path in combined_candidates:
        if os.path.exists(path):
            arr = load_npy_or_npz(path, key='combined')
            return arr
    # Real events: either from a global shared file or per-window
    global_real = REAL_EVENTS_GLOBAL_PATH if REAL_EVENTS_GLOBAL_PATH else _auto_detect_global_real_events(WINDOW_PREDICTIONS_DIR)
    real_candidates = []
    if global_real:
        real_candidates.append(global_real)
    # Also allow per-window files if present
    real_candidates.extend([
        os.path.join(window_dir, 'real_events.npy'),
        os.path.join(window_dir, 'real_events.npz'),
    ])
    # Predictions may live per-window or at the base folder
    pred_candidates = [
        # zero-padded
        os.path.join(window_dir, f"pred_events_dt_{dt_ms:02d}ms.npy"),
        os.path.join(window_dir, f"pred_events_dt_{dt_ms:02d}ms.npz"),
        os.path.join(WINDOW_PREDICTIONS_DIR, f"pred_events_dt_{dt_ms:02d}ms.npy"),
        os.path.join(WINDOW_PREDICTIONS_DIR, f"pred_events_dt_{dt_ms:02d}ms.npz"),
        # non-padded
        os.path.join(window_dir, f"pred_events_dt_{dt_ms}ms.npy"),
        os.path.join(window_dir, f"pred_events_dt_{dt_ms}ms.npz"),
        os.path.join(WINDOW_PREDICTIONS_DIR, f"pred_events_dt_{dt_ms}ms.npy"),
        os.path.join(WINDOW_PREDICTIONS_DIR, f"pred_events_dt_{dt_ms}ms.npz"),
    ]
    real = None
    for p in real_candidates:
        if os.path.exists(p):
            real = load_npy_or_npz(p, key='real')
            break
    if real is None:
        # Allow running even when real events are absent; treat as empty
        real = np.zeros((0, 4), dtype=np.float32)
    pred = None
    for p in pred_candidates:
        if os.path.exists(p):
            pred = load_npy_or_npz(p, key='pred')
            break
    if pred is None:
        # Fallback: flexibly search for any predicted file containing the dt token
        # within both the window directory and the base directory
        dt_token_padded = f"dt_{dt_ms:02d}ms"
        dt_token = f"dt_{dt_ms}ms"
        candidates = []
        for root_dir in (window_dir, WINDOW_PREDICTIONS_DIR):
            try:
                for fname in os.listdir(root_dir):
                    lower = fname.lower()
                    if (dt_token_padded in lower or dt_token in lower) and ("pred" in lower or "predict" in lower):
                        if fname.endswith('.npy') or fname.endswith('.npz'):
                            candidates.append(os.path.join(root_dir, fname))
            except FileNotFoundError:
                pass
        for p in candidates:
            try:
                pred = load_npy_or_npz(p, key='pred')
                break
            except Exception:
                try:
                    pred = load_npy_or_npz(p, key=None)
                    break
                except Exception:
                    continue
    if pred is None:
        raise FileNotFoundError(f"Missing predictions for dt={dt_ms}ms in {window_dir}")
    # Combine into 5-col array with flags and sort by t
    real_flag = np.zeros((len(real), 1), dtype=np.float32)
    pred_flag = np.ones((len(pred), 1), dtype=np.float32)
    stacks = []
    if len(real) > 0:
        stacks.append(np.column_stack([real, real_flag]))
    if len(pred) > 0:
        stacks.append(np.column_stack([pred, pred_flag]))
    combined = np.vstack(stacks) if stacks else np.zeros((0, 5), dtype=np.float32)
    return combined[np.argsort(combined[:, 3])]

def cancel_events_time_aware(real_events, predicted_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels):
    """True temporal gate with spatial KDTree (k-NN, bounded radius, parallel).

    Returns:
        unmatched_real_mask: bool array len(real_events)
        unmatched_predicted_mask: bool array len(predicted_events)
        total_matches: int
        mean_displacement_px: float (average pixel distance of matched pairs)
        frac_ge_3px: float (fraction of matches with displacement >= 3px)
        frac_ge_5px: float (fraction of matches with displacement >= 5px)
    """
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    if num_real == 0 or num_predicted == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0

    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    matched_real = np.zeros(num_real, dtype=bool)
    matched_predicted = np.zeros(num_predicted, dtype=bool)
    total_matches = 0
    sum_displacement = 0.0
    count_ge_3 = 0
    count_ge_5 = 0

    # Limit events for performance if specified
    # For dt=0, we need more events to get meaningful cancellation rates
    if MAX_EVENTS_PER_COMBINATION is not None:
        # Use more events for dt=0 since we expect high cancellation rates
        max_events = MAX_EVENTS_PER_COMBINATION * 5 if dt_seconds == 0 else MAX_EVENTS_PER_COMBINATION
        
        if num_real > max_events:
            real_events = real_events[:max_events]
            num_real = len(real_events)
        if num_predicted > max_events:
            predicted_events = predicted_events[:max_events]
            num_predicted = len(predicted_events)

    pred_tree = cKDTree(predicted_events[:, :2])
    # Adaptive chunking: smaller chunks for larger datasets
    if num_real > 50000:
        chunk_size = min(2000, num_real)  # Very small chunks for large datasets
    elif num_real > 20000:
        chunk_size = min(5000, num_real)  # Medium chunks
    else:
        chunk_size = min(10000, num_real)  # Larger chunks for small datasets
    num_chunks = ceil(max(num_real, 1) / max(chunk_size, 1))
    pbar = None
    if VERBOSE and USE_TQDM and tqdm is not None:
        pbar = tqdm(total=num_chunks, desc="    matching chunks", leave=False)

    K = 4  # Reduced from 8 to 4 for faster queries
    for chunk_start in range(0, num_real, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_real)
        chunk_real = real_events[chunk_start:chunk_end]
        chunk_target_times = chunk_real[:, 3] + dt_seconds

        dists, inds = pred_tree.query(chunk_real[:, :2], k=K, distance_upper_bound=spatial_tolerance_pixels, workers=-1)
        if K == 1:
            dists = dists[:, None]
            inds = inds[:, None]

        for i, (real_event, target_time) in enumerate(zip(chunk_real, chunk_target_times)):
            real_idx = chunk_start + i
            if matched_real[real_idx]:
                continue
            spatial_candidates = inds[i]
            valid_mask = (spatial_candidates < num_predicted)
            if not np.any(valid_mask):
                continue
            spatial_candidates = spatial_candidates[valid_mask]
            avail_mask = ~matched_predicted[spatial_candidates]
            if not np.any(avail_mask):
                continue
            spatial_candidates = spatial_candidates[avail_mask]
            candidate_times = predicted_events[spatial_candidates, 3]
            temporal_mask = np.abs(candidate_times - target_time) <= temporal_tolerance_s
            if not np.any(temporal_mask):
                continue
            final_candidates = spatial_candidates[temporal_mask]
            cand_events = predicted_events[final_candidates]
            # Polarity
            real_pol = real_event[2]
            if POLARITY_MODE == "ignore":
                pol_mask = np.ones(len(cand_events), dtype=bool)
            elif POLARITY_MODE == "equal":
                pol_mask = (cand_events[:, 2] == real_pol)
            else:
                pol_mask = (cand_events[:, 2] != real_pol)
            if not np.any(pol_mask):
                continue
            valid_candidates = final_candidates[pol_mask]
            valid_events = cand_events[pol_mask]
            distances = np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1)
            argmin_idx = int(np.argmin(distances))
            best_candidate = valid_candidates[argmin_idx]
            best_dist_px = float(np.sqrt(distances[argmin_idx]))
            matched_real[real_idx] = True
            matched_predicted[best_candidate] = True
            total_matches += 1
            sum_displacement += best_dist_px
            if best_dist_px >= 3.0:
                count_ge_3 += 1
            if best_dist_px >= 5.0:
                count_ge_5 += 1
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()
    mean_disp = (sum_displacement / total_matches) if total_matches > 0 else 0.0
    frac_ge_3 = (count_ge_3 / total_matches) if total_matches > 0 else 0.0
    frac_ge_5 = (count_ge_5 / total_matches) if total_matches > 0 else 0.0
    return ~matched_real, ~matched_predicted, total_matches, mean_disp, frac_ge_3, frac_ge_5

def cancel_events_in_time_bin(real_events, predicted_events, spatial_tolerance_pixels):
    """Match real and predicted events within a time bin and return unmatched events"""
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    
    if num_real == 0 or num_predicted == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    # Create a tree for fast spatial lookup of predicted events
    predicted_tree = cKDTree(predicted_events[:, :2])
    
    # Find closest predicted event for each real event within tolerance
    distances, closest_indices = predicted_tree.query(
        real_events[:, :2], k=1, distance_upper_bound=spatial_tolerance_pixels
    )
    
    # Find real events that have a valid match within tolerance
    valid_matches = np.where(closest_indices < num_predicted)[0]
    
    if len(valid_matches) == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    # Create list of potential matches with distance, real index, predicted index
    potential_matches = []
    for real_idx in valid_matches:
        predicted_idx = int(closest_indices[real_idx])
        distance = float(distances[real_idx])
        
        # Check if polarities are compatible
        if check_polarity_match(real_events[real_idx, 2], predicted_events[predicted_idx, 2]):
            potential_matches.append((distance, real_idx, predicted_idx))
    
    if len(potential_matches) == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    # Sort by distance (closest matches first)
    potential_matches.sort(key=lambda match: match[0])
    
    # Greedily assign matches
    used_predicted = set()
    matched_real = np.zeros(num_real, bool)
    matched_predicted = np.zeros(num_predicted, bool)
    
    for distance, real_idx, predicted_idx in potential_matches:
        if predicted_idx not in used_predicted:
            used_predicted.add(predicted_idx)
            matched_real[real_idx] = True
            matched_predicted[predicted_idx] = True
    
    # Return unmatched events (inverse of matched)
    unmatched_real = ~matched_real
    unmatched_predicted = ~matched_predicted
    num_matches = int(matched_real.sum())
    
    return unmatched_real, unmatched_predicted, num_matches

def time_edges(tmin, tmax, bin_ms):
    """Generate time bin edges"""
    w = bin_ms * 1e-3
    n = int(np.ceil((tmax - tmin) / w)) + 1
    return tmin + np.arange(n+1) * w

def run_cancellation_true_gate(combined_events, temporal_tolerance_ms, spatial_tolerance_pixels, dt_ms=None):
    """Run ego-motion cancellation using TRUE temporal gate (no binning)."""
    real_events = combined_events[combined_events[:, 4] == 0.0]
    pred_events = combined_events[combined_events[:, 4] == 1.0]
    if len(real_events) == 0 or len(pred_events) == 0:
        return (
            np.zeros((0, 5), dtype=combined_events.dtype),
            np.zeros((0, 5), dtype=combined_events.dtype),
            0,
            0.0,
            0.0,
            0.0,
        )

    # Estimate dt from data (robust median of positive deltas)
    sample_real_times = real_events[:min(1000, len(real_events)), 3]
    sample_pred_times = pred_events[:min(1000, len(pred_events)), 3]
    time_diffs = []
    for rt in sample_real_times:
        closest_pred_times = sample_pred_times[np.abs(sample_pred_times - rt) < 0.1]
        if len(closest_pred_times) > 0:
            closest_diff = np.min(closest_pred_times - rt)
            if closest_diff > 0:
                time_diffs.append(closest_diff)
    # Use provided dt_ms or estimate from data
    if dt_ms is not None:
        dt_seconds = dt_ms * 1e-3
    else:
        dt_seconds = np.median(time_diffs) if len(time_diffs) > 0 else 0.002

    # Pre-filter ROI to accelerate
    cx, cy, r = DISC_CENTER_X, DISC_CENTER_Y, DISC_RADIUS
    r_pred = r + spatial_tolerance_pixels + 2.0
    def _circle_mask_xy(arr, rad):
        return (arr[:, 0] - cx)**2 + (arr[:, 1] - cy)**2 <= (rad * 1.05)**2
    real_events_roi = real_events[_circle_mask_xy(real_events, r)]
    pred_events_roi = pred_events[_circle_mask_xy(pred_events, r_pred)]

    unmatched_real_mask, unmatched_pred_mask, total_matches, mean_disp, frac_ge_3, frac_ge_5 = cancel_events_time_aware(
        real_events_roi, pred_events_roi, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels
    )

    residual_real = real_events_roi[unmatched_real_mask]
    residual_pred = pred_events_roi[unmatched_pred_mask]
    return residual_real, residual_pred, total_matches, mean_disp, frac_ge_3, frac_ge_5

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

def analyze_all_combinations():
    """Analyze all combinations of dt values and tolerance parameters"""
    print("=== Comprehensive DT and Tolerance Analysis ===")
    
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
    
    # Resolve windows: either discovered window_* or a single auto-detected window
    discovered = _discover_window_dirs(WINDOW_PREDICTIONS_DIR)
    windows_to_use = WINDOWS
    working_dir = WINDOW_PREDICTIONS_DIR
    # Decide mode
    flat_available = _has_global_real(WINDOW_PREDICTIONS_DIR)
    use_flat = False
    if flat_available and PREFER_FLAT_IF_AVAILABLE:
        use_flat = True  # force flat if global real exists
    elif len(discovered) == 1:
        # Single window directory with all files -> treat as flat within that directory
        use_flat = True
        working_dir, _ = discovered[0]
    elif not discovered and AUTO_SINGLE_WINDOW:
        use_flat = True
    # If base folder has prediction files for dt, prefer flat mode even if window_* dirs exist
    elif _has_any_base_pred(WINDOW_PREDICTIONS_DIR, DT_VALUES_MS):
        use_flat = True

    if use_flat:
        tr = _detect_time_range_from_any(working_dir)
        if tr is not None:
            windows_to_use = [(tr[0], tr[1])]
            mode_note = "base" if working_dir == WINDOW_PREDICTIONS_DIR else "single window dir"
            print(f"Mode: FLAT ({mode_note}). Using detected window: {tr[0]:.3f}s to {tr[1]:.3f}s")
        else:
            print("Mode: FLAT. Warning: could not detect time range; using configured WINDOWS")
    else:
        print(f"Mode: WINDOW DIRS. Found {len(discovered)} window_* directories")

    # Narrow DT list to those actually present in chosen mode to avoid noisy warnings
    dt_values_to_use = DT_VALUES_MS
    if use_flat:
        if working_dir == WINDOW_PREDICTIONS_DIR:
            avail = _available_dt_in_base(working_dir, DT_VALUES_MS)
        else:
            avail = _available_dt_in_dir(working_dir, DT_VALUES_MS)
        if len(avail) > 0:
            dt_values_to_use = avail
            print(f"Detected dt files in {working_dir}: {dt_values_to_use}")
        else:
            print("Warning: No dt files detected in base folder; proceeding with configured DT_VALUES_MS")

    total_combinations = len(dt_values_to_use) * len(spatial_values) * len(temporal_values) * len(windows_to_use)
    print(f"Total combinations: {total_combinations}")
    print(f"Performance optimizations:")
    print(f"  - Spatial tolerance range: {len(spatial_values)} values (reduced from 10)")
    print(f"  - Temporal tolerance range: {len(temporal_values)} values (reduced from 10)")
    print(f"  - Max events per combination: {MAX_EVENTS_PER_COMBINATION}")
    print(f"  - Chunk size: 5000 (reduced from 50000)")
    print(f"  - K-nearest neighbors: 4 (reduced from 8)")

    # Test each dt value
    for dt_ms in dt_values_to_use:
        print(f"\nProcessing dt={dt_ms}ms...")
        
        # Test each window
        for window_idx, window in enumerate(windows_to_use):
            t0, t1 = window
            
            # Load prediction data for this window and dt (supports combined or split)
            # If flat folder mode (no discovered windows), use base dir directly
            if not use_flat:
                window_dir = os.path.join(WINDOW_PREDICTIONS_DIR, f"window_{window_idx + 1}_{t0:.3f}s_to_{t1:.3f}s")
            else:
                window_dir = working_dir
            try:
                combined_events = try_load_combined_or_split(window_dir, dt_ms)
                # Optional cap if MAX_DURATION_S is set (keep None to disable)
                if MAX_DURATION_S is not None and len(combined_events) > 0:
                    tmin = float(combined_events[0, 3])
                    tmax = float(combined_events[-1, 3])
                    if (tmax - tmin) > MAX_DURATION_S:
                        slice_end = tmin + MAX_DURATION_S
                        time_mask = (combined_events[:, 3] >= tmin) & (combined_events[:, 3] < slice_end)
                        combined_events = combined_events[time_mask]
            except FileNotFoundError as e:
                print(f"  Warning: {e}; skipping window {window_idx + 1}")
                continue
            
            # Test all tolerance combinations for this dt and window
            for spatial_tol in spatial_values:
                for temporal_tol in temporal_values:
                    current_combination += 1
                    
                    if current_combination % 10 == 0:  # More frequent progress updates
                        elapsed = time.time() - start_time
                        if current_combination > 0:
                            avg_time_per_combination = elapsed / current_combination
                            remaining_combinations = total_combinations - current_combination
                            eta_seconds = remaining_combinations * avg_time_per_combination
                            eta_minutes = eta_seconds / 60
                            print(f"  Progress: {current_combination}/{total_combinations} ({current_combination/total_combinations*100:.1f}%) - ETA: {eta_minutes:.1f}min")
                        else:
                            print(f"  Progress: {current_combination}/{total_combinations} ({current_combination/total_combinations*100:.1f}%)")
                    
                    # Run cancellation with these tolerances
                    # TRUE temporal gate matching (from visualize_time_window)
                    residual_real, residual_pred, matched_pairs, mean_disp_px, frac_ge_3px, frac_ge_5px = run_cancellation_true_gate(
                        combined_events, temporal_tol, spatial_tol, dt_ms
                    )
                    
                    # Debug: Check if we have events at all
                    if dt_ms == 0 and current_combination <= 3:  # Only for first few combinations
                        real_events = combined_events[combined_events[:, 4] == 0.0]
                        pred_events = combined_events[combined_events[:, 4] == 1.0]
                        print(f"    Debug: real_events={len(real_events)}, pred_events={len(pred_events)}, "
                              f"matched_pairs={matched_pairs}")
                        print(f"    Debug: Using max_events={MAX_EVENTS_PER_COMBINATION * 5} for dt=0")
                        print(f"    Debug: dt_ms={dt_ms}, dt_seconds={dt_ms * 1e-3}")
                        
                        # Check if real and pred events are identical (they should be at dt=0)
                        if len(real_events) > 0 and len(pred_events) > 0:
                            sample_real = real_events[:5]
                            sample_pred = pred_events[:5]
                            print(f"    Debug: Sample real times: {sample_real[:, 3]}")
                            print(f"    Debug: Sample pred times: {sample_pred[:, 3]}")
                            print(f"    Debug: Time differences: {sample_pred[:, 3] - sample_real[:, 3]}")
                    
                    # Calculate ROI cancellation rate using the processed subset
                    # We need to use the same subset that was processed for matching
                    processed_real = combined_events[combined_events[:, 4] == 0.0]
                    processed_pred = combined_events[combined_events[:, 4] == 1.0]
                    
                    # Limit to the same subset that was processed
                    if MAX_EVENTS_PER_COMBINATION is not None:
                        max_events = MAX_EVENTS_PER_COMBINATION * 5 if dt_ms == 0 else MAX_EVENTS_PER_COMBINATION
                        if len(processed_real) > max_events:
                            processed_real = processed_real[:max_events]
                        if len(processed_pred) > max_events:
                            processed_pred = processed_pred[:max_events]
                    
                    # Create combined subset for ROI calculation
                    real_flag = np.zeros((len(processed_real), 1), dtype=np.float32)
                    pred_flag = np.ones((len(processed_pred), 1), dtype=np.float32)
                    combined_subset = np.vstack([
                        np.column_stack([processed_real, real_flag]),
                        np.column_stack([processed_pred, pred_flag])
                    ])
                    
                    roi_cancellation_rate, total_roi_real, total_roi_cancelled = calculate_roi_cancellation_rate(
                        combined_subset, residual_real, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS
                    )
                    
                    # Debug: Print cancellation rate for dt=0 to understand what's happening
                    if dt_ms == 0:
                        print(f"    dt=0: spatial={spatial_tol:.1f}px, temporal={temporal_tol:.1f}ms, "
                              f"cancellation={roi_cancellation_rate:.1f}%, roi_real={total_roi_real}, "
                              f"roi_cancelled={total_roi_cancelled}, matched_pairs={matched_pairs}")
                        print(f"    Debug: processed_real={len(processed_real)}, processed_pred={len(processed_pred)}")
                        print(f"    Debug: residual_real={len(residual_real)}")
                    
                    # Early termination: skip remaining spatial/temporal combinations if cancellation rate is too low
                    # BUT NOT for dt=0 where we expect high cancellation rates
                    if (dt_ms > 0 and total_roi_real >= MIN_EVENTS_FOR_EARLY_TERMINATION and 
                        roi_cancellation_rate < MIN_CANCELLATION_RATE_THRESHOLD):
                        print(f"    Early termination: cancellation rate {roi_cancellation_rate:.1f}% < {MIN_CANCELLATION_RATE_THRESHOLD}%")
                        # Skip remaining temporal tolerances for this spatial tolerance
                        break
                    
                    # Store results
                    results.append({
                        'dt_ms': dt_ms,
                        'window_idx': window_idx + 1,
                        'window_start': t0,
                        'window_end': t1,
                        'spatial_tolerance': spatial_tol,
                        'temporal_tolerance': temporal_tol,
                        'cancellation_rate': roi_cancellation_rate,
                        'total_roi_real': total_roi_real,
                        'total_roi_cancelled': total_roi_cancelled,
                        'total_matched_pairs': matched_pairs,
                        'mean_disp_px': mean_disp_px,
                        'frac_matches_ge_3px': frac_ge_3px,
                        'frac_matches_ge_5px': frac_ge_5px,
                    })
    
    # Final timing summary
    total_time = time.time() - start_time
    print(f"\n=== Analysis Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per combination: {total_time/total_combinations:.2f} seconds")
    print(f"Processed {len(results)} combinations successfully")
    
    return results

def save_interactive_3d_surface(X, Y, Z, best_dt, output_dir):
    """Save interactive 3D surface plots using PyVista and/or Mayavi"""
    
    # Create separate folder for interactive 3D files
    interactive_3d_dir = os.path.join(output_dir, "interactive_3d")
    os.makedirs(interactive_3d_dir, exist_ok=True)
    print(f"Creating interactive 3D files in: {interactive_3d_dir}")
    
    # Save data as numpy arrays for manual loading later
    data_path = os.path.join(interactive_3d_dir, "3d_surface_data.npz")
    np.savez(data_path, X=X, Y=Y, Z=Z, dt=best_dt)
    print(f"Saved 3D surface data: {data_path}")
    print("  You can load this data later with: data = np.load('3d_surface_data.npz')")
    
    # PyVista format (VTK-based, widely supported)
    if HAS_PYVISTA:
        try:
            # Create structured grid
            grid = pv.StructuredGrid(X, Y, Z)
            grid["Cancellation Rate"] = Z.ravel(order='F')
            
            # Save as VTK file (can be opened in ParaView, PyVista, etc.)
            vtk_path = os.path.join(interactive_3d_dir, "tolerance_3d_surface.vtk")
            grid.save(vtk_path)
            print(f"Saved PyVista/VTK format: {vtk_path}")
            print("  Open with: pv.read('tolerance_3d_surface.vtk').plot()")
            
            # Create matplotlib-style interactive plotter
            plotter = pv.Plotter(window_size=[1200, 900], off_screen=True)
            
            # Add the surface with matplotlib-like styling
            mesh = plotter.add_mesh(grid, scalars="Cancellation Rate", 
                                  cmap='viridis', 
                                  show_scalar_bar=True,
                                  scalar_bar_args={
                                      'title': 'Cancellation Rate (%)',
                                      'title_font_size': 16,
                                      'label_font_size': 14,
                                      'n_labels': 6,
                                      'italic': False,
                                      'width': 0.08,
                                      'height': 0.6
                                  },
                                  smooth_shading=True)
            
            # Add axes with matplotlib-style labels
            plotter.show_axes()
            
            # Add grid - this creates a more matplotlib-like appearance
            plotter.show_grid(
                xlabel='Spatial Tolerance (pixels)',
                ylabel='Temporal Tolerance (ms)', 
                zlabel='Cancellation Rate (%)',
                font_size=14,
                grid=True,
                location='outer'
            )
            
            # Set title
            plotter.add_text(f'Cancellation Rate vs Tolerance Parameters (dt = {best_dt}ms)', 
                           position='upper_edge', font_size=16, color='black')
            
            # Set camera to isometric view (similar to matplotlib)
            plotter.camera_position = 'iso'
            plotter.camera.azimuth = -45
            plotter.camera.elevation = 30
            
            # Set background color to match matplotlib
            plotter.background_color = 'white'
            
            # Save interactive HTML file with matplotlib styling
            html_path = os.path.join(interactive_3d_dir, "tolerance_3d_surface_interactive.html")
            plotter.export_html(html_path, backend='pythreejs')
            print(f"Saved interactive HTML: {html_path}")
            print("  Open in web browser for interactive viewing")
            
            # Also save a high-quality static image for comparison
            static_img_path = os.path.join(interactive_3d_dir, "tolerance_3d_surface_pyvista.png")
            plotter.screenshot(static_img_path, scale=2)
            print(f"Saved PyVista static image: {static_img_path}")
            
        except Exception as e:
            print(f"Warning: PyVista save failed: {e}")
    else:
        print("PyVista not available. Install with: pip install pyvista")
    
    # Mayavi format
    if HAS_MAYAVI:
        try:
            # Create Mayavi figure
            mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
            surf = mlab.surf(X, Y, Z, colormap='viridis')
            mlab.axes(xlabel='Spatial Tolerance (pixels)', 
                     ylabel='Temporal Tolerance (ms)', 
                     zlabel='Cancellation Rate (%)')
            mlab.title(f'Cancellation Rate vs Tolerance Parameters (dt = {best_dt}ms)')
            mlab.colorbar(surf, title='Cancellation Rate (%)')
            
            # Save as Mayavi scene file
            mayavi_path = os.path.join(interactive_3d_dir, "tolerance_3d_surface.mv2")
            mlab.savefig(mayavi_path)
            print(f"Saved Mayavi format: {mayavi_path}")
            print("  Open with Mayavi2 application or mlab.load_engine()")
            
            mlab.close()
            
        except Exception as e:
            print(f"Warning: Mayavi save failed: {e}")
    else:
        print("Mayavi not available. Install with: pip install mayavi")
    
    # Create a simple Python script to reload and view the data
    viewer_script = f'''#!/usr/bin/env python3
"""
Interactive 3D Surface Viewer
Load and display the 3D tolerance surface plot interactively.

Usage:
    python view_3d_surface.py

Requirements:
    pip install pyvista  # or mayavi
"""

import numpy as np
import os

# Try PyVista first (recommended)
try:
    import pyvista as pv
    
    def view_with_pyvista():
        # Load the VTK file
        if os.path.exists("tolerance_3d_surface.vtk"):
            grid = pv.read("tolerance_3d_surface.vtk")
        else:
            # Load from numpy data
            data = np.load("3d_surface_data.npz")
            X, Y, Z = data['X'], data['Y'], data['Z']
            dt = data['dt']
            grid = pv.StructuredGrid(X, Y, Z)
            grid["Cancellation Rate"] = Z.ravel(order='F')
        
        # Create matplotlib-style plotter
        plotter = pv.Plotter(window_size=[1200, 900])
        
        # Add the surface with matplotlib-like styling
        plotter.add_mesh(grid, scalars="Cancellation Rate", 
                        cmap='viridis', 
                        show_scalar_bar=True,
                        scalar_bar_args={{
                            'title': 'Cancellation Rate (%)',
                            'title_font_size': 16,
                            'label_font_size': 14,
                            'n_labels': 6,
                            'italic': False,
                            'width': 0.08,
                            'height': 0.6
                        }},
                        smooth_shading=True)
        
        # Add axes and grid with matplotlib-style labels
        plotter.show_axes()
        plotter.show_grid(
            xlabel='Spatial Tolerance (pixels)',
            ylabel='Temporal Tolerance (ms)', 
            zlabel='Cancellation Rate (%)',
            font_size=14,
            grid=True,
            location='outer'
        )
        
        # Set title
        if os.path.exists("3d_surface_data.npz"):
            data = np.load("3d_surface_data.npz")
            dt = data['dt']
            plotter.add_text(f'Cancellation Rate vs Tolerance Parameters (dt = {{dt}}ms)', 
                           position='upper_edge', font_size=16, color='black')
        
        # Set camera to match matplotlib view
        plotter.camera_position = 'iso'
        plotter.camera.azimuth = -45
        plotter.camera.elevation = 30
        plotter.background_color = 'white'
        
        plotter.show()
    
    if __name__ == "__main__":
        view_with_pyvista()

except ImportError:
    # Fallback to Mayavi
    try:
        from mayavi import mlab
        
        def view_with_mayavi():
            if os.path.exists("tolerance_3d_surface.mv2"):
                # Load Mayavi scene
                mlab.load_engine()
                # Note: Direct scene loading might need manual implementation
                print("Load the .mv2 file manually in Mayavi2 application")
            else:
                # Load from numpy data
                data = np.load("3d_surface_data.npz")
                X, Y, Z = data['X'], data['Y'], data['Z']
                dt = data['dt']
                
                mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
                surf = mlab.surf(X, Y, Z, colormap='viridis')
                mlab.axes(xlabel='Spatial Tolerance (pixels)', 
                         ylabel='Temporal Tolerance (ms)', 
                         zlabel='Cancellation Rate (%)')
                mlab.title(f'Cancellation Rate vs Tolerance Parameters (dt = {{dt}}ms)')
                mlab.colorbar(surf, title='Cancellation Rate (%)')
                mlab.show()
        
        if __name__ == "__main__":
            view_with_mayavi()
            
    except ImportError:
        print("Neither PyVista nor Mayavi available.")
        print("Install one of them:")
        print("  pip install pyvista  # Recommended")
        print("  pip install mayavi   # Alternative")
        
        # Fallback: show how to load data manually
        print("\\nYou can still load the data manually:")
        print("  data = np.load('3d_surface_data.npz')")
        print("  X, Y, Z = data['X'], data['Y'], data['Z']")
        print("  # Then use your preferred 3D plotting library")
'''
    
    script_path = os.path.join(interactive_3d_dir, "view_3d_surface.py")
    with open(script_path, 'w') as f:
        f.write(viewer_script)
    print(f"Created viewer script: {script_path}")
    print("  Run with: python view_3d_surface.py")
    
    # Also create a README file in the interactive_3d folder
    readme_content = f"""# Interactive 3D Surface Files

This folder contains interactive 3D visualization files for the tolerance analysis surface plot.

## Files:
- `3d_surface_data.npz` - Raw NumPy data (X, Y, Z coordinates and dt value)
- `tolerance_3d_surface.vtk` - VTK format (ParaView, PyVista compatible)
- `tolerance_3d_surface.ply` - PLY format (widely supported)
- `tolerance_3d_surface_interactive.html` - Interactive HTML (open in web browser)
- `tolerance_3d_surface.mv2` - Mayavi scene file
- `view_3d_surface.py` - Python script to view the surface

## How to View:

### Option 1: Python Script (Recommended)
```bash
cd interactive_3d
python view_3d_surface.py
```

### Option 2: PyVista (if installed)
```python
import pyvista as pv
mesh = pv.read('tolerance_3d_surface.vtk')
mesh.plot(scalars="Cancellation Rate", cmap='viridis')
```

### Option 3: Web Browser
Open `tolerance_3d_surface_interactive.html` in any web browser

### Option 4: ParaView
Open `tolerance_3d_surface.vtk` in ParaView application

## Surface Data:
- **dt value**: {best_dt}ms
- **X-axis**: Spatial Tolerance (pixels)
- **Y-axis**: Temporal Tolerance (ms)  
- **Z-axis**: Cancellation Rate (%)
- **Color**: Cancellation Rate (%)

## Dependencies:
```bash
pip install pyvista  # Recommended
# or
pip install mayavi  # Alternative
```
"""
    
    readme_path = os.path.join(interactive_3d_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"Created README: {readme_path}")

def create_comprehensive_plots(results_df, output_dir):
    """Create comprehensive visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. DT vs Cancellation Rate (averaged across tolerances)
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    # Average across all tolerance combinations for each dt
    dt_avg = results_df.groupby('dt_ms')['cancellation_rate'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    ax1.errorbar(dt_avg['dt_ms'], dt_avg['mean'], yerr=dt_avg['std'], 
                capsize=5, capthick=2, marker='o', markersize=8, linewidth=2)
    ax1.fill_between(dt_avg['dt_ms'], dt_avg['min'], dt_avg['max'], alpha=0.3)
    ax1.set_xlabel('dt (ms)')
    ax1.set_ylabel('Cancellation Rate (%)')
    ax1.set_title('Cancellation Rate vs dt (averaged across all tolerance combinations)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Mean ± Std', 'Min-Max Range'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dt_vs_cancellation_rate.png"), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    
    # 2. Tolerance heatmaps for best dt values
    best_dt_values = results_df.groupby('dt_ms')['cancellation_rate'].mean().nlargest(3).index
    fig2, axes = plt.subplots(1, len(best_dt_values), figsize=(5*len(best_dt_values), 4))
    if len(best_dt_values) == 1:
        axes = [axes]
    
    for i, dt_val in enumerate(best_dt_values):
        ax = axes[i]
        
        # Average across all windows for this dt
        dt_data = results_df[results_df['dt_ms'] == dt_val]
        pivot_data = dt_data.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean().reset_index()
        pivot_table = pivot_data.pivot_table(values='cancellation_rate', index='temporal_tolerance', columns='spatial_tolerance')
        
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='viridis', ax=ax)
        ax.set_title(f'dt = {dt_val}ms')
        ax.set_xlabel('Spatial Tolerance (pixels)')
        ax.set_ylabel('Temporal Tolerance (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tolerance_heatmaps_best_dt.png"), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    
    # 3. Global 3D cube plot: x=dt, y=spatial tol, z=temporal tol, color=rate
    fig3 = plt.figure(figsize=(12, 9))
    ax3 = fig3.add_subplot(111, projection='3d')
    xs = results_df['dt_ms'].astype(float).to_numpy()
    ys = results_df['spatial_tolerance'].astype(float).to_numpy()
    zs = results_df['temporal_tolerance'].astype(float).to_numpy()
    cs = results_df['cancellation_rate'].astype(float).to_numpy()
    sc = ax3.scatter(xs, ys, zs, c=cs, cmap='viridis', s=15, alpha=0.9)
    cb = fig3.colorbar(sc, ax=ax3, shrink=0.6, pad=0.1)
    cb.set_label('Cancellation Rate (%)')
    ax3.set_xlabel('dt (ms)')
    ax3.set_ylabel('Spatial Tol (px)')
    ax3.set_zlabel('Temporal Tol (ms)')
    ax3.set_title('Cancellation over (dt, spatial tol, temporal tol)')
    ax3.view_init(elev=22, azim=-45)
    out_path = os.path.join(output_dir, 'global_3d_dt_spatial_temporal.png')
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches='tight')
    if INTERACTIVE_3D:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig3)
    print(f"Saved 3D global plot: {out_path}")
    print(f"Saved comprehensive plots to {output_dir}")

    # 4. Best-per-dt table
    best_per_dt = _compute_best_per_dt(results_df)
    best_csv = os.path.join(output_dir, "best_per_dt.csv")
    best_per_dt.to_csv(best_csv, index=False)
    print(f"Saved best-per-dt summary: {best_csv}")

    # 5. Pareto fronts per dt (write small CSVs)
    pareto_dir = os.path.join(output_dir, "pareto_fronts")
    os.makedirs(pareto_dir, exist_ok=True)
    for dt_val in sorted(results_df['dt_ms'].unique()):
        dt_data = results_df[results_df['dt_ms'] == dt_val]
        pf = _pareto_front(dt_data)
        pf.to_csv(os.path.join(pareto_dir, f"dt_{int(dt_val):02d}_pareto.csv"), index=False)
    print(f"Saved Pareto fronts to {pareto_dir}")

def find_optimal_parameters(results_df):
    """Find the optimal parameter combinations"""
    print("\n=== Optimal Parameter Analysis ===")
    
    # Find best tolerance combination for each dt value
    print("Best tolerance combinations for each dt value:")
    print("=" * 60)
    
    for dt_val in sorted(results_df['dt_ms'].unique()):
        dt_data = results_df[results_df['dt_ms'] == dt_val]
        
        # Find best combination for this dt (averaged across windows)
        dt_avg = dt_data.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean()
        best_tolerance = dt_avg.idxmax()
        best_rate = dt_avg[best_tolerance]
        
        print(f"dt = {int(dt_val):2d}ms: spatial={best_tolerance[0]:.1f}px, "
              f"temporal={best_tolerance[1]:.1f}ms, rate={best_rate:.2f}%")
        
        # Show top 3 tolerance combinations for this dt
        top3_dt = dt_avg.nlargest(3)
        print(f"  Top 3 for dt={int(dt_val)}ms:")
        for i, ((spat, temp), rate) in enumerate(top3_dt.items(), 1):
            print(f"    {i}. spatial={spat:.1f}px, temporal={temp:.1f}ms, rate={rate:.2f}%")
        print()
    
    # Find best overall combination (excluding dt=0)
    best_overall = results_df.loc[results_df['cancellation_rate'].idxmax()]
    
    print("Best overall combination:")
    print(f"  dt: {int(best_overall['dt_ms'])}ms")
    print(f"  Spatial tolerance: {best_overall['spatial_tolerance']:.1f} pixels")
    print(f"  Temporal tolerance: {best_overall['temporal_tolerance']:.1f} ms")
    print(f"  Cancellation rate: {best_overall['cancellation_rate']:.2f}%")
    print(f"  Window: {best_overall['window_start']:.3f}s to {best_overall['window_end']:.3f}s")
    
    # Find best dt (averaged across all tolerances)
    dt_avg = results_df.groupby('dt_ms')['cancellation_rate'].mean()
    best_dt = dt_avg.idxmax()
    print(f"\nBest dt (averaged across all tolerances): {int(best_dt)}ms ({dt_avg[best_dt]:.2f}%)")
    
    # Show dt performance ranking
    print(f"\nDT performance ranking (averaged across all tolerances):")
    dt_ranking = dt_avg.sort_values(ascending=False)
    for i, (dt_val, rate) in enumerate(dt_ranking.items(), 1):
        print(f"  {i}. dt={int(dt_val):2d}ms: {rate:.2f}%")
    
    # Top 10 combinations overall
    print(f"\nTop 10 combinations overall:")
    top10 = results_df.nlargest(10, 'cancellation_rate')
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"  {i:2d}. dt={int(row['dt_ms']):2d}ms, spatial={row['spatial_tolerance']:.1f}px, "
              f"temporal={row['temporal_tolerance']:.1f}ms, rate={row['cancellation_rate']:.2f}%")
    
    return best_overall

def _compute_best_per_dt(results_df: pd.DataFrame) -> pd.DataFrame:
    """Pick best per dt prioritizing cancellation_rate, tie-break by mean_disp_px desc."""
    best_rows = []
    for dt_val in sorted(results_df['dt_ms'].unique()):
        dt_data = results_df[results_df['dt_ms'] == dt_val]
        # sort by cancellation desc, mean_disp desc, then smallest tolerances
        dt_sorted = dt_data.sort_values(by=['cancellation_rate', 'mean_disp_px', 'spatial_tolerance', 'temporal_tolerance'], ascending=[False, False, True, True])
        best_rows.append(dt_sorted.iloc[0])
    return pd.DataFrame(best_rows).reset_index(drop=True)

def _pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Pareto front over objectives: maximize cancellation_rate, mean_disp_px; minimize spatial/temporal tolerance."""
    pts = df[['cancellation_rate','mean_disp_px','spatial_tolerance','temporal_tolerance']].to_numpy()
    is_dominated = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if is_dominated[i]:
            continue
        for j in range(len(pts)):
            if i == j:
                continue
            # j dominates i if: better or equal in all, strictly better in at least one
            better_or_equal = (
                pts[j,0] >= pts[i,0] and  # higher cancellation better
                pts[j,1] >= pts[i,1] and  # higher displacement better
                pts[j,2] <= pts[i,2] and  # lower spatial tol better
                pts[j,3] <= pts[i,3]      # lower temporal tol better
            )
            strictly_better = (
                (pts[j,0] > pts[i,0]) or
                (pts[j,1] > pts[i,1]) or
                (pts[j,2] < pts[i,2]) or
                (pts[j,3] < pts[i,3])
            )
            if better_or_equal and strictly_better:
                is_dominated[i] = True
                break
    return df[~is_dominated].copy()

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("=== Comprehensive DT and Tolerance Analysis ===")
    print(f"DT values: {DT_VALUES_MS}")
    print(f"Spatial range: {SPATIAL_TOLERANCE_RANGE[0]} to {SPATIAL_TOLERANCE_RANGE[1]} (step: {SPATIAL_TOLERANCE_RANGE[2]})")
    print(f"Temporal range: {TEMPORAL_TOLERANCE_RANGE[0]} to {TEMPORAL_TOLERANCE_RANGE[1]} (step: {TEMPORAL_TOLERANCE_RANGE[2]})")
    print(f"Polarity mode: {POLARITY_MODE}")
    print(f"ROI: Circle center ({DISC_CENTER_X:.1f}, {DISC_CENTER_Y:.1f}), radius {DISC_RADIUS:.0f}px")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Interactive display: {INTERACTIVE_3D}")
    
    # Run comprehensive analysis
    results = analyze_all_combinations()
    
    if not results:
        print("No results generated. Check if prediction data exists.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_filename = os.path.join(OUTPUT_DIR, "comprehensive_analysis_results.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"\nSaved results: {csv_filename}")
    
    # Create visualizations
    print("\nCreating comprehensive visualizations...")
    create_comprehensive_plots(results_df, OUTPUT_DIR)
    
    # Find optimal parameters
    best_parameters = find_optimal_parameters(results_df)
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Total combinations tested: {len(results_df)}")
    print(f"Average cancellation rate: {results_df['cancellation_rate'].mean():.2f}%")
    print(f"Best cancellation rate: {results_df['cancellation_rate'].max():.2f}%")
    print(f"Worst cancellation rate: {results_df['cancellation_rate'].min():.2f}%")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed_time:.1f}s")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
