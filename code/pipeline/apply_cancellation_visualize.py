#!/usr/bin/env python3
"""
Clean ego-motion cancellation visualization script.
Implements true temporal gate matching with optimized performance.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive execution
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree
import os

# Configuration
COMBINED_PATH = "./combined_events_with_predictions.npy"
BIN_MS = 5.0
R_PIX = 2.0
POLARITY_MODE = "opposite"
IMG_W, IMG_H = 1280, 720
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264
WINDOWS = [(5.000, 5.010), (8.200, 8.210), (9.000, 9.010)]
OUTPUT_DIR = "./main_results"

def load_combined(path):
    """Load combined events data"""
    arr = np.load(path, mmap_mode="r")
    if not np.all(arr[:-1, 3] <= arr[1:, 3]):
        arr = arr[np.argsort(arr[:, 3])]
    print(f"Loaded {len(arr):,} events (real={int(np.sum(arr[:,4]==0.0)):,}, pred={int(np.sum(arr[:,4]==1.0)):,})")
    return arr

def check_polarity_match(real_polarity, predicted_polarity):
    """Check polarity compatibility based on mode"""
    if POLARITY_MODE == "ignore":
        return True
    elif POLARITY_MODE == "equal":
        return real_polarity == predicted_polarity
    else:  # "opposite"
        return real_polarity != predicted_polarity

def cancel_events_time_aware(real_events, predicted_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels):
    """
    Match events using true temporal gate: |t_j-(t_i+dt)| <= tolerance
    Optimized with spatial indexing for large datasets
    """
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    
    if num_real == 0 or num_predicted == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    matched_real = np.zeros(num_real, dtype=bool)
    matched_predicted = np.zeros(num_predicted, dtype=bool)
    total_matches = 0
    
    # Create spatial KDTree for fast spatial search
    pred_tree = cKDTree(predicted_events[:, :2])
    
    # Process in chunks for memory efficiency
    chunk_size = min(50000, num_real)
    print(f"  Processing {num_real:,} real events in chunks of {chunk_size:,}")
    
    for chunk_start in range(0, num_real, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_real)
        chunk_real = real_events[chunk_start:chunk_end]
        chunk_target_times = chunk_real[:, 3] + dt_seconds
        
        # Find spatial candidates for entire chunk
        spatial_candidates_list = pred_tree.query_ball_point(chunk_real[:, :2], spatial_tolerance_pixels)
        
        # Process each event in chunk
        for i, (real_event, target_time, spatial_candidates) in enumerate(zip(chunk_real, chunk_target_times, spatial_candidates_list)):
            real_idx = chunk_start + i
            if matched_real[real_idx]:
                continue
            
            if len(spatial_candidates) == 0:
                continue
            
            # Filter available candidates
            spatial_candidates = np.array(spatial_candidates)
            available_candidates = spatial_candidates[~matched_predicted[spatial_candidates]]
            
            if len(available_candidates) == 0:
                continue
            
            # Apply temporal gate
            candidate_times = predicted_events[available_candidates, 3]
            temporal_mask = np.abs(candidate_times - target_time) <= temporal_tolerance_s
            
            if not np.any(temporal_mask):
                continue
            
            # Get final candidates
            final_candidates = available_candidates[temporal_mask]
            candidate_events = predicted_events[final_candidates]
            
            # Check polarity constraint
            real_polarity = real_event[2]
            pred_polarities = candidate_events[:, 2]
            
            if POLARITY_MODE == "ignore":
                polarity_matches = np.ones(len(candidate_events), dtype=bool)
            elif POLARITY_MODE == "equal":
                polarity_matches = (pred_polarities == real_polarity)
            else:  # "opposite"
                polarity_matches = (pred_polarities != real_polarity)
            
            if np.any(polarity_matches):
                # Find closest valid candidate
                valid_candidates = final_candidates[polarity_matches]
                valid_events = candidate_events[polarity_matches]
                distances = np.sqrt(np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1))
                best_candidate = valid_candidates[np.argmin(distances)]
                
                matched_real[real_idx] = True
                matched_predicted[best_candidate] = True
                total_matches += 1
        
        # Progress update
        progress = (chunk_end / num_real) * 100
        print(f"  Progress: {progress:.1f}% - {total_matches:,} matches found", end="\r")
    
    print()  # New line after progress
    return ~matched_real, ~matched_predicted, total_matches

def run_cancellation(combined_events, temporal_tolerance_ms, spatial_tolerance_pixels):
    """Run ego-motion cancellation using true temporal gate"""
    real_events = combined_events[combined_events[:, 4] == 0.0]
    pred_events = combined_events[combined_events[:, 4] == 1.0]
    
    total_real_events = len(real_events)
    total_predicted_events = len(pred_events)
    
    print(f"Start cancellation: temporal_tol={temporal_tolerance_ms} ms, spatial_tol={spatial_tolerance_pixels} px")
    print(f"Events: {total_real_events:,} real, {total_predicted_events:,} predicted")
    start_time = time.time()
    
    # Estimate dt from data
    if len(real_events) > 0 and len(pred_events) > 0:
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
    else:
        dt_seconds = 0.002
    
    print(f"Estimated dt: {dt_seconds*1000:.1f}ms")
    
    # Run cancellation
    unmatched_real_mask, unmatched_predicted_mask, total_matched_pairs = cancel_events_time_aware(
        real_events, pred_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels
    )
    
    residual_real_events = real_events[unmatched_real_mask]
    residual_predicted_events = pred_events[unmatched_predicted_mask]
    
    elapsed_time = time.time() - start_time
    
    # Calculate rates
    real_cancelled = total_real_events - len(residual_real_events)
    pred_cancelled = total_predicted_events - len(residual_predicted_events)
    real_cancellation_rate = (real_cancelled / total_real_events * 100) if total_real_events > 0 else 0
    pred_cancellation_rate = (pred_cancelled / total_predicted_events * 100) if total_predicted_events > 0 else 0
    
    print(f"Done in {elapsed_time:.1f}s")
    print(f"Real: {total_real_events:,} -> residual {len(residual_real_events):,} (cancelled {real_cancelled:,}, {real_cancellation_rate:.1f}%)")
    print(f"Pred: {total_predicted_events:,} -> residual {len(residual_predicted_events):,} (cancelled {pred_cancelled:,}, {pred_cancellation_rate:.1f}%)")
    print(f"Matched pairs: {total_matched_pairs:,}")
    print(f"Using TRUE temporal gate: |t_j-(t_i+dt)| <= {temporal_tolerance_ms}ms")
    
    return residual_real_events, residual_predicted_events

def circle_mask(x, y, cx, cy, r, scale=1.05):
    """Create boolean mask for points inside scaled circle"""
    return (x - cx)**2 + (y - cy)**2 <= (r * scale)**2

def events_in_window(events, t0, t1):
    """Create boolean mask for events within time window"""
    return (events[:, 3] >= t0) & (events[:, 3] < t1)

def make_frame(H, W, xs, ys, ps=None):
    """Create pixel-true frame from event coordinates and polarities"""
    frame = np.zeros((H, W), dtype=np.int32)
    if len(xs) == 0:
        return frame
    
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs_valid = xs[valid]
    ys_valid = ys[valid]
    
    if ps is not None:
        ps_valid = ps[valid]
        np.add.at(frame, (ys_valid, xs_valid), ps_valid)
    else:
        np.add.at(frame, (ys_valid, xs_valid), 1)
    
    return frame

def convert_polarity_to_signed(polarity_values):
    """Convert polarity from 0/1 to -1/+1 for visualization"""
    return np.where(polarity_values > 0.5, 1, -1).astype(np.int16)

def create_per_pixel_count_image(width, height, events):
    """Create per-pixel signed count image using bilinear interpolation"""
    if len(events) == 0:
        return np.zeros((height, width), dtype=np.float32)
    
    x = events[:, 0].astype(np.float32)
    y = events[:, 1].astype(np.float32)
    s = convert_polarity_to_signed(events[:, 2]).astype(np.float32)
    
    # Bilinear interpolation
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0
    
    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy
    
    img = np.zeros((height, width), dtype=np.float32)
    
    # Bounds checking masks
    m00 = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    m10 = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)
    m01 = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)
    m11 = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)
    
    # Accumulate weighted contributions
    np.add.at(img, (y0[m00], x0[m00]), s[m00] * w00[m00])
    np.add.at(img, (y0[m10], x1[m10]), s[m10] * w10[m10])
    np.add.at(img, (y1[m01], x0[m01]), s[m01] * w01[m01])
    np.add.at(img, (y1[m11], x1[m11]), s[m11] * w11[m11])
    
    return img

def build_window_images(combined_events, time_window, image_width, image_height):
    """Create per-pixel count images for real, predicted, and combined events"""
    start_time, end_time = time_window
    
    time_mask = (combined_events[:, 3] >= start_time) & (combined_events[:, 3] < end_time)
    window_events = combined_events[time_mask]
    
    real_events = window_events[window_events[:, 4] == 0.0][:, :3]
    predicted_events = window_events[window_events[:, 4] == 1.0][:, :3]
    all_events = window_events[:, :3]
    
    real_image = create_per_pixel_count_image(image_width, image_height, real_events)
    predicted_image = create_per_pixel_count_image(image_width, image_height, predicted_events)
    combined_image = create_per_pixel_count_image(image_width, image_height, all_events)
    
    return real_image, predicted_image, combined_image, len(real_events), len(predicted_events)

def normalize_images_for_display(real_image, predicted_image, combined_image):
    """Normalize three images to the same scale for fair comparison"""
    real_max = np.abs(real_image).max()
    predicted_max = np.abs(predicted_image).max()
    combined_max = np.abs(combined_image).max()
    overall_max = max(real_max, predicted_max, combined_max, 1)
    
    def normalize_single_image(image):
        normalized = (image / (2.0 * overall_max)) + 0.5
        return np.clip(normalized, 0.0, 1.0)
    
    normalized_real = normalize_single_image(real_image)
    normalized_predicted = normalize_single_image(predicted_image)
    normalized_combined = normalize_single_image(combined_image)
    
    return normalized_real, normalized_predicted, normalized_combined, int(overall_max)

def make_panel_figure(combined, resid_real, resid_pred, window, img_w=IMG_W, img_h=IMG_H, use_gray=False):
    """Create comprehensive analysis panel for a time window"""
    t0, t1 = window

    # Extract events for this window
    allm = (combined[:,3] >= t0) & (combined[:,3] < t1)
    w_all = combined[allm]
    w_real = w_all[w_all[:,4] == 0.0]
    w_pred = w_all[w_all[:,4] == 1.0]
    wr = resid_real[(resid_real[:,3] >= t0) & (resid_real[:,3] < t1)]
    wp = resid_pred[(resid_pred[:,3] >= t0) & (resid_pred[:,3] < t1)]
    cancelled = len(w_real) - len(wr)

    # Build per-pixel images
    img_r, img_p, img_c, nr, npred = build_window_images(combined, window, img_w, img_h)
    img_r_n, img_p_n, img_c_n, max_abs = normalize_images_for_display(img_r, img_p, img_c)

    # Create figure
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(nrows=3, ncols=3, height_ratios=[1.1, 1.1, 1.0], figure=fig)

    # Row 1: Scatter plots
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    
    ax0.scatter(w_real[:,0], w_real[:,1], s=2, alpha=0.6, c="tab:blue")
    ax0.set_title(f"Real ({len(w_real):,})")
    
    ax1.scatter(w_pred[:,0], w_pred[:,1], s=2, alpha=0.6, c="tab:red")
    ax1.set_title(f"Predicted ({len(w_pred):,})")
    
    ax2.scatter(wr[:,0], wr[:,1], s=2, alpha=0.7, c="tab:blue", label=f"Real ({len(wr):,})")
    ax2.scatter(wp[:,0], wp[:,1], s=2, alpha=0.7, c="tab:red", label=f"Pred ({len(wp):,})")
    ax2.legend()
    
    for ax in (ax0, ax1, ax2):
        ax.set_xlim(0, img_w)
        ax.set_ylim(0, img_h)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
    ax0.set_ylabel("y")

    # Row 2: Per-pixel images
    cmap = "gray" if use_gray else "seismic"
    b0 = fig.add_subplot(gs[1, 0])
    b1 = fig.add_subplot(gs[1, 1])
    b2 = fig.add_subplot(gs[1, 2])
    
    im0 = b0.imshow(img_r_n, cmap=cmap, origin="upper", vmin=0, vmax=1)
    b0.set_title(f"Real (N={nr:,})")
    b0.set_xlabel("x")
    b0.set_ylabel("y")
    b0.grid(alpha=0.2)
    
    im1 = b1.imshow(img_p_n, cmap=cmap, origin="upper", vmin=0, vmax=1)
    b1.set_title(f"Pred (N={npred:,})")
    b1.set_xlabel("x")
    b1.set_ylabel("y")
    b1.grid(alpha=0.2)
    
    im2 = b2.imshow(img_c_n, cmap=cmap, origin="upper", vmin=0, vmax=1)
    b2.set_title("Combined")
    b2.set_xlabel("x")
    b2.set_ylabel("y")
    b2.grid(alpha=0.2)
    
    # Add disc overlay to combined plot
    if cmap == "seismic":
        disc_circle = plt.Circle((DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS, 
                               fill=True, color='yellow', alpha=0.15, linewidth=0)
        b2.add_patch(disc_circle)
        
        disc_outline = plt.Circle((DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS, 
                                fill=False, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
        b2.add_patch(disc_outline)
        
        b2.plot(DISC_CENTER_X, DISC_CENTER_Y, 'yo', markersize=6, markeredgecolor='black', 
               markeredgewidth=1, label=f'Disc Center')
        b2.legend(loc='upper right', fontsize=7)
    
    for ax_im, im in zip((b0, b1, b2), (im0, im1, im2)):
        cb = fig.colorbar(im, ax=ax_im, fraction=0.046, pad=0.04)
        cb.set_label("signed count (Σ polarity)")
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.set_ticklabels([f"-{max_abs}", "0", f"+{max_abs}"])

    # Row 3: Histograms
    c0 = fig.add_subplot(gs[2, 0])
    c1 = fig.add_subplot(gs[2, 1])
    c2 = fig.add_subplot(gs[2, 2])
    
    def plot_histogram(ax, data, title):
        ax.hist(data, bins=100, log=True, edgecolor="k", alpha=0.85)
        if data.size:
            med = float(np.median(data))
            p95 = float(np.percentile(data, 95))
            ax.axvline(med, color="tab:orange", ls="--", lw=1.2, label=f"median={med:.1f}")
            ax.axvline(p95, color="tab:green", ls="--", lw=1.0, label=f"95%={p95:.1f}")
        ax.set_xlabel("|signed count|")
        ax.set_ylabel("pixels (log)")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(title)
    
    m_r = np.abs(img_r.ravel())
    m_p = np.abs(img_p.ravel())
    m_c = np.abs(img_c.ravel())
    nz_r = m_r[m_r > 0]
    nz_p = m_p[m_p > 0]
    nz_c = m_c[m_c > 0]

    plot_histogram(c0, nz_r, f"Real |count| (nonzero px={nz_r.size:,})")
    plot_histogram(c1, nz_p, f"Pred |count| (nonzero px={nz_p.size:,})")
    plot_histogram(c2, nz_c, f"Combined |count| (nonzero px={nz_c.size:,})")

    # Calculate statistics
    total_real_events = len(w_real)
    actual_cancellation_rate = (cancelled / total_real_events) * 100 if total_real_events > 0 else 0
    
    fig.suptitle(
        f"Time window {t0:.3f}–{t1:.3f}s • cancel={cancelled:,} "
        f"({actual_cancellation_rate:.1f}%) • "
        f"spatial_tol={R_PIX}px, temporal_tol={BIN_MS}ms • "
        f"{'gray' if use_gray else 'seismic'} images",
        y=0.995, fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def create_roi_analysis_figure(combined_events, residual_real_events, residual_predicted_events, 
                              window, cx, cy, radius, img_w, img_h, show_overlay: bool = False):
    """Create ROI analysis figure with inside/outside cancellation rates.

    If show_overlay is False (default), the top combined seismic overlay panel is omitted
    and the figure contains only the 2x3 inside/outside triptychs.
    """
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.gridspec as gridspec
    
    t0, t1 = window
    
    # Extract events for the time window
    time_mask = (combined_events[:, 3] >= t0) & (combined_events[:, 3] < t1)
    window_events = combined_events[time_mask]
    real_events = window_events[window_events[:, 4] == 0.0]
    pred_events = window_events[window_events[:, 4] == 1.0]
    
    # Get residual events for this window
    resid_real_mask = (residual_real_events[:, 3] >= t0) & (residual_real_events[:, 3] < t1)
    resid_pred_mask = (residual_predicted_events[:, 3] >= t0) & (residual_predicted_events[:, 3] < t1)
    resid_real_window = residual_real_events[resid_real_mask]
    resid_pred_window = residual_predicted_events[resid_pred_mask]
    
    # Build masks
    sel_r_t = events_in_window(real_events, t0, t1)
    sel_p_t = events_in_window(pred_events, t0, t1)
    
    # Full dataset masks for inside/outside
    inside_r_full = circle_mask(real_events[:, 0], real_events[:, 1], cx, cy, radius, scale=1.05)
    inside_p_full = circle_mask(pred_events[:, 0], pred_events[:, 1], cx, cy, radius, scale=1.05)
    
    # Windowed masks
    inside_r = inside_r_full[sel_r_t]
    outside_r = ~inside_r
    
    # Residual masks - use same matching logic as extract_roi_data
    from scipy.spatial import cKDTree
    
    residual_full = np.zeros(len(real_events), dtype=bool)
    if len(resid_real_window) > 0 and len(real_events) > 0:
        resid_tree = cKDTree(resid_real_window[:, :2])
        spatial_tol = 1.0  # 1 pixel tolerance
        spatial_matches_list = resid_tree.query_ball_point(real_events[:, :2], spatial_tol)
        temporal_tol_s = 10e-3  # 10ms
        
        for i, (event, spatial_candidates) in enumerate(zip(real_events, spatial_matches_list)):
            if len(spatial_candidates) == 0:
                continue
            spatial_candidates = np.array(spatial_candidates)
            candidate_events = resid_real_window[spatial_candidates]
            time_diffs = np.abs(candidate_events[:, 3] - event[3])
            temporal_mask = time_diffs <= temporal_tol_s
            if np.any(temporal_mask):
                matching_candidates = candidate_events[temporal_mask]
                coord_match = np.any(
                    (np.abs(matching_candidates[:, 0] - event[0]) < 1.0) &
                    (np.abs(matching_candidates[:, 1] - event[1]) < 1.0) &
                    (np.abs(matching_candidates[:, 2] - event[2]) < 0.5)
                )
                if coord_match:
                    residual_full[i] = True
    
    residual_t = residual_full[sel_r_t]
    resid_in = residual_t & inside_r
    resid_out = residual_t & outside_r
    
    # Matched pairs
    matched_real_full = ~residual_full
    matched_r_t = matched_real_full[sel_r_t]
    match_in = matched_r_t & inside_r
    match_out = matched_r_t & outside_r
    
    # Compute pixel area masks
    yy, xx = np.mgrid[0:img_h, 0:img_w]
    pix_in_mask = circle_mask(xx, yy, cx, cy, radius, scale=1.05)
    in_area = int(pix_in_mask.sum())
    out_area = img_w * img_h - in_area
    
    # Compute cancellation rates
    count_real_inside = np.sum(inside_r)
    count_residual_inside = np.sum(resid_in)
    count_real_outside = np.sum(outside_r)
    count_residual_outside = np.sum(resid_out)
    
    cr_inside = (count_real_inside - count_residual_inside) / count_real_inside * 100 if count_real_inside > 0 else 0
    cr_outside = (count_real_outside - count_residual_outside) / count_real_outside * 100 if count_real_outside > 0 else 0
    
    # Events per pixel
    e_in = count_residual_inside / in_area if in_area > 0 else 0
    e_out = count_residual_outside / out_area if out_area > 0 else 0
    
    # Print ROI statistics
    print(f"[INSIDE ROI] real={count_real_inside}, residual={count_residual_inside}, cancellation_rate={cr_inside:.2f}%, events_per_pixel={e_in:.4f}")
    print(f"[OUTSIDE ROI] real={count_real_outside}, residual={count_residual_outside}, cancellation_rate={cr_outside:.2f}%, events_per_pixel={e_out:.4f}")
    
    # Extract windowed coordinates and polarities
    xr = real_events[sel_r_t, 0].astype(int)
    yr = real_events[sel_r_t, 1].astype(int)
    pr = real_events[sel_r_t, 2].astype(int)
    pr_signed = convert_polarity_to_signed(pr)
    
    # Build frames
    F_real_in = make_frame(img_h, img_w, xr[inside_r], yr[inside_r], pr_signed[inside_r])
    F_resid_in = make_frame(img_h, img_w, xr[resid_in], yr[resid_in], pr_signed[resid_in])
    F_pair_in = make_frame(img_h, img_w, xr[match_in], yr[match_in], None)
    
    F_real_out = make_frame(img_h, img_w, xr[outside_r], yr[outside_r], pr_signed[outside_r])
    F_resid_out = make_frame(img_h, img_w, xr[resid_out], yr[resid_out], pr_signed[resid_out])
    F_pair_out = make_frame(img_h, img_w, xr[match_out], yr[match_out], None)
    
    # Create the comprehensive figure
    if show_overlay:
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = gridspec.GridSpec(3, 3, figure=fig)
        row_offset = 1
        # Row 0: Seismic overlay
        ax_main = fig.add_subplot(gs[0, :])
        img_r, img_p, img_c, nr, npred = build_window_images(combined_events, window, img_w, img_h)
        img_r_n, img_p_n, img_c_n, max_abs = normalize_images_for_display(img_r, img_p, img_c)
        im2 = ax_main.imshow(img_c_n, cmap="seismic", origin="upper", vmin=0, vmax=1)
        ax_main.set_title("Combined Seismic Overlay")
        ax_main.set_xlabel("x")
        ax_main.set_ylabel("y")
        ax_main.grid(alpha=0.2)
        disc_circle = plt.Circle((cx, cy), radius, fill=True, color='yellow', alpha=0.15, linewidth=0)
        ax_main.add_patch(disc_circle)
        disc_outline = plt.Circle((cx, cy), radius, fill=False, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
        ax_main.add_patch(disc_outline)
        ax_main.plot(cx, cy, 'yo', markersize=6, markeredgecolor='black', markeredgewidth=1, label=f'Disc Center')
        ax_main.legend(loc='upper right', fontsize=7)
        cb = fig.colorbar(im2, ax=ax_main, fraction=0.046, pad=0.04)
        cb.set_label("signed count (Σ polarity)")
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.set_ticklabels([f"-{max_abs}", "0", f"+{max_abs}"])
    else:
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 3, figure=fig)
        row_offset = 0
    
    # Row 1: INSIDE triptych
    ax_i1 = fig.add_subplot(gs[0 + row_offset, 0])
    ax_i2 = fig.add_subplot(gs[0 + row_offset, 1])
    ax_i3 = fig.add_subplot(gs[0 + row_offset, 2])
    
    # Row 2: OUTSIDE triptych
    ax_o1 = fig.add_subplot(gs[1 + row_offset, 0])
    ax_o2 = fig.add_subplot(gs[1 + row_offset, 1])
    ax_o3 = fig.add_subplot(gs[1 + row_offset, 2])
    
    # Plot INSIDE frames
    vmax_in = max(1, int(np.percentile(np.abs(F_real_in), 99))) if np.any(F_real_in != 0) else 1
    norm_in = TwoSlopeNorm(vmin=-vmax_in, vcenter=0, vmax=vmax_in)
    
    F_real_in_safe = np.clip(np.nan_to_num(F_real_in, nan=0, posinf=0, neginf=0), -1e6, 1e6)
    F_resid_in_safe = np.clip(np.nan_to_num(F_resid_in, nan=0, posinf=0, neginf=0), -1e6, 1e6)
    F_pair_in_safe = np.clip(np.nan_to_num(F_pair_in, nan=0, posinf=0, neginf=0), 0, 1e6)
    
    im_i1 = ax_i1.imshow(F_real_in_safe, cmap='seismic', interpolation='nearest', origin='upper', 
                        extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), norm=norm_in)
    ax_i1.set_title(f"INSIDE: Real ({count_real_inside:,})")
    ax_i1.set_xlabel("x [px]")
    ax_i1.set_ylabel("y [px]")
    
    im_i2 = ax_i2.imshow(F_resid_in_safe, cmap='seismic', interpolation='nearest', origin='upper',
                        extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), norm=norm_in)
    ax_i2.set_title(f"INSIDE: Residual ({count_residual_inside:,})")
    ax_i2.set_xlabel("x [px]")
    ax_i2.set_ylabel("y [px]")
    
    im_i3 = ax_i3.imshow(F_pair_in_safe, cmap='Reds', interpolation='nearest', origin='upper',
                        extent=(-0.5, img_w-0.5, img_h-0.5, -0.5))
    ax_i3.set_title(f"INSIDE: Matched (real-side) ({np.sum(match_in):,})")
    ax_i3.set_xlabel("x [px]")
    ax_i3.set_ylabel("y [px]")
    
    # Plot OUTSIDE frames
    vmax_out = max(1, int(np.percentile(np.abs(F_real_out), 99))) if np.any(F_real_out != 0) else 1
    norm_out = TwoSlopeNorm(vmin=-vmax_out, vcenter=0, vmax=vmax_out)
    
    F_real_out_safe = np.clip(np.nan_to_num(F_real_out, nan=0, posinf=0, neginf=0), -1e6, 1e6)
    F_resid_out_safe = np.clip(np.nan_to_num(F_resid_out, nan=0, posinf=0, neginf=0), -1e6, 1e6)
    F_pair_out_safe = np.clip(np.nan_to_num(F_pair_out, nan=0, posinf=0, neginf=0), 0, 1e6)
    
    im_o1 = ax_o1.imshow(F_real_out_safe, cmap='seismic', interpolation='nearest', origin='upper',
                        extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), norm=norm_out)
    ax_o1.set_title(f"OUTSIDE: Real ({count_real_outside:,})")
    ax_o1.set_xlabel("x [px]")
    ax_o1.set_ylabel("y [px]")
    
    im_o2 = ax_o2.imshow(F_resid_out_safe, cmap='seismic', interpolation='nearest', origin='upper',
                        extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), norm=norm_out)
    ax_o2.set_title(f"OUTSIDE: Residual ({count_residual_outside:,})")
    ax_o2.set_xlabel("x [px]")
    ax_o2.set_ylabel("y [px]")
    
    im_o3 = ax_o3.imshow(F_pair_out_safe, cmap='Reds', interpolation='nearest', origin='upper',
                        extent=(-0.5, img_w-0.5, img_h-0.5, -0.5))
    ax_o3.set_title(f"OUTSIDE: Matched (real-side) ({np.sum(match_out):,})")
    ax_o3.set_xlabel("x [px]")
    ax_o3.set_ylabel("y [px]")
    
    # Add ROI circle outlines to all subplots
    for ax in [ax_i1, ax_i2, ax_i3, ax_o1, ax_o2, ax_o3]:
        circle = plt.Circle((cx, cy), radius * 1.05, fill=False, color='yellow', linewidth=1, linestyle='--', alpha=0.8)
        ax.add_patch(circle)
    
    fig.suptitle(f"ROI Analysis: Window {t0:.3f}–{t1:.3f}s | Inside: {cr_inside:.1f}% cancelled | Outside: {cr_outside:.1f}% cancelled", 
                 fontsize=14, y=0.98)
    
    # Disable cursor display to prevent overflow errors
    axes_to_freeze = [ax_i1, ax_i2, ax_i3, ax_o1, ax_o2, ax_o3]
    if show_overlay:
        axes_to_freeze.insert(0, ax_main)
    for ax in axes_to_freeze:
        ax.format_coord = lambda x, y: ""
        ax.set_navigate(False)
        for im in ax.get_images():
            im.set_interpolation('nearest')
    
    return fig

def extract_roi_data(combined_events, residual_real_events, residual_predicted_events, 
                     window, cx, cy, radius, img_w, img_h):
    """Extract and compute ROI data for a time window. Returns all frames and statistics."""
    t0, t1 = window
    
    # Extract events for the time window
    time_mask = (combined_events[:, 3] >= t0) & (combined_events[:, 3] < t1)
    window_events = combined_events[time_mask]
    real_events = window_events[window_events[:, 4] == 0.0]
    pred_events = window_events[window_events[:, 4] == 1.0]
    
    # Get residual events for this window
    resid_real_mask = (residual_real_events[:, 3] >= t0) & (residual_real_events[:, 3] < t1)
    resid_pred_mask = (residual_predicted_events[:, 3] >= t0) & (residual_predicted_events[:, 3] < t1)
    resid_real_window = residual_real_events[resid_real_mask]
    resid_pred_window = residual_predicted_events[resid_pred_mask]
    
    # Build masks
    sel_r_t = events_in_window(real_events, t0, t1)
    
    # Full dataset masks for inside/outside
    inside_r_full = circle_mask(real_events[:, 0], real_events[:, 1], cx, cy, radius, scale=1.05)
    
    # Windowed masks
    inside_r = inside_r_full[sel_r_t]
    outside_r = ~inside_r
    
    # Residual masks - match events using relaxed tolerance for visualization
    # Since residual_real_events contains the exact unmatched events, we just need to find them
    # Use lenient matching: within 1 pixel spatially and reasonable time tolerance
    from scipy.spatial import cKDTree
    
    residual_full = np.zeros(len(real_events), dtype=bool)
    if len(resid_real_window) > 0 and len(real_events) > 0:
        # Build spatial KDTree for residual events
        resid_tree = cKDTree(resid_real_window[:, :2])
        
        # Use lenient spatial tolerance (1 pixel) - events should be very close
        spatial_tol = 1.0  # 1 pixel tolerance
        spatial_matches_list = resid_tree.query_ball_point(real_events[:, :2], spatial_tol)
        
        # Use lenient temporal tolerance (10ms should be enough for matching)
        temporal_tol_s = 10e-3  # 10ms
        
        for i, (event, spatial_candidates) in enumerate(zip(real_events, spatial_matches_list)):
            if len(spatial_candidates) == 0:
                continue
            
            # Check temporal and coordinate match with lenient tolerance
            spatial_candidates = np.array(spatial_candidates)
            candidate_events = resid_real_window[spatial_candidates]
            
            # Check temporal tolerance
            time_diffs = np.abs(candidate_events[:, 3] - event[3])
            temporal_mask = time_diffs <= temporal_tol_s
            
            if np.any(temporal_mask):
                # Check for matches with lenient coordinate tolerance
                matching_candidates = candidate_events[temporal_mask]
                # Match if coordinates are within 1 pixel and polarity matches
                coord_match = np.any(
                    (np.abs(matching_candidates[:, 0] - event[0]) < 1.0) &
                    (np.abs(matching_candidates[:, 1] - event[1]) < 1.0) &
                    (np.abs(matching_candidates[:, 2] - event[2]) < 0.5)  # Polarity should match exactly (0 or 1)
                )
                if coord_match:
                    residual_full[i] = True
    
    residual_t = residual_full[sel_r_t]
    resid_in = residual_t & inside_r
    resid_out = residual_t & outside_r
    
    # Matched pairs
    matched_real_full = ~residual_full
    matched_r_t = matched_real_full[sel_r_t]
    match_in = matched_r_t & inside_r
    match_out = matched_r_t & outside_r
    
    # Compute cancellation rates
    count_real_inside = np.sum(inside_r)
    count_residual_inside = np.sum(resid_in)
    count_real_outside = np.sum(outside_r)
    count_residual_outside = np.sum(resid_out)
    
    cr_inside = (count_real_inside - count_residual_inside) / count_real_inside * 100 if count_real_inside > 0 else 0
    cr_outside = (count_real_outside - count_residual_outside) / count_real_outside * 100 if count_real_outside > 0 else 0
    
    # Extract windowed coordinates and polarities
    xr = real_events[sel_r_t, 0].astype(int)
    yr = real_events[sel_r_t, 1].astype(int)
    pr = real_events[sel_r_t, 2].astype(int)
    pr_signed = convert_polarity_to_signed(pr)
    
    # Build frames
    F_real_in = make_frame(img_h, img_w, xr[inside_r], yr[inside_r], pr_signed[inside_r])
    F_resid_in = make_frame(img_h, img_w, xr[resid_in], yr[resid_in], pr_signed[resid_in])
    F_pair_in = make_frame(img_h, img_w, xr[match_in], yr[match_in], None)
    
    F_real_out = make_frame(img_h, img_w, xr[outside_r], yr[outside_r], pr_signed[outside_r])
    F_resid_out = make_frame(img_h, img_w, xr[resid_out], yr[resid_out], pr_signed[resid_out])
    F_pair_out = make_frame(img_h, img_w, xr[match_out], yr[match_out], None)
    
    return {
        'F_real_in': F_real_in, 'F_resid_in': F_resid_in, 'F_pair_in': F_pair_in,
        'F_real_out': F_real_out, 'F_resid_out': F_resid_out, 'F_pair_out': F_pair_out,
        'count_real_inside': count_real_inside, 'count_residual_inside': count_residual_inside,
        'count_real_outside': count_real_outside, 'count_residual_outside': count_residual_outside,
        'cr_inside': cr_inside, 'cr_outside': cr_outside,
        'match_in': match_in, 'match_out': match_out,
        'cx': cx, 'cy': cy, 'radius': radius, 't0': t0, 't1': t1
    }

def create_individual_roi_plot(frame, title, cx, cy, radius, img_w, img_h, region_type, plot_type, 
                                 count=None, cr=None, cmap='seismic', vmax=None):
    """Create a single individual ROI plot (no subplots)"""
    from matplotlib.colors import TwoSlopeNorm
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Safe frame
    frame_safe = np.clip(np.nan_to_num(frame, nan=0, posinf=0, neginf=0), -1e6, 1e6)
    
    # Determine normalization
    if plot_type == 'matched':
        # Matched plots use Reds colormap
        norm = None
        vmin = 0
        if vmax is None:
            vmax = max(1, int(np.percentile(frame_safe, 99))) if np.any(frame_safe != 0) else 1
    else:
        # Real and Residual use seismic with TwoSlopeNorm
        if vmax is None:
            vmax = max(1, int(np.percentile(np.abs(frame_safe), 99))) if np.any(frame_safe != 0) else 1
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        vmin = -vmax
    
    # Plot
    if norm is not None:
        # When using a norm, don't pass vmin/vmax
        im = ax.imshow(frame_safe, cmap=cmap, interpolation='nearest', origin='upper',
                       extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), norm=norm)
    else:
        # When not using a norm, pass vmin/vmax directly
        im = ax.imshow(frame_safe, cmap=cmap, interpolation='nearest', origin='upper',
                       extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), vmin=vmin, vmax=vmax)
    
    # Add ROI circle outline
    circle = plt.Circle((cx, cy), radius * 1.05, fill=False, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
    ax.add_patch(circle)
    
    # Title with stats
    if count is not None:
        title_str = f"{title} ({count:,} events)"
        if cr is not None:
            title_str += f" | {cr:.1f}% cancelled"
    else:
        title_str = title
    
    ax.set_title(title_str, fontsize=12, fontweight='bold')
    ax.set_xlabel("x [px]", fontsize=11)
    ax.set_ylabel("y [px]", fontsize=11)
    ax.grid(alpha=0.3)
    
    # Colorbar
    if plot_type == 'matched':
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Matched event count", fontsize=10)
    else:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Signed count (Σ polarity)", fontsize=10)
        cb.set_ticks([vmin, 0, vmax])
        cb.set_ticklabels([f"-{vmax}", "0", f"+{vmax}"])
    
    ax.format_coord = lambda x, y: ""
    ax.set_navigate(False)
    
    fig.tight_layout()
    return fig

def generate_individual_roi_plots(combined_events, residual_real_events, residual_predicted_events,
                                  window, cx, cy, radius, img_w, img_h, output_dir):
    """Generate 6 individual ROI plots (one per figure) and save them"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract ROI data
    roi_data = extract_roi_data(combined_events, residual_real_events, residual_predicted_events,
                                window, cx, cy, radius, img_w, img_h)
    
    t0, t1 = roi_data['t0'], roi_data['t1']
    base_prefix = f"roi_{t0:.3f}s_to_{t1:.3f}s"
    
    # Determine vmax for consistent scaling
    vmax_in = max(1, int(np.percentile(np.abs(roi_data['F_real_in']), 99))) if np.any(roi_data['F_real_in'] != 0) else 1
    vmax_out = max(1, int(np.percentile(np.abs(roi_data['F_real_out']), 99))) if np.any(roi_data['F_real_out'] != 0) else 1
    
    # 1. INSIDE: Real
    fig = create_individual_roi_plot(
        roi_data['F_real_in'], "INSIDE ROI: Real Events",
        roi_data['cx'], roi_data['cy'], roi_data['radius'], img_w, img_h,
        'inside', 'real', roi_data['count_real_inside'], None, 'seismic', vmax_in
    )
    fig.savefig(os.path.join(output_dir, f"{base_prefix}_inside_real.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base_prefix}_inside_real.png")
    
    # 2. INSIDE: Residual
    fig = create_individual_roi_plot(
        roi_data['F_resid_in'], "INSIDE ROI: Residual Events",
        roi_data['cx'], roi_data['cy'], roi_data['radius'], img_w, img_h,
        'inside', 'residual', roi_data['count_residual_inside'], roi_data['cr_inside'], 'seismic', vmax_in
    )
    fig.savefig(os.path.join(output_dir, f"{base_prefix}_inside_residual.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base_prefix}_inside_residual.png")
    
    # 3. INSIDE: Matched
    fig = create_individual_roi_plot(
        roi_data['F_pair_in'], "INSIDE ROI: Matched Events",
        roi_data['cx'], roi_data['cy'], roi_data['radius'], img_w, img_h,
        'inside', 'matched', np.sum(roi_data['match_in']), None, 'Reds', None
    )
    fig.savefig(os.path.join(output_dir, f"{base_prefix}_inside_matched.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base_prefix}_inside_matched.png")
    
    # 4. OUTSIDE: Real
    fig = create_individual_roi_plot(
        roi_data['F_real_out'], "OUTSIDE ROI: Real Events",
        roi_data['cx'], roi_data['cy'], roi_data['radius'], img_w, img_h,
        'outside', 'real', roi_data['count_real_outside'], None, 'seismic', vmax_out
    )
    fig.savefig(os.path.join(output_dir, f"{base_prefix}_outside_real.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base_prefix}_outside_real.png")
    
    # 5. OUTSIDE: Residual
    fig = create_individual_roi_plot(
        roi_data['F_resid_out'], "OUTSIDE ROI: Residual Events",
        roi_data['cx'], roi_data['cy'], roi_data['radius'], img_w, img_h,
        'outside', 'residual', roi_data['count_residual_outside'], roi_data['cr_outside'], 'seismic', vmax_out
    )
    fig.savefig(os.path.join(output_dir, f"{base_prefix}_outside_residual.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base_prefix}_outside_residual.png")
    
    # 6. OUTSIDE: Matched
    fig = create_individual_roi_plot(
        roi_data['F_pair_out'], "OUTSIDE ROI: Matched Events",
        roi_data['cx'], roi_data['cy'], roi_data['radius'], img_w, img_h,
        'outside', 'matched', np.sum(roi_data['match_out']), None, 'Reds', None
    )
    fig.savefig(os.path.join(output_dir, f"{base_prefix}_outside_matched.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base_prefix}_outside_matched.png")
    
    return roi_data

def create_roi_three_panel_figure(roi_data, img_w, img_h, region='inside'):
    """Create a 3-panel figure showing Real, Residual, and Matched for one region"""
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])
    
    if region == 'inside':
        F_real = roi_data['F_real_in']
        F_resid = roi_data['F_resid_in']
        F_matched = roi_data['F_pair_in']
        count_real = roi_data['count_real_inside']
        count_resid = roi_data['count_residual_inside']
        count_matched = np.sum(roi_data['match_in'])
        cr = roi_data['cr_inside']
        title_prefix = "INSIDE ROI"
    else:
        F_real = roi_data['F_real_out']
        F_resid = roi_data['F_resid_out']
        F_matched = roi_data['F_pair_out']
        count_real = roi_data['count_real_outside']
        count_resid = roi_data['count_residual_outside']
        count_matched = np.sum(roi_data['match_out'])
        cr = roi_data['cr_outside']
        title_prefix = "OUTSIDE ROI"
    
    # Determine vmax for consistent scaling
    vmax = max(1, int(np.percentile(np.abs(F_real), 99))) if np.any(F_real != 0) else 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # Safe frames
    F_real_safe = np.clip(np.nan_to_num(F_real, nan=0, posinf=0, neginf=0), -1e6, 1e6)
    F_resid_safe = np.clip(np.nan_to_num(F_resid, nan=0, posinf=0, neginf=0), -1e6, 1e6)
    F_matched_safe = np.clip(np.nan_to_num(F_matched, nan=0, posinf=0, neginf=0), 0, 1e6)
    
    # Panel 1: Real
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(F_real_safe, cmap='seismic', interpolation='nearest', origin='upper',
                     extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), norm=norm)
    ax1.set_title(f"(a) {title_prefix}: Real\n({count_real:,} events)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("x [px]", fontsize=11)
    ax1.set_ylabel("y [px]", fontsize=11)
    ax1.grid(alpha=0.3)
    circle1 = plt.Circle((roi_data['cx'], roi_data['cy']), roi_data['radius'] * 1.05, 
                         fill=False, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
    ax1.add_patch(circle1)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("Signed count", fontsize=9)
    cb1.set_ticks([-vmax, 0, vmax])
    cb1.set_ticklabels([f"-{vmax}", "0", f"+{vmax}"])
    
    # Panel 2: Residual
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(F_resid_safe, cmap='seismic', interpolation='nearest', origin='upper',
                     extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), norm=norm)
    ax2.set_title(f"(b) {title_prefix}: Residual\n({count_resid:,} events, {cr:.1f}% cancelled)", 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("x [px]", fontsize=11)
    ax2.set_ylabel("y [px]", fontsize=11)
    ax2.grid(alpha=0.3)
    circle2 = plt.Circle((roi_data['cx'], roi_data['cy']), roi_data['radius'] * 1.05, 
                         fill=False, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
    ax2.add_patch(circle2)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("Signed count", fontsize=9)
    cb2.set_ticks([-vmax, 0, vmax])
    cb2.set_ticklabels([f"-{vmax}", "0", f"+{vmax}"])
    
    # Panel 3: Matched
    ax3 = fig.add_subplot(gs[0, 2])
    vmax_matched = max(1, int(np.percentile(F_matched_safe, 99))) if np.any(F_matched_safe != 0) else 1
    im3 = ax3.imshow(F_matched_safe, cmap='Reds', interpolation='nearest', origin='upper',
                     extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), vmin=0, vmax=vmax_matched)
    ax3.set_title(f"(c) {title_prefix}: Matched\n({count_matched:,} events)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("x [px]", fontsize=11)
    ax3.set_ylabel("y [px]", fontsize=11)
    ax3.grid(alpha=0.3)
    circle3 = plt.Circle((roi_data['cx'], roi_data['cy']), roi_data['radius'] * 1.05, 
                         fill=False, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
    ax3.add_patch(circle3)
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label("Matched count", fontsize=9)
    
    # Disable cursor display
    for ax in [ax1, ax2, ax3]:
        ax.format_coord = lambda x, y: ""
        ax.set_navigate(False)
    
    fig.suptitle(f"{title_prefix} Analysis: Window {roi_data['t0']:.3f}–{roi_data['t1']:.3f}s", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    return fig

def export_windows_panel_images(combined, resid_real, resid_pred, windows, output_dir, use_gray=False):
    """Export individual panel images for each time window"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, w in enumerate(windows):
        fig = make_panel_figure(combined, resid_real, resid_pred, w, use_gray=use_gray)
        suffix = "_gray" if use_gray else "_seismic"
        filename = f"ego_panel_{i+1}_{w[0]:.3f}s_to_{w[1]:.3f}s{suffix}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved panel image: {filename}")

def export_individual_window_assets(combined_events, residual_real_events, residual_predicted_events, 
                                    windows, output_root, image_width=IMG_W, image_height=IMG_H, use_gray=False):
    """Export per-window, per-plot individual assets into organized folders.

    For each window, saves:
      - scatter_real.png, scatter_pred.png, scatter_residual.png
      - img_real.png, img_pred.png, img_combined.png (normalized, seismic or gray)
      - hist_real.png, hist_pred.png, hist_combined.png
    """
    cmap = "gray" if use_gray else "seismic"
    os.makedirs(output_root, exist_ok=True)
    for i, window in enumerate(windows):
        t0, t1 = window
        win_dir = os.path.join(output_root, f"window_{i+1}_{t0:.3f}s_to_{t1:.3f}s")
        os.makedirs(win_dir, exist_ok=True)

        # Extract window events
        time_mask = (combined_events[:, 3] >= t0) & (combined_events[:, 3] < t1)
        w_all = combined_events[time_mask]
        w_real = w_all[w_all[:, 4] == 0.0]
        w_pred = w_all[w_all[:, 4] == 1.0]
        wr = residual_real_events[(residual_real_events[:, 3] >= t0) & (residual_real_events[:, 3] < t1)]
        wp = residual_predicted_events[(residual_predicted_events[:, 3] >= t0) & (residual_predicted_events[:, 3] < t1)]

        # 1) Scatter: real
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(w_real[:,0], w_real[:,1], s=2, alpha=0.7, c="tab:blue")
        ax.set_xlim(0, image_width); ax.set_ylim(0, image_height); ax.invert_yaxis(); ax.grid(True, alpha=0.3)
        ax.set_title(f"Real events ({len(w_real):,}) {t0:.3f}-{t1:.3f}s"); ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.savefig(os.path.join(win_dir, "scatter_real.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

        # 2) Scatter: predicted
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(w_pred[:,0], w_pred[:,1], s=2, alpha=0.7, c="tab:red")
        ax.set_xlim(0, image_width); ax.set_ylim(0, image_height); ax.invert_yaxis(); ax.grid(True, alpha=0.3)
        ax.set_title(f"Predicted events ({len(w_pred):,}) {t0:.3f}-{t1:.3f}s"); ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.savefig(os.path.join(win_dir, "scatter_pred.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

        # 3) Scatter: residuals (real+pred)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(wr[:,0], wr[:,1], s=2, alpha=0.7, c="tab:blue", label=f"Real resid ({len(wr):,})")
        ax.scatter(wp[:,0], wp[:,1], s=2, alpha=0.7, c="tab:red", label=f"Pred resid ({len(wp):,})")
        ax.set_xlim(0, image_width); ax.set_ylim(0, image_height); ax.invert_yaxis(); ax.grid(True, alpha=0.3)
        ax.set_title(f"Residual events {t0:.3f}-{t1:.3f}s"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
        fig.savefig(os.path.join(win_dir, "scatter_residual.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

        # 4) Per-pixel images
        img_r, img_p, img_c, nr, npred = build_window_images(combined_events, window, image_width, image_height)
        img_r_n, img_p_n, img_c_n, max_abs = normalize_images_for_display(img_r, img_p, img_c)

        def save_img(arr, fname, cmap):
            f = plt.figure(figsize=(8,6)); a = f.add_subplot(1,1,1)
            im = a.imshow(arr, cmap=cmap, origin="upper", vmin=0, vmax=1)
            a.set_title(fname.replace("_"," ").split(".")[0]); a.set_xlabel("x"); a.set_ylabel("y"); a.grid(alpha=0.2)
            cb = f.colorbar(im, ax=a, fraction=0.046, pad=0.04)
            cb.set_label("signed count (Σ polarity)"); cb.set_ticks([0.0,0.5,1.0]); cb.set_ticklabels([f"-{max_abs}","0",f"+{max_abs}"])
            f.savefig(os.path.join(win_dir, fname), dpi=150, bbox_inches="tight"); plt.close(f)

        save_img(img_r_n, f"img_real{'_gray' if use_gray else ''}.png", cmap)
        save_img(img_p_n, f"img_pred{'_gray' if use_gray else ''}.png", cmap)
        save_img(img_c_n, f"img_combined{'_gray' if use_gray else ''}.png", cmap)

        # 5) Histograms
        def save_histogram(vals, title, fname):
            f = plt.figure(figsize=(8,6)); a = f.add_subplot(1,1,1)
            nz = np.abs(vals.ravel()); nz = nz[nz > 0]
            a.hist(nz, bins=100, log=True, edgecolor="k", alpha=0.85)
            if nz.size:
                med = float(np.median(nz)); p95 = float(np.percentile(nz, 95))
                a.axvline(med, color="tab:orange", ls="--", lw=1.2, label=f"median={med:.1f}")
                a.axvline(p95, color="tab:green", ls="--", lw=1.0, label=f"95%={p95:.1f}")
            a.set_xlabel("|signed count|"); a.set_ylabel("pixels (log)"); a.grid(alpha=0.3); a.legend(); a.set_title(title)
            f.savefig(os.path.join(win_dir, fname), dpi=150, bbox_inches="tight"); plt.close(f)

        save_histogram(img_r, "Real |count|", "hist_real.png")
        save_histogram(img_p, "Pred |count|", "hist_pred.png")
        save_histogram(img_c, "Combined |count|", "hist_combined.png")

def main():
    """Main execution function"""
    print("Loading combined events data...")
    combined = load_combined(COMBINED_PATH)
    
    # For testing, work with a manageable subset
    if len(combined) > 2_000_000:
        print(f"Large dataset detected ({len(combined):,} events). Using subset for testing...")
        
        # Find subset that overlaps with analysis windows
        target_start = 5.000
        target_end = 5.010
        
        time_mask = (combined[:, 3] >= target_start) & (combined[:, 3] < target_end + 0.1)
        target_events = combined[time_mask]
        
        if len(target_events) > 0:
            subset_size = min(2_000_000, len(target_events))
            combined = target_events[:subset_size]
            print(f"Using subset from target window: {len(combined):,} events")
        else:
            subset_size = 2_000_000
            combined = combined[:subset_size]
            print(f"Using first {subset_size:,} events (no target window found)")
        
        print(f"Real events: {int(np.sum(combined[:,4]==0.0)):,}")
        print(f"Pred events: {int(np.sum(combined[:,4]==1.0)):,}")
        print(f"Time range: {combined[0,3]:.3f}s to {combined[-1,3]:.3f}s")
        
        # Adjust analysis windows to match subset
        subset_start = combined[0,3]
        subset_end = combined[-1,3]
        subset_duration = subset_end - subset_start
        
        global WINDOWS
        WINDOWS = [
            (subset_start, subset_start + subset_duration/3),
            (subset_start + subset_duration/3, subset_start + 2*subset_duration/3),
            (subset_start + 2*subset_duration/3, subset_end)
        ]
        print(f"Adjusted analysis windows to match subset:")
        for i, w in enumerate(WINDOWS):
            print(f"  Window {i+1}: {w[0]:.3f}s to {w[1]:.3f}s")

    print("Running ego-motion cancellation...")
    print("Using TRUE temporal gate method (corrected)")
    residual_real_events, residual_predicted_events = run_cancellation(combined, BIN_MS, R_PIX)

    print("Exporting analysis panels...")
    export_windows_panel_images(combined, residual_real_events, residual_predicted_events, WINDOWS,
                                OUTPUT_DIR, use_gray=False)
    export_windows_panel_images(combined, residual_real_events, residual_predicted_events, WINDOWS,
                                OUTPUT_DIR, use_gray=True)

    print("Exporting individual window assets (organized folders)...")
    indiv_root = os.path.join(OUTPUT_DIR, "individual")
    export_individual_window_assets(combined, residual_real_events, residual_predicted_events, WINDOWS,
                                    indiv_root, use_gray=False)
    export_individual_window_assets(combined, residual_real_events, residual_predicted_events, WINDOWS,
                                    indiv_root, use_gray=True)

    print("Creating ROI analysis visualization...")
    w0 = WINDOWS[0]
    
    # Create thesis output directory structure
    thesis_roi_dir = os.path.join("../../thesis/images/results_figures/roi_individual")
    os.makedirs(thesis_roi_dir, exist_ok=True)
    
    # Generate individual ROI plots (6 separate figures)
    print("\nGenerating individual ROI plots...")
    roi_data = generate_individual_roi_plots(
        combined, residual_real_events, residual_predicted_events,
        w0, DISC_CENTER_X, DISC_CENTER_Y, DISC_RADIUS, IMG_W, IMG_H,
        thesis_roi_dir
    )
    
    # Create 3-panel combined figures for inside and outside
    print("\nGenerating 3-panel combined figures...")
    
    # Inside ROI 3-panel
    fig_inside = create_roi_three_panel_figure(roi_data, IMG_W, IMG_H, region='inside')
    inside_filename = f"roi_{w0[0]:.3f}s_to_{w0[1]:.3f}s_inside_three_panel.png"
    fig_inside.savefig(os.path.join(thesis_roi_dir, inside_filename), dpi=300, bbox_inches="tight")
    plt.close(fig_inside)
    print(f"  Saved: {inside_filename}")
    
    # Outside ROI 3-panel
    fig_outside = create_roi_three_panel_figure(roi_data, IMG_W, IMG_H, region='outside')
    outside_filename = f"roi_{w0[0]:.3f}s_to_{w0[1]:.3f}s_outside_three_panel.png"
    fig_outside.savefig(os.path.join(thesis_roi_dir, outside_filename), dpi=300, bbox_inches="tight")
    plt.close(fig_outside)
    print(f"  Saved: {outside_filename}")
    
    # Also create the original comprehensive figure
    fig_roi = create_roi_analysis_figure(combined, residual_real_events, residual_predicted_events, 
                                       w0, DISC_CENTER_X, DISC_CENTER_Y, DISC_RADIUS, IMG_W, IMG_H,
                                       show_overlay=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    roi_filename = f"roi_analysis_{w0[0]:.3f}s_to_{w0[1]:.3f}s.png"
    fig_roi.savefig(os.path.join(OUTPUT_DIR, roi_filename), dpi=150, bbox_inches="tight")
    
    # Also save to thesis folder
    thesis_main_dir = os.path.join("../../thesis/images/results_figures")
    os.makedirs(thesis_main_dir, exist_ok=True)
    thesis_filename = os.path.join(thesis_main_dir, "roi_analysis_spatial.png")
    fig_roi.savefig(thesis_filename, dpi=300, bbox_inches="tight")
    print(f"\nSaved ROI analysis: {roi_filename}")
    print(f"Saved to thesis: {thesis_filename}")

    print(f"\nAnalysis complete!")
    print(f"Time windows analyzed: {len(WINDOWS)}")
    print(f"Cancellation parameters: {BIN_MS}ms temporal, {R_PIX}px spatial")
    print(f"Polarity mode: {POLARITY_MODE}")
    print(f"ROI analysis: Circle center ({DISC_CENTER_X:.1f}, {DISC_CENTER_Y:.1f}), radius {DISC_RADIUS:.0f}px")

    # Close all remaining figures to avoid blocking
    plt.close('all')
    
    print("\nAll plots have been saved to files.")
    print(f"Individual plots saved to: {thesis_roi_dir}")
    print(f"Main analysis saved to: {thesis_main_dir}")
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()
