#!/usr/bin/env python3
"""
Causal, physically-faithful cancellation pipeline accelerated with PyTorch GPU (with fallback).
- Inputs: real_events.npy, predicted_events.npy (or generate from dt/txt/csv if needed)
- Logic: strictly per-event, no time or dt chunking, rolling buffer as needed
- GPU accelerates spatial search for each candidate prediction (for large sets)
"""
import argparse
import numpy as np
import sys
import os
import time
from pathlib import Path
import torch

# =====================
# Utility functions
# =====================
def parse_args():
    p = argparse.ArgumentParser(description="Causal cancellation via GPU acceleration (if available)")
    p.add_argument("--real", required=True, help="Path to real_events.npy")
    p.add_argument("--pred", required=True, help="Path to predicted_events.npy")
    p.add_argument("--eps_t", type=float, default=1.0, help="Temporal gate [ms]")
    p.add_argument("--eps_xy", type=float, default=2.0, help="Spatial gate [px]")
    p.add_argument("--output", default="cancellation_gpu_results.npz", help="Output metrics file (npz)")
    p.add_argument("--gpu", action="store_true", help="Force use GPU, error if unavailable")
    return p.parse_args()

def find_event_indices_in_time_range(times, target_times, eps_t):
    """
    For each predicted event time, return indices of real events within [t' - eps_t, t' + eps_t], vectorized.
    Args:
      times: sorted array of real event times (in seconds)
      target_times: array of predicted event target times t' (in seconds)
      eps_t: tolerance (in seconds)
    Returns:
      index_list: for each predicted event i, (start_index, end_index) in real events
    """
    index_list = []
    for t in target_times:
        left = np.searchsorted(times, t - eps_t, side='left')
        right = np.searchsorted(times, t + eps_t, side='right')
        index_list.append((left, right))
    return index_list


def spatial_and_polarity_gate_torch(pred_xy, pred_pol, real_xy, real_pol, eps_xy):
    """
    Args:
      pred_xy: (M,2) float32, predicted event spatial positions
      pred_pol: (M,) float32, predicted polarity
      real_xy: (N,2) float32, candidate real event positions
      real_pol: (N,) float32, candidate real polarity
      eps_xy: radius, float
    Returns:
      For each pred: index in real within eps_xy and polarity - inverse lookup; None if no match.
    """
    # Distance matrix (M, N)
    dists = torch.cdist(pred_xy, real_xy)
    mask_dist = dists <= eps_xy
    mask_pol = pred_pol.unsqueeze(1) != real_pol.unsqueeze(0)
    mask = mask_dist & mask_pol
    matches = []
    for i in range(pred_xy.shape[0]):
        inds = torch.nonzero(mask[i], as_tuple=False).flatten()
        if inds.numel() == 0:
            matches.append(None)
            continue
        # Take closest spatially (mutual NN: check is symmetric in post step)
        minj = inds[torch.argmin(dists[i, inds])]
        matches.append(int(minj))
    return matches


def main():
    args = parse_args()
    t0 = time.time()
    real_events = np.load(args.real, mmap_mode='r')
    pred_events = np.load(args.pred, mmap_mode='r')
    eps_t_s = args.eps_t * 1e-3
    eps_xy = args.eps_xy
    n_real = len(real_events)
    n_pred = len(pred_events)
    print(f"Loaded {n_real} real, {n_pred} predicted events")
    times_real = real_events[:, 3]
    times_pred = pred_events[:, 3]

    # Mask to keep track of cancelled matches
    real_matched = np.zeros(n_real, dtype=bool)
    pred_matched = np.zeros(n_pred, dtype=bool)

    batch_size = 4096 if n_pred > 100000 else 512  # tune this for available VRAM

    for batch_start in range(0, n_pred, batch_size):
        batch_end = min(batch_start+batch_size, n_pred)
        pred_batch = pred_events[batch_start:batch_end]
        tp = times_pred[batch_start:batch_end]

        # For each predicted event, find candidate real events
        idx_ranges = find_event_indices_in_time_range(times_real, tp, eps_t_s)
        pred_xy = torch.tensor(pred_batch[:, :2], dtype=torch.float32)
        pred_pol = torch.tensor(pred_batch[:,2], dtype=torch.float32)

        # Gather all real candidates for this batch
        real_idx_sets = [np.arange(start, stop) for start, stop in idx_ranges]
        # Big unique set for vectorization
        all_real_inds = np.unique(np.concatenate(real_idx_sets) if real_idx_sets else np.empty(0, dtype=int))
        if all_real_inds.size == 0:
            continue
        real_xy_full = torch.tensor(real_events[all_real_inds, :2], dtype=torch.float32)
        real_pol_full = torch.tensor(real_events[all_real_inds,2], dtype=torch.float32)
        # For GPU or CPU
        if args.gpu or torch.cuda.is_available():
            pred_xy = pred_xy.cuda()
            pred_pol = pred_pol.cuda()
            real_xy_full = real_xy_full.cuda()
            real_pol_full = real_pol_full.cuda()
        # Vectorized spatial/polarity filter for the batch
        # For each pred event, restrict to relevant real indices only
        # (small buffer, not global matches)
        # For each pred, build selection mask among real_xy_full
        for i, real_inds in enumerate(real_idx_sets):
            if len(real_inds) == 0:
                continue
            inds_in_full = torch.tensor(np.searchsorted(all_real_inds, real_inds), dtype=torch.long)
            real_subxy = real_xy_full[inds_in_full]
            real_subpol = real_pol_full[inds_in_full]
            pxy = pred_xy[i].unsqueeze(0)
            ppol = pred_pol[i].unsqueeze(0)
            dists = torch.cdist(pxy, real_subxy)[0]
            msk_dist = dists <= eps_xy
            msk_pol = ppol != real_subpol
            joint = msk_dist & msk_pol
            if not joint.any():
                continue
            # Find closest real
            minj = torch.argmin(dists + (~joint) * 1e6)
            # One-to-one: check neither already matched
            global_pred_idx = batch_start + i
            global_real_idx = real_inds[minj.item()]
            if not pred_matched[global_pred_idx] and not real_matched[global_real_idx]:
                pred_matched[global_pred_idx] = True
                real_matched[global_real_idx] = True
        if args.gpu or torch.cuda.is_available():
            torch.cuda.empty_cache()

    cratio = np.sum(pred_matched) / n_pred * 100
    rratio = np.sum(real_matched) / n_real * 100
    print(f"Done in {time.time()-t0:.1f}s: Cancellation ratio (pred): {cratio:.2f}% | (real): {rratio:.2f}%")
    np.savez(args.output, cratio=cratio, rratio=rratio, pred_matched=pred_matched, real_matched=real_matched)
    
if __name__ == "__main__":
    main()
