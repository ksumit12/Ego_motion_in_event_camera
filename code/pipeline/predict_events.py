#!/usr/bin/env python3
"""
Event Camera Ego-Motion Prediction System

This script provides fully modular functions for predicting events based on ego-motion.
The core prediction logic can be imported and used in any other script.

Core Functions:
- predict_events_from_motion(): Main prediction function (fully modular)
- load_event_data_fast(): Load event data from CSV
- load_tracker_series(): Load motion tracking data
- apply_rotation(): Apply rotational motion model
- combine_and_sort(): Combine real and predicted events

Usage Example:
    from main_motion_test import predict_events_from_motion, load_event_data_fast, load_tracker_series
    
    # Load data
    real_events = load_event_data_fast("events.csv")
    t_s, cx_s, cy_s, om_s = load_tracker_series("tracker.csv")
    
    # Predict events
    predicted_events = predict_events_from_motion(real_events, t_s, cx_s, cy_s, om_s, 
                                                 dt_seconds=0.002, omega_bias=0.0)
"""

import time
import numpy as np
import pandas as pd
from typing import Tuple, Optional

# ---------------- Paths ----------------
REAL_EVENTS_FILE      = "/home/sumit/anu_research/recording/new_data/perlin_1280hz_hand_outframe.csv"
TRACKER_CSV_FILE      = "/home/sumit/anu_research/ego_motion/results_csv/perlin_1280hz_hand_outframe_combined.csv"   # needs: timestamp, center_x, center_y, omega_circlefit_rad_s (or theta_dot_rad_s)
PREDICTED_EVENTS_FILE = "./predicted_events.npy"
COMBINED_EVENTS_FILE  = "./combined_events_with_predictions.npy"

# ---------------- Params ----------------
DT_SECONDS = 0.002        # Δt
# DT_SECONDS = 0
CHUNK_SIZE = 1_000_000
OMEGA_SOURCE = "circlefit"  # "circlefit" | "theta_dot"
OMEGA_BIAS   = 0.0          # optional constant bias (rad/s)

# ---------------- Core Motion Model Functions ----------------

def apply_rotation(x: np.ndarray, y: np.ndarray, cx: float, cy: float, 
                   omega: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rotational motion model to event coordinates.
    
    Args:
        x: X coordinates of events
        y: Y coordinates of events  
        cx: Rotation center X coordinate
        cy: Rotation center Y coordinate
        omega: Angular velocity (rad/s) for each event
        dt: Time step (seconds)
        
    Returns:
        Tuple of (new_x, new_y) coordinates after rotation
        
    Formula: Rotate (x,y) about (cx,cy) by θ = ω·dt
    """
    theta = omega * dt
    c = np.cos(theta)
    s = np.sin(theta)
    dx = x - cx
    dy = y - cy
    x_new = cx + c*dx - s*dy
    y_new = cy + s*dx + c*dy
    return x_new.astype(np.float32), y_new.astype(np.float32)

# ---------------- Data Loading Functions ----------------

def load_event_data_fast(path: str) -> np.ndarray:
    """
    Load event data from CSV file with fast pandas processing.
    
    Args:
        path: Path to CSV file
        
    Returns:
        Array with columns [x, y, polarity, timestamp] sorted by time
        - x, y: Event coordinates (float32)
        - polarity: Event polarity {0, 1} (float32) 
        - timestamp: Time in seconds (float64)
        
    Expected CSV format: x, y, polarity, timestamp(μs)
    Automatically converts polarity from {-1,+1} to {0,1} if needed
    """
    df = pd.read_csv(path, names=["x","y","p","t"],
                     dtype={"x":np.float32,"y":np.float32,"p":np.float32,"t":np.float64})
    df["t"] = df["t"] * 1e-6  # μs -> s
    
    # Keep timestamps as float64, only spatial coordinates as float32
    ev = df[["x","y","p","t"]].to_numpy()
    ev[:, :3] = ev[:, :3].astype(np.float32)  # x, y, p as float32
    # t remains float64

    if not np.all(ev[:-1,3] <= ev[1:,3]):
        ev = ev[np.argsort(ev[:,3])]

    # Map {-1,+1} to {0,1} if present
    uniq = np.unique(ev[:,2])
    if uniq.shape[0] == 2 and uniq.min() == -1.0 and uniq.max() == 1.0:
        ev[:,2] = (ev[:,2] + 1.0) * 0.5

    return ev

def load_tracker_series(path: str, source: str = "circlefit") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load motion tracking data from CSV file.
    
    Args:
        path: Path to tracker CSV file
        source: Omega source - "circlefit" or "theta_dot"
        
    Returns:
        Tuple of (timestamps, center_x, center_y, omega) arrays (all float64)
        
    Required CSV columns: timestamp, center_x, center_y
    Omega column: omega_circlefit_rad_s, theta_dot_rad_s, or omega_rad_s
    """
    df = pd.read_csv(path)
    need = {"timestamp","center_x","center_y"}
    if not need.issubset(df.columns):
        raise ValueError("tracker CSV must have columns: timestamp, center_x, center_y")

    df = df.sort_values("timestamp")
    t_s  = df["timestamp"].to_numpy(np.float64)
    cx_s = df["center_x"].to_numpy(np.float64)
    cy_s = df["center_y"].to_numpy(np.float64)

    if source == "theta_dot":
        col = "theta_dot_rad_s" if "theta_dot_rad_s" in df.columns else "omega_rad_s"
    else:
        col = "omega_circlefit_rad_s" if "omega_circlefit_rad_s" in df.columns else "omega_rad_s"
    if col not in df.columns:
        raise ValueError("No suitable omega column in tracker CSV.")
    om_s = df[col].to_numpy(np.float64)

    return t_s, cx_s, cy_s, om_s

# ---------------- Utility Functions ----------------

def interp1(tq: np.ndarray, tx: np.ndarray, vx: np.ndarray) -> np.ndarray:
    """
    1D linear interpolation with edge hold (constant extrapolation).
    
    Args:
        tq: Query times
        tx: Reference times  
        vx: Reference values
        
    Returns:
        Interpolated values at query times
    """
    return np.interp(tq, tx, vx, left=vx[0], right=vx[-1])

# ---------------- Core Prediction Function ----------------

def predict_events_from_motion(real_events: np.ndarray, 
                              timestamps: np.ndarray, 
                              center_x: np.ndarray, 
                              center_y: np.ndarray, 
                              omega: np.ndarray,
                              dt_seconds: float = 0.002,
                              omega_bias: float = 0.0,
                              verbose: bool = True) -> np.ndarray:
    """
    Predict events based on ego-motion using TRUE event-by-event processing.
    
    This processes each event individually, sampling motion parameters at each
    event's exact timestamp - no batching or chunking.
    
    Args:
        real_events: Array [x, y, polarity, timestamp] of real events
        timestamps: Motion tracking timestamps (float64)
        center_x: Rotation center X coordinates over time (float64)
        center_y: Rotation center Y coordinates over time (float64) 
        omega: Angular velocity over time in rad/s (float64)
        dt_seconds: Time step for prediction (seconds)
        omega_bias: Constant bias to add to omega (rad/s)
        verbose: Whether to print progress updates
        
    Returns:
        Array [x, y, polarity, timestamp] of predicted events
        
    Algorithm (Event-by-Event):
        For each real event at time t:
        1. Sample motion parameters (cx, cy, omega) at exactly time t
        2. Apply rotation: rotate (x,y) about (cx,cy) by θ = omega·dt
        3. Set predicted time to t + dt
        4. Flip polarity for cancellation
    """
    if len(real_events) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    
    N = len(real_events)
    predicted_events = []
    
    for i, event in enumerate(real_events):
        # Extract event properties
        x, y, p, t = event[0], event[1], event[2], event[3]
        
        # Sample motion parameters at this event's exact timestamp
        cx = interp1(t, timestamps, center_x)
        cy = interp1(t, timestamps, center_y)
        om = interp1(t, timestamps, omega)
        
        # Apply omega bias if specified
        if omega_bias != 0.0:
            om = om + omega_bias

        # Apply rotational motion model to this single event
        px, py = apply_rotation(np.array([x]), np.array([y]), cx, cy, np.array([om]), dt_seconds)
        
        # Set predicted event properties
        pt = t + dt_seconds
        pp = 1.0 - p  # Flip polarity for cancellation
        
        # Store predicted event
        predicted_events.append([px[0], py[0], pp, pt])
        
        # Progress update
        if verbose and (i + 1) % 10000 == 0:
            print(f"Processed {i+1:,}/{N:,} events ({(i+1)/N*100:.1f}%)")

    return np.array(predicted_events, dtype=np.float32)

# ---------------- Data Combination Functions ----------------

def combine_and_sort(real_events: np.ndarray, predicted_events: np.ndarray) -> np.ndarray:
    """
    Combine real and predicted events with flags and sort by timestamp.
    
    Args:
        real_events: Array [x, y, polarity, timestamp] of real events
        predicted_events: Array [x, y, polarity, timestamp] of predicted events
        
    Returns:
        Combined array [x, y, polarity, timestamp, flag] sorted by timestamp
        - flag: 0.0 for real events, 1.0 for predicted events
    """
    real_flag = np.zeros((len(real_events),1), dtype=np.float32)
    pred_flag = np.ones((len(predicted_events),1),  dtype=np.float32)
    
    # Ensure consistent data types for combining
    real_combined = np.column_stack([real_events[:,:3].astype(np.float32), real_events[:,3], real_flag])
    pred_combined = np.column_stack([predicted_events[:,:3].astype(np.float32), predicted_events[:,3], pred_flag])
    
    combined = np.vstack([real_combined, pred_combined])
    return combined[np.argsort(combined[:,3])]

# ---------------- Main Execution ----------------

def main():
    """Main execution function for event prediction."""
    t0 = time.time()
    print("=== Event Camera Ego-Motion Prediction ===")
    print(f"DT={DT_SECONDS*1e3:.1f} ms | omega_source={OMEGA_SOURCE} | bias={OMEGA_BIAS:.3f}")

    # Load data
    real_events = load_event_data_fast(REAL_EVENTS_FILE)
    timestamps, center_x, center_y, omega = load_tracker_series(TRACKER_CSV_FILE, source=OMEGA_SOURCE)

    # Predict events
    predicted_events = predict_events_from_motion(
        real_events=real_events,
        timestamps=timestamps,
        center_x=center_x,
        center_y=center_y,
        omega=omega,
        dt_seconds=DT_SECONDS,
        omega_bias=OMEGA_BIAS,
        verbose=True
    )
    
    # Save results
    np.save(PREDICTED_EVENTS_FILE, predicted_events)
    combined_events = combine_and_sort(real_events, predicted_events)
    np.save(COMBINED_EVENTS_FILE, combined_events)
    
    print(f"Saved predicted events: {PREDICTED_EVENTS_FILE} ({len(predicted_events):,} events)")
    print(f"Saved combined events: {COMBINED_EVENTS_FILE} ({len(combined_events):,} events)")
    print(f"Time range: {combined_events[0,3]:.3f}s to {combined_events[-1,3]:.3f}s")
    print(f"Completed in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
