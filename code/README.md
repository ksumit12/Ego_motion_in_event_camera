# Code structure

This folder contains the core 3-stage pipeline and supporting analysis scripts.

## Pipeline
1. extract_angular_velocity.py
   - Input: AEB tracker outputs (multiple tracked blobs)
   - Action: Extract (x, y) trajectories per blob, estimate angular velocity (omega) per track, select/aggregate to a single best omega, and write a combined CSV.
   - Output: combined_omega.csv

2. predict_events.py
   - Input: combined_omega.csv + raw event stream
   - Action: Predict per-event motion/forward projection and save predicted events to a NumPy array on disk.
   - Output: predicted_events.npy

3. apply_cancellation_visualize.py
   - Input: predicted_events.npy (and original events)
   - Action: Apply cancellation logic in time windows, produce visualizations/metrics.
   - Output: figures/, metrics CSVs

## Suggested supporting layout
- data/              # raw events, tracker outputs
- outputs/
  - omega/
  - predictions/
  - cancellation/
- analysis/          # plotting/metrics helpers

## Running
- Stage 1:
  python pipeline/extract_angular_velocity.py --input data/tracker/ --out outputs/omega/combined_omega.csv

- Stage 2:
  python pipeline/predict_events.py --omega outputs/omega/combined_omega.csv --events data/events/ --out outputs/predictions/predicted_events.npy

- Stage 3:
  python pipeline/apply_cancellation_visualize.py --pred outputs/predictions/predicted_events.npy --events data/events/ --out outputs/cancellation/

Adjust flags to match your script arguments.
