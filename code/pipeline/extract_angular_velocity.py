#!/usr/bin/env python3
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
TRACKER_FOLDER = "/home/sumit/anu_research/ego_motion/AEB_tracker"
OUTPUT_CSV_FOLDER = "results_csv"
OUTPUT_PLOT_FOLDER = "results_plots"
DATASET_PREFIX = "perlin_1280hz_hand_outframe_"

WINDOW_S = 0.50
STEP_S = 0.05
MIN_POINTS = 300
R_MIN, R_MAX = 50.0, 500.0
MIN_ARC_DEG = 90.0
INLIER_TOL = 5.0
MAX_RMS = 3.0

def fix_time(t):
    """Convert timestamps to seconds if needed"""
    t = np.asarray(t, float)
    if not len(t): return t
    m = float(t.max())
    if m > 1e12: t *= 1e-9    # ns -> s
    elif m > 1e6: t *= 1e-6   # us -> s
    elif m > 6e4 and np.median(np.diff(np.sort(t))) > 1.0: t *= 1e-3  # ms -> s
    return t

def read_positions(path):
    """Read tracker CSV file and return sorted t, x, y arrays"""
    rows = []
    for line in open(path):
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            ts = float(parts[0])
            xy = parts[1].split()
            if len(xy) < 2:
                continue
            x, y = float(xy[0]), float(xy[1])
            rows.append((ts, x, y))
        except Exception:
            continue
    
    if not rows: return np.array([]), np.array([]), np.array([])
    a = np.asarray(rows, float)
    t, x, y = fix_time(a[:,0]), a[:,1], a[:,2]
    o = np.argsort(t)
    return t[o], x[o], y[o]

def least_square_fit(x, y):
    """Fit circle using least squares method"""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x*x + y*y)
    a, bcoef, c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = -a/2.0, -bcoef/2.0
    r = np.sqrt(max(cx*cx + cy*cy - c, 0.0))
    return float(cx), float(cy), float(r)

def fit_circle_robust(x, y):
    """Fit circle with outlier removal"""
    if len(x) < 3: return None
    cx, cy, r = least_square_fit(x, y)
    inliers = np.abs(np.hypot(x - cx, y - cy) - r) <= INLIER_TOL
    if inliers.sum() < 3: return None
    
    cx2, cy2, r2 = least_square_fit(x[inliers], y[inliers])
    rms = float(np.sqrt(np.mean((np.hypot(x[inliers]-cx2, y[inliers]-cy2) - r2)**2)))
    return dict(cx=cx2, cy=cy2, r=r2, rms=rms)

def calc_omega(t, x, y, cx, cy):
    """Calculate angular velocity from trajectory"""
    theta = np.unwrap(np.arctan2(y - cy, x - cx))
    arc_deg = float(np.degrees(theta.max() - theta.min()))
    if len(t) < 5: return None, arc_deg
    omega = np.polyfit(t - t[0], theta, 1)[0]
    return float(omega), arc_deg

def extract_windows(t, x, y):
    """Extract sliding windows and fit circles"""
    results = []
    left = t[0]
    
    while left + WINDOW_S <= t[-1]:
        right = left + WINDOW_S
        i0 = np.searchsorted(t, left, side="left")
        i1 = np.searchsorted(t, right, side="right")
        left += STEP_S
        
        if i1 - i0 < MIN_POINTS: continue
        tw, xw, yw = t[i0:i1], x[i0:i1], y[i0:i1]
        
        fit = fit_circle_robust(xw, yw)
        if fit is None: continue
        
        cx, cy, r, rms = fit["cx"], fit["cy"], fit["r"], fit["rms"]
        if not (R_MIN <= r <= R_MAX) or rms > MAX_RMS: continue
        
        omega, arc = calc_omega(tw, xw, yw, cx, cy)
        if omega is None or not np.isfinite(omega) or arc < MIN_ARC_DEG: continue
        
        results.append({
            'timestamp': 0.5*(tw[0] + tw[-1]),
            'window_left': float(tw[0]),
            'window_right': float(tw[-1]),
            'window_len_s': float(tw[-1] - tw[0]),
            'center_x': float(cx),
            'center_y': float(cy),
            'radius': float(r),
            'omega_circlefit_rad_s': float(omega),
            'inlier_rms': float(rms),
            'num_points': int(len(tw)),
            'arc_deg': float(arc)
        })
    
    return pd.DataFrame(results)

def plot_individual(t, x, y, df, title, out_png):
    """Create individual file plot with trajectory and circle"""
    if df.empty: return
    
    cxm, cym = df["center_x"].median(), df["center_y"].median()
    rm = df["radius"].median()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trajectory and circle
    ax1.plot(x, y, lw=0.6, alpha=0.4, label="trajectory")
    ax1.plot(cxm, cym, "rx", ms=10, mew=2, label=f"center ({cxm:.1f},{cym:.1f})")
    
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x, circle_y = cxm + rm * np.cos(theta), cym + rm * np.sin(theta)
    ax1.plot(circle_x, circle_y, "r--", lw=2, alpha=0.7, label=f"circle (r={rm:.1f})")
    
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title("trajectory + center + circle")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    
    # Omega over time
    ax2.plot(df["timestamp"], df["omega_circlefit_rad_s"], ".", ms=2)
    ax2.axhline(df["omega_circlefit_rad_s"].median(), ls="--", c="r")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("omega(t)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("rad/s")
    
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)

def plot_combined(all_data, dataset_name, out_png):
    """Create combined plot with all circles"""
    if not all_data: return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (file_data, file_name) in enumerate(all_data):
        if file_data['df'].empty: continue
        
        color = colors[i % len(colors)]
        df = file_data['df']
        cxm, cym, rm = df["center_x"].median(), df["center_y"].median(), df["radius"].median()
        
        # Center point
        ax.plot(cxm, cym, 'o', color=color, ms=10, mew=2, markerfacecolor='white',
                label=f"{file_name} center ({cxm:.1f},{cym:.1f})")
        
        # Circle outline
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x, circle_y = cxm + rm * np.cos(theta), cym + rm * np.sin(theta)
        ax.plot(circle_x, circle_y, "-", color=color, lw=3, alpha=0.9,
                label=f"{file_name} circle (r={rm:.1f})")
    
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f"Combined view: {dataset_name} - Circle centers and outlines")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Combined plot saved: {out_png}")

def main():
    """Main processing function"""
    os.makedirs(OUTPUT_CSV_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_PLOT_FOLDER, exist_ok=True)

    files = sorted(glob.glob(os.path.join(TRACKER_FOLDER, f"{DATASET_PREFIX}*.csv")))
    if not files:
        print("No files found")
        return

    all_results, all_centers, all_file_data = [], [], []
    
    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"-> {base_name}")
        
        t, x, y = read_positions(file_path)
        if len(t) < MIN_POINTS:
            print("   too few points")
            continue

        df = extract_windows(t, x, y)
        if df.empty:
            print("   no valid windows")
            continue

        df["source_file"] = base_name
        all_centers.append((df["center_x"].median(), df["center_y"].median()))
        all_file_data.append(({'df': df, 'x': x, 'y': y}, base_name))
        
        # Save individual results
        csv_path = os.path.join(OUTPUT_CSV_FOLDER, f"{base_name}_windows.csv")
        plot_path = os.path.join(OUTPUT_PLOT_FOLDER, f"{base_name}.png")
        df.to_csv(csv_path, index=False)
        plot_individual(t, x, y, df, base_name, plot_path)
        print(f"   windows={len(df)}  saved: {csv_path}")
        
        all_results.append(df)
    
    if not all_results:
        print("Nothing to combine")
        return
    
    # Combined analysis
    combined_plot_path = os.path.join(OUTPUT_PLOT_FOLDER, f"{DATASET_PREFIX}combined_circles.png")
    plot_combined(all_file_data, DATASET_PREFIX.rstrip("_"), combined_plot_path)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_csv = os.path.join(OUTPUT_CSV_FOLDER, f"{DATASET_PREFIX}combined_windows.csv")
    combined_df.to_csv(combined_csv, index=False)
    
    dataset_center = (np.median([c[0] for c in all_centers]), np.median([c[1] for c in all_centers]))
    print(f"\nCombined CSV: {combined_csv}  rows={len(combined_df)}")
    print(f"Dataset median center: ({dataset_center[0]:.2f}, {dataset_center[1]:.2f})")

if __name__ == "__main__":
    main()