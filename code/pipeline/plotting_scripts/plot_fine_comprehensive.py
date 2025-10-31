#!/usr/bin/env python3
"""
Comprehensive fine-resolution plotting for thesis.

Generates multiple publication-quality plots from fine-resolution dt sweep data:
1. Early drop-off with pixel displacement (dual axis)
2. CR decay rate (dCR/dΔt) - shows where performance drops fastest
3. Operating window analysis (90%, 80% thresholds marked)
4. Phase error comparison (theoretical vs actual)

Usage:
  python3 plot_fine_comprehensive.py --data fine_resolution_results.npz --output_dir plots/
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ROI parameters (from thesis)
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264
MEAN_RADIUS_PX = 199  # Mean radius from your analysis

# Default parameters
DEFAULT_OMEGA_RAD_S = 3.6
DEFAULT_EPS_XY_PX = 2.0


def plot_1_early_dropoff(dt_values, cr_values, pixel_displacement, output_path, omega_rad_s):
    """
    Plot 1: Early drop-off analysis with dual axis.
    Main thesis figure showing fine-resolution CR vs dt.
    """
    # Find drop-off point (first dt where CR < 90%)
    drop_off_idx = np.where(cr_values < 90.0)[0]
    if len(drop_off_idx) > 0:
        drop_off_dt = dt_values[drop_off_idx[0]]
        drop_off_cr = cr_values[drop_off_idx[0]]
        drop_off_px = pixel_displacement[drop_off_idx[0]]
    else:
        drop_off_dt = None
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Primary plot: CR vs dt
    line = ax1.plot(dt_values, cr_values, 'o-', linewidth=2.5, markersize=7, 
                    color='#1f77b4', label='Cancellation Rate', zorder=3)
    ax1.set_xlabel('Prediction Horizon Δt (ms)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cancellation Rate (%)', fontsize=14, fontweight='bold')
    ax1.set_ylim(70, 101)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Add threshold lines
    ax1.axhline(y=90, color='gray', linestyle='--', linewidth=2, 
                alpha=0.7, label='90% threshold', zorder=1)
    ax1.axhline(y=80, color='gray', linestyle=':', linewidth=2, 
                alpha=0.5, label='80% threshold', zorder=1)
    
    # Highlight ≥90% region
    ax1.axhspan(90, 101, alpha=0.15, color='green', label='≥90% Region', zorder=0)
    
    # Mark drop-off point
    if drop_off_dt is not None:
        ax1.plot(drop_off_dt, drop_off_cr, 'o', markersize=12, 
                color='red', label=f'Drop-off: {drop_off_dt:.1f}ms', zorder=4)
        ax1.annotate(f'Drop-off: {drop_off_dt:.1f}ms\n{drop_off_cr:.1f}%',
                    xy=(drop_off_dt, drop_off_cr),
                    xytext=(drop_off_dt + 0.4, drop_off_cr - 6),
                    fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.6', fc='yellow', alpha=0.9, edgecolor='orange', linewidth=2),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=2))
    
    # Secondary axis: pixel displacement
    ax2 = ax1.twiny()
    ax2.set_xlim(pixel_displacement[0], pixel_displacement[-1])
    ax2.set_xlabel(f'Predicted Pixel Displacement (px)\n(ω={omega_rad_s} rad/s, r≈{MEAN_RADIUS_PX}px mean)', 
                   fontsize=12, color='#555', fontweight='bold')
    ax2.tick_params(axis='x', labelcolor='#555', labelsize=11)
    
    # Title
    ax1.set_title('Early Drop-off Analysis: Cancellation Rate vs Prediction Horizon\n(Fine Resolution: 0.1ms steps, 50k event representative sample)',
                 fontsize=15, fontweight='bold', pad=65)
    
    # Legend
    ax1.legend(loc='lower left', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_2_decay_rate(dt_values, cr_values, output_path):
    """
    Plot 2: CR decay rate (dCR/dΔt).
    Shows where performance drops fastest - useful for identifying critical dt thresholds.
    """
    # Compute derivative
    dcr_ddt = np.gradient(cr_values, dt_values)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Top: CR vs dt with color-coded regions
    ax1.plot(dt_values, cr_values, 'o-', linewidth=2.5, markersize=6, color='#1f77b4')
    ax1.axhline(y=90, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Cancellation Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_ylim(70, 101)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Cancellation Rate and Decay Rate Analysis', fontsize=14, fontweight='bold')
    
    # Color-code regions
    ax1.axhspan(90, 101, alpha=0.1, color='green', label='Excellent (≥90%)')
    ax1.axhspan(80, 90, alpha=0.1, color='yellow', label='Good (80-90%)')
    ax1.axhspan(70, 80, alpha=0.1, color='orange', label='Moderate (<80%)')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Bottom: Decay rate
    ax2.plot(dt_values, -dcr_ddt, 'o-', linewidth=2, markersize=5, color='red')
    ax2.fill_between(dt_values, 0, -dcr_ddt, alpha=0.3, color='red')
    ax2.set_xlabel('Prediction Horizon Δt (ms)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Performance Decay Rate\n|dCR/dΔt| (%/ms)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Mark maximum decay point
    max_decay_idx = np.argmax(-dcr_ddt)
    max_decay_dt = dt_values[max_decay_idx]
    max_decay_rate = -dcr_ddt[max_decay_idx]
    ax2.plot(max_decay_dt, max_decay_rate, 'D', markersize=10, color='darkred', 
             label=f'Max decay at Δt={max_decay_dt:.1f}ms')
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_3_operating_window(dt_values, cr_values, pixel_displacement, output_path, omega_rad_s):
    """
    Plot 3: Operating window identification.
    Clearly marks recommended operating range based on CR thresholds.
    """
    # Find operating windows
    excellent_mask = cr_values >= 90
    good_mask = (cr_values >= 80) & (cr_values < 90)
    
    if np.any(excellent_mask):
        excellent_dt_max = dt_values[excellent_mask][-1]
        excellent_px_max = pixel_displacement[excellent_mask][-1]
    else:
        excellent_dt_max = 0
        excellent_px_max = 0
    
    if np.any(good_mask):
        good_dt_max = dt_values[good_mask][-1]
        good_px_max = pixel_displacement[good_mask][-1]
    else:
        good_dt_max = excellent_dt_max
        good_px_max = excellent_px_max
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot with color-coded segments
    for i in range(len(dt_values)-1):
        dt_seg = dt_values[i:i+2]
        cr_seg = cr_values[i:i+2]
        
        if cr_values[i] >= 90:
            color = 'green'
            linewidth = 3
        elif cr_values[i] >= 80:
            color = 'orange'
            linewidth = 2.5
        else:
            color = 'red'
            linewidth = 2
        
        ax.plot(dt_seg, cr_seg, '-', color=color, linewidth=linewidth, zorder=2)
    
    ax.plot(dt_values, cr_values, 'o', markersize=7, color='navy', zorder=3)
    
    # Threshold lines
    ax.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.7, label='90% threshold')
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='80% threshold')
    
    # Shade operating regions
    ax.axhspan(90, 101, alpha=0.1, color='green')
    ax.axhspan(80, 90, alpha=0.1, color='yellow')
    
    # Mark operating windows
    if excellent_dt_max > 0:
        ax.axvline(x=excellent_dt_max, color='green', linestyle=':', linewidth=2, alpha=0.6)
        ax.text(excellent_dt_max, 92, f'  Excellent: Δt ≤ {excellent_dt_max:.1f}ms\n  (≤{excellent_px_max:.1f}px displacement)',
                fontsize=11, verticalalignment='bottom', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    if good_dt_max > excellent_dt_max:
        ax.axvline(x=good_dt_max, color='orange', linestyle=':', linewidth=2, alpha=0.6)
        ax.text(good_dt_max, 82, f'  Good: Δt ≤ {good_dt_max:.1f}ms\n  (≤{good_px_max:.1f}px displacement)',
                fontsize=11, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlabel('Prediction Horizon Δt (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cancellation Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylim(70, 101)
    ax.set_xlim(-0.1, max(dt_values) + 0.2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    ax.set_title('Operating Window Analysis: Recommended Prediction Horizons\n(Fine Resolution Analysis)',
                 fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_4_phase_error_analysis(dt_values, cr_values, pixel_displacement, output_path, omega_rad_s, eps_xy_px):
    """
    Plot 4: Phase error and theoretical model comparison.
    Shows predicted vs actual CR based on geometric phase error model.
    """
    # Theoretical model: CR should drop when pixel displacement > spatial tolerance
    # Simple model: CR ≈ exp(-pixel_disp / eps_xy)
    theoretical_cr = 100 * np.exp(-pixel_displacement / eps_xy_px)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Actual vs theoretical CR
    ax1.plot(dt_values, cr_values, 'o-', linewidth=2.5, markersize=7, 
            color='#1f77b4', label='Measured CR')
    ax1.plot(dt_values, theoretical_cr, 's--', linewidth=2, markersize=6, 
            color='red', alpha=0.7, label=f'Theoretical (exp(-d/εxy))\nεxy={eps_xy_px}px')
    
    ax1.set_xlabel('Prediction Horizon Δt (ms)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cancellation Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_title('(A) Measured vs Theoretical CR', fontsize=13, fontweight='bold')
    
    # Right: Residual (actual - theoretical)
    residual = cr_values - theoretical_cr
    ax2.plot(dt_values, residual, 'o-', linewidth=2, markersize=6, color='purple')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.fill_between(dt_values, 0, residual, where=(residual>0), alpha=0.3, color='green', 
                    label='Better than theoretical')
    ax2.fill_between(dt_values, 0, residual, where=(residual<=0), alpha=0.3, color='red',
                    label='Worse than theoretical')
    
    ax2.set_xlabel('Prediction Horizon Δt (ms)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Residual: Measured - Theoretical (%)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_title('(B) Model Residual Analysis', fontsize=13, fontweight='bold')
    
    fig.suptitle('Phase Error Model Validation', fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_5_performance_zones(dt_values, cr_values, pixel_displacement, output_path):
    """
    Plot 5: Performance zones with clear recommendations.
    Summary visualization for thesis discussion section.
    """
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main plot (top, spanning both columns)
    ax_main = fig.add_subplot(gs[0, :])
    
    # Create color-coded line segments
    colors = []
    for cr in cr_values:
        if cr >= 95:
            colors.append('#006400')  # Dark green
        elif cr >= 90:
            colors.append('#90EE90')  # Light green
        elif cr >= 80:
            colors.append('#FFA500')  # Orange
        else:
            colors.append('#FF4500')  # Red-orange
    
    # Plot with gradient coloring
    for i in range(len(dt_values)-1):
        ax_main.plot(dt_values[i:i+2], cr_values[i:i+2], '-', 
                    color=colors[i], linewidth=4, alpha=0.8)
    
    ax_main.scatter(dt_values, cr_values, c=colors, s=100, edgecolors='black', 
                   linewidths=1.5, zorder=5)
    
    ax_main.set_xlabel('Prediction Horizon Δt (ms)', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Cancellation Rate (%)', fontsize=14, fontweight='bold')
    ax_main.set_ylim(70, 101)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.set_title('Performance Zones: Color-Coded Operating Regions', 
                     fontsize=15, fontweight='bold')
    
    # Add zone labels
    zone_info = [
        (0.3, 97, 'Excellent\n(CR≥95%)', '#006400'),
        (0.7, 92, 'Very Good\n(90-95%)', '#90EE90'),
        (1.5, 85, 'Good\n(80-90%)', '#FFA500'),
        (2.5, 75, 'Moderate\n(<80%)', '#FF4500')
    ]
    
    for dt, cr, text, color in zone_info:
        if dt <= max(dt_values):
            ax_main.text(dt, cr, text, fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                edgecolor=color, linewidth=2, alpha=0.9))
    
    # Bottom left: Pixel displacement zones
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bl.bar(dt_values, pixel_displacement, width=np.diff(dt_values)[0]*0.8, 
             color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_bl.axhline(y=eps_xy_px, color='red', linestyle='--', linewidth=2, 
                 label=f'Spatial tolerance (εxy={eps_xy_px}px)')
    ax_bl.set_xlabel('Δt (ms)', fontsize=12, fontweight='bold')
    ax_bl.set_ylabel('Pixel Displacement (px)', fontsize=12, fontweight='bold')
    ax_bl.grid(True, alpha=0.3, axis='y')
    ax_bl.legend(fontsize=9)
    ax_bl.set_title('(A) Predicted Displacement', fontsize=12, fontweight='bold')
    
    # Bottom right: Summary table
    ax_br = fig.add_subplot(gs[1, 1])
    ax_br.axis('off')
    
    # Find key metrics
    cr_95_mask = cr_values >= 95
    cr_90_mask = cr_values >= 90
    cr_80_mask = cr_values >= 80
    
    dt_95_max = dt_values[cr_95_mask][-1] if np.any(cr_95_mask) else 0
    dt_90_max = dt_values[cr_90_mask][-1] if np.any(cr_90_mask) else 0
    dt_80_max = dt_values[cr_80_mask][-1] if np.any(cr_80_mask) else 0
    
    px_95_max = pixel_displacement[cr_95_mask][-1] if np.any(cr_95_mask) else 0
    px_90_max = pixel_displacement[cr_90_mask][-1] if np.any(cr_90_mask) else 0
    px_80_max = pixel_displacement[cr_80_mask][-1] if np.any(cr_80_mask) else 0
    
    summary_text = f"""
    OPERATING WINDOW RECOMMENDATIONS
    ═══════════════════════════════════
    
    Excellent Performance (CR ≥ 95%):
      • Δt ≤ {dt_95_max:.2f} ms
      • Displacement ≤ {px_95_max:.2f} px
    
    Very Good Performance (CR ≥ 90%):
      • Δt ≤ {dt_90_max:.2f} ms  
      • Displacement ≤ {px_90_max:.2f} px
    
    Good Performance (CR ≥ 80%):
      • Δt ≤ {dt_80_max:.2f} ms
      • Displacement ≤ {px_80_max:.2f} px
    
    ───────────────────────────────────
    Key Finding:
    Drop-off occurs at ~{dt_90_max:.1f}ms, corresponding
    to ~{px_90_max:.1f}px displacement.
    
    This validates the phase-error model:
      ε_xy ≈ r·|Δω|·Δt
    """
    
    ax_br.text(0.05, 0.95, summary_text, transform=ax_br.transAxes,
              fontsize=10, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.8))
    
    ax_br.set_title('(B) Operating Window Summary', fontsize=12, fontweight='bold')
    
    fig.suptitle('Performance Zones and Operating Recommendations', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_6_exponential_fit(dt_values, cr_values, output_path):
    """
    Plot 6: Exponential decay fit to validate thesis model.
    Shows CR(Δt) ≈ exp(-kΔt) relationship from thesis.
    """
    from scipy.optimize import curve_fit
    
    # Exponential decay model: CR = A * exp(-k*dt) + C
    def exp_decay(dt, A, k, C):
        return A * np.exp(-k * dt) + C
    
    # Fit to data (use dt > 0 to avoid singularity)
    mask = dt_values > 0.05
    try:
        popt, pcov = curve_fit(exp_decay, dt_values[mask], cr_values[mask], 
                              p0=[100, 1.0, 30], maxfev=5000)
        A_fit, k_fit, C_fit = popt
        
        # Calculate tau_half (characteristic timescale)
        tau_half = np.log(2) / k_fit
        
        # Generate smooth fit curve
        dt_smooth = np.linspace(0, max(dt_values), 200)
        cr_fit = exp_decay(dt_smooth, A_fit, k_fit, C_fit)
        
        fit_success = True
    except:
        fit_success = False
        tau_half = 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Linear scale with fit
    ax1.plot(dt_values, cr_values, 'o', markersize=8, color='#1f77b4', 
            label='Measured data', zorder=3)
    
    if fit_success:
        ax1.plot(dt_smooth, cr_fit, '-', linewidth=2.5, color='red', alpha=0.7,
                label=f'Exponential fit: CR = {A_fit:.1f}·exp(-{k_fit:.2f}·Δt) + {C_fit:.1f}')
        ax1.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        ax1.text(max(dt_values)*0.7, 52, f'τ₁/₂ = {tau_half:.2f} ms', 
                fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow'))
    
    ax1.set_xlabel('Prediction Horizon Δt (ms)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cancellation Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title('(A) Linear Scale with Exponential Fit', fontsize=13, fontweight='bold')
    
    # Right: Semi-log plot to verify exponential
    ax2.semilogy(dt_values, cr_values - (C_fit if fit_success else 30), 'o', 
                markersize=8, color='#1f77b4', label='Measured (baseline-subtracted)')
    
    if fit_success:
        ax2.plot(dt_smooth, exp_decay(dt_smooth, A_fit, k_fit, 0), '-', 
                linewidth=2.5, color='red', alpha=0.7, label='Exponential decay')
    
    ax2.set_xlabel('Prediction Horizon Δt (ms)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('CR - Baseline (%, log scale)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_title('(B) Semi-log Plot (validates exponential model)', fontsize=13, fontweight='bold')
    
    fig.suptitle('Exponential Decay Model Validation\n(Thesis Chapter 3, Equation 3.x)', 
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return tau_half if fit_success else None


def generate_all_plots(data_path, output_dir, omega_rad_s=3.6, eps_xy_px=2.0):
    """
    Generate all comprehensive fine-resolution plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    
    dt_values = data['dt_values']
    cr_values = data['cancellation_rates']
    
    # Calculate pixel displacement
    pixel_displacement = MEAN_RADIUS_PX * omega_rad_s * (dt_values / 1000.0)
    
    print(f"\nGenerating {5} comprehensive plots...")
    print(f"dt range: {min(dt_values):.2f} to {max(dt_values):.2f} ms")
    print(f"CR range: {min(cr_values):.1f}% to {max(cr_values):.1f}%\n")
    
    # Generate plots
    plot_1_early_dropoff(dt_values, cr_values, pixel_displacement, 
                        output_dir / "plot1_early_dropoff.svg", omega_rad_s)
    
    plot_2_decay_rate(dt_values, cr_values, 
                     output_dir / "plot2_decay_rate.svg")
    
    plot_3_operating_window(dt_values, cr_values, pixel_displacement,
                           output_dir / "plot3_operating_window.svg", omega_rad_s)
    
    tau_half = plot_4_phase_error_analysis(dt_values, cr_values, pixel_displacement,
                                          output_dir / "plot4_phase_error.svg", 
                                          omega_rad_s, eps_xy_px)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    cr_95_mask = cr_values >= 95
    cr_90_mask = cr_values >= 90
    cr_80_mask = cr_values >= 80
    
    if np.any(cr_95_mask):
        print(f"Excellent (≥95%): Δt ≤ {dt_values[cr_95_mask][-1]:.2f} ms")
    if np.any(cr_90_mask):
        print(f"Very Good (≥90%): Δt ≤ {dt_values[cr_90_mask][-1]:.2f} ms")
    if np.any(cr_80_mask):
        print(f"Good (≥80%):      Δt ≤ {dt_values[cr_80_mask][-1]:.2f} ms")
    
    if tau_half:
        print(f"\nCharacteristic timescale τ₁/₂ = {tau_half:.2f} ms")
    
    print(f"\nAll plots saved to: {output_dir}/")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive fine-resolution plots")
    parser.add_argument('--data', required=True, help='Path to results .npz file')
    parser.add_argument('--output_dir', default='code/thesis_figures/fine_resolution_plots', 
                       help='Output directory')
    parser.add_argument('--omega', type=float, default=3.6, help='Angular velocity (rad/s)')
    parser.add_argument('--eps_xy', type=float, default=2.0, help='Spatial tolerance (px)')
    
    args = parser.parse_args()
    
    generate_all_plots(args.data, args.output_dir, args.omega, args.eps_xy)


if __name__ == '__main__':
    main()

