#!/usr/bin/env python3
"""
Create a simple, reliable animation showing the forward prediction algorithm.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import os

# Import core functions
from predict_events import load_event_data_fast, load_tracker_series, apply_rotation, interp1

# Configuration
REAL_EVENTS_FILE = "/home/sumit/anu_research/recording/new_data/perlin_1280hz_hand_outframe.csv"
TRACKER_CSV_FILE = "/home/sumit/anu_research/ego_motion/results_csv/perlin_1280hz_hand_outframe_combined.csv"
DT_SECONDS = 0.002
OMEGA_SOURCE = "circlefit"

def create_simple_flow_diagram():
    """Create a clean, simple flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#2E7D32',      # Dark Green
        'process': '#1976D2',    # Blue  
        'output': '#F57C00',     # Orange
        'arrow': '#424242'       # Dark Gray
    }
    
    # Step boxes with better spacing
    steps = [
        ("Event Arrives\n(x, y, p, t)", 2, 2, colors['input']),
        ("Sample Motion\n(cx, cy, ω)", 5, 2, colors['process']),
        ("Apply Rotation\n(x', y') = R(x,y)", 8, 2, colors['process']),
        ("Emit Anti-Event\n(x', y', 1-p, t+Δt)", 11, 2, colors['output'])
    ]
    
    # Draw step boxes
    for text, x, y, color in steps:
        box = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1.0, 
                           boxstyle="round,pad=0.15", 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Draw arrows
    arrow_positions = [(2.8, 2), (5.8, 2), (8.8, 2)]
    for x, y in arrow_positions:
        ax.annotate('', xy=(x+0.4, y), xytext=(x-0.4, y),
                   arrowprops=dict(arrowstyle='->', lw=4, color=colors['arrow']))
    
    # Add mathematical formula
    ax.text(6.5, 0.8, r'Rotation: $x\' = c_x + \cos(\theta)(x-c_x) - \sin(\theta)(y-c_y)$', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
    
    ax.text(6.5, 0.4, r'$y\' = c_y + \sin(\theta)(x-c_x) + \cos(\theta)(y-c_y)$ where $\theta = \omega \cdot \Delta t$', 
            ha='center', va='center', fontsize=10, style='italic')
    
    ax.set_title('Forward Prediction Algorithm Flow', fontsize=18, fontweight='bold', pad=30)
    
    plt.tight_layout()
    return fig

def create_step_by_step_demo():
    """Create a step-by-step demonstration with real data"""
    print("Loading data for step-by-step demo...")
    
    # Load data
    real_events = load_event_data_fast(REAL_EVENTS_FILE)
    timestamps, center_x, center_y, omega = load_tracker_series(TRACKER_CSV_FILE, source=OMEGA_SOURCE)
    
    # Use first 500 events
    demo_events = real_events[:500]
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Step 1: Original events
    scatter1 = ax1.scatter(demo_events[:, 0], demo_events[:, 1], 
                          c=demo_events[:, 2], cmap='RdYlBu', s=4, alpha=0.8)
    ax1.set_title('Step 1: Original Events\n(x, y, polarity, t)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X coordinate (pixels)')
    ax1.set_ylabel('Y coordinate (pixels)')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1280)
    ax1.set_ylim(0, 720)
    
    # Step 2: Motion parameters
    time_subset = timestamps[timestamps <= demo_events[-1, 3]]
    if len(time_subset) > 0:
        idx_subset = np.searchsorted(timestamps, time_subset)
        ax2.plot(time_subset, center_x[idx_subset], 'b-', label='Center X', linewidth=2, alpha=0.8)
        ax2.plot(time_subset, center_y[idx_subset], 'r-', label='Center Y', linewidth=2, alpha=0.8)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(time_subset, omega[idx_subset], 'g-', label='Angular Velocity', linewidth=2, alpha=0.8)
        ax2_twin.set_ylabel('ω (rad/s)', color='g', fontsize=12)
        ax2.set_ylabel('Center coordinates', fontsize=12)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_title('Step 2: Motion Parameters\n(cx, cy, ω) over time', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        ax2_twin.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # Step 3: Predicted events
    predicted_events = []
    for event in demo_events:
        x, y, p, t = event[0], event[1], event[2], event[3]
        
        # Sample motion parameters
        cx = interp1(t, timestamps, center_x)
        cy = interp1(t, timestamps, center_y)
        om = interp1(t, timestamps, omega)
        
        # Apply rotation
        px, py = apply_rotation(np.array([x]), np.array([y]), cx, cy, np.array([om]), DT_SECONDS)
        
        # Create anti-event
        predicted_events.append([px[0], py[0], 1.0 - p, t + DT_SECONDS])
    
    predicted_events = np.array(predicted_events)
    
    scatter3 = ax3.scatter(predicted_events[:, 0], predicted_events[:, 1], 
                          c=predicted_events[:, 2], cmap='RdYlBu', s=4, alpha=0.8)
    ax3.set_title('Step 3: Predicted Anti-Events\n(x\', y\', 1-p, t+Δt)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X coordinate (pixels)')
    ax3.set_ylabel('Y coordinate (pixels)')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 1280)
    ax3.set_ylim(0, 720)
    
    # Step 4: Transformation visualization
    ax4.scatter(demo_events[:, 0], demo_events[:, 1], 
               c='blue', s=3, alpha=0.6, label='Original Events')
    ax4.scatter(predicted_events[:, 0], predicted_events[:, 1], 
               c='red', s=3, alpha=0.6, label='Predicted Anti-Events')
    
    # Draw transformation arrows (every 25th event)
    for i in range(0, len(demo_events), 25):
        ax4.annotate('', xy=(predicted_events[i, 0], predicted_events[i, 1]),
                    xytext=(demo_events[i, 0], demo_events[i, 1]),
                    arrowprops=dict(arrowstyle='->', lw=1, alpha=0.4, color='green'))
    
    ax4.set_title('Step 4: Transformation\nBlue→Red: Original→Predicted', fontsize=14, fontweight='bold')
    ax4.set_xlabel('X coordinate (pixels)')
    ax4.set_ylabel('Y coordinate (pixels)')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()
    ax4.set_xlim(0, 1280)
    ax4.set_ylim(0, 720)
    
    plt.suptitle('Forward Prediction Algorithm: Step-by-Step Demonstration\nUsing Real Event Camera Data', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    return fig

def create_simple_animation():
    """Create a simple, reliable animation"""
    print("Creating simple animation...")
    
    # Load data
    real_events = load_event_data_fast(REAL_EVENTS_FILE)
    timestamps, center_x, center_y, omega = load_tracker_series(TRACKER_CSV_FILE, source=OMEGA_SOURCE)
    
    # Use first 50 events for smooth animation
    demo_events = real_events[:50]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    ax.invert_yaxis()
    ax.set_title('Forward Prediction Algorithm Animation', fontsize=16, fontweight='bold')
    ax.set_xlabel('X coordinate (pixels)')
    ax.set_ylabel('Y coordinate (pixels)')
    ax.grid(True, alpha=0.3)
    
    # Initialize empty plots
    original_scatter = ax.scatter([], [], c='blue', s=50, alpha=0.8, label='Original Event')
    predicted_scatter = ax.scatter([], [], c='red', s=50, alpha=0.8, label='Predicted Anti-Event')
    
    ax.legend(fontsize=12)
    
    def animate(frame):
        if frame >= len(demo_events):
            return original_scatter, predicted_scatter
        
        # Get current event
        event = demo_events[frame]
        x, y, p, t = event[0], event[1], event[2], event[3]
        
        # Sample motion parameters
        cx = interp1(t, timestamps, center_x)
        cy = interp1(t, timestamps, center_y)
        om = interp1(t, timestamps, omega)
        
        # Apply rotation
        px, py = apply_rotation(np.array([x]), np.array([y]), cx, cy, np.array([om]), DT_SECONDS)
        
        # Update plots
        original_scatter.set_offsets([[x, y]])
        predicted_scatter.set_offsets([[px[0], py[0]]])
        
        # Update title
        ax.set_title(f'Forward Prediction Algorithm\nEvent {frame+1}/{len(demo_events)}: '
                    f'ω={om:.2f} rad/s, Δt={DT_SECONDS*1000:.1f}ms', 
                    fontsize=14, fontweight='bold')
        
        return original_scatter, predicted_scatter
    
    # Create animation with fewer frames for reliability
    anim = animation.FuncAnimation(fig, animate, frames=len(demo_events), 
                                 interval=300, blit=False, repeat=True)
    
    return fig, anim

def main():
    """Main function"""
    print("Creating improved algorithm visualizations...")
    
    # Create output directory
    output_dir = "./algorithm_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Simple flow diagram
    print("1. Creating simple flow diagram...")
    fig1 = create_simple_flow_diagram()
    fig1.savefig(os.path.join(output_dir, "algorithm_flow_simple.png"), 
                 dpi=300, bbox_inches='tight')
    print("   Saved: algorithm_flow_simple.png")
    plt.close(fig1)
    
    # 2. Step-by-step demonstration
    print("2. Creating step-by-step demonstration...")
    fig2 = create_step_by_step_demo()
    fig2.savefig(os.path.join(output_dir, "algorithm_step_by_step.png"), 
                 dpi=300, bbox_inches='tight')
    print("   Saved: algorithm_step_by_step.png")
    plt.close(fig2)
    
    # 3. Simple animation
    print("3. Creating simple animation...")
    fig3, anim = create_simple_animation()
    
    # Save animation
    try:
        anim.save(os.path.join(output_dir, "algorithm_simple_animation.gif"), 
                 writer='pillow', fps=3, dpi=100)
        print("   Saved: algorithm_simple_animation.gif")
    except Exception as e:
        print(f"   Animation error: {e}")
        print("   Saving static version instead...")
        fig3.savefig(os.path.join(output_dir, "algorithm_static.png"), 
                     dpi=300, bbox_inches='tight')
        print("   Saved: algorithm_static.png")
    
    plt.close(fig3)
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("New files created:")
    print("  - algorithm_flow_simple.png (clean flow chart)")
    print("  - algorithm_step_by_step.png (4-step demo)")
    print("  - algorithm_simple_animation.gif (reliable animation)")

if __name__ == "__main__":
    main()


















