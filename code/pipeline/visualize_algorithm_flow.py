#!/usr/bin/env python3
"""
Visual demonstration of the forward prediction algorithm.
Shows: Event arrives → Sample motion → Apply rotation → Emit anti-event
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import time
import os

# Import core functions from your existing script
from predict_events import load_event_data_fast, load_tracker_series, apply_rotation, interp1

# Configuration
REAL_EVENTS_FILE = "/home/sumit/anu_research/recording/new_data/perlin_1280hz_hand_outframe.csv"
TRACKER_CSV_FILE = "/home/sumit/anu_research/ego_motion/results_csv/perlin_1280hz_hand_outframe_combined.csv"
DT_SECONDS = 0.002
OMEGA_SOURCE = "circlefit"

def create_algorithm_flow_diagram():
    """Create a static flow diagram showing the algorithm steps"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#4CAF50',      # Green
        'process': '#2196F3',    # Blue  
        'output': '#FF9800',     # Orange
        'arrow': '#666666'       # Gray
    }
    
    # Step boxes
    steps = [
        ("Event Arrives\n(x, y, p, t)", 1.5, 3, colors['input']),
        ("Sample Motion\n(cx, cy, ω)", 3.5, 3, colors['process']),
        ("Apply Rotation\n(x', y') = R(x,y)", 5.5, 3, colors['process']),
        ("Emit Anti-Event\n(x', y', 1-p, t+Δt)", 7.5, 3, colors['output'])
    ]
    
    # Draw step boxes
    for text, x, y, color in steps:
        box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, 
                           boxstyle="round,pad=0.1", 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw arrows
    arrow_positions = [(2.2, 3), (4.2, 3), (6.2, 3)]
    for x, y in arrow_positions:
        ax.annotate('', xy=(x+0.3, y), xytext=(x-0.3, y),
                   arrowprops=dict(arrowstyle='->', lw=3, color=colors['arrow']))
    
    # Add mathematical formula (simplified)
    ax.text(5, 1.5, r'Rotation: $x\' = c_x + \cos(\theta)(x-c_x) - \sin(\theta)(y-c_y)$' + '\n' + 
                   r'$y\' = c_y + \sin(\theta)(x-c_x) + \cos(\theta)(y-c_y)$',
            ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    ax.text(5, 0.8, f'Where: θ = ω·Δt (Δt = {DT_SECONDS*1000:.1f}ms)', 
            ha='center', va='center', fontsize=10, style='italic')
    
    ax.set_title('Forward Prediction Algorithm Flow', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_visual_demonstration():
    """Create a visual demonstration using real data"""
    print("Loading real data for demonstration...")
    
    # Load a small subset of real data
    real_events = load_event_data_fast(REAL_EVENTS_FILE)
    timestamps, center_x, center_y, omega = load_tracker_series(TRACKER_CSV_FILE, source=OMEGA_SOURCE)
    
    # Use first 1000 events for demonstration
    demo_events = real_events[:1000]
    print(f"Using {len(demo_events)} events for demonstration")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Original events
    ax1.scatter(demo_events[:, 0], demo_events[:, 1], 
               c=demo_events[:, 2], cmap='RdYlBu', s=3, alpha=0.7)
    ax1.set_title('Step 1: Original Events\n(x, y, polarity, t)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Plot 2: Motion parameters over time
    time_subset = timestamps[timestamps <= demo_events[-1, 3]]
    if len(time_subset) > 0:
        idx_subset = np.searchsorted(timestamps, time_subset)
        ax2.plot(time_subset, center_x[idx_subset], 'b-', label='Center X', alpha=0.7)
        ax2.plot(time_subset, center_y[idx_subset], 'r-', label='Center Y', alpha=0.7)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(time_subset, omega[idx_subset], 'g-', label='Angular Velocity', alpha=0.7)
        ax2_twin.set_ylabel('ω (rad/s)', color='g')
        ax2.set_ylabel('Center coordinates')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Step 2: Motion Parameters\n(cx, cy, ω) over time', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Predicted events (anti-events)
    predicted_events = []
    for event in demo_events:
        x, y, p, t = event[0], event[1], event[2], event[3]
        
        # Sample motion parameters at this event's timestamp
        cx = interp1(t, timestamps, center_x)
        cy = interp1(t, timestamps, center_y)
        om = interp1(t, timestamps, omega)
        
        # Apply rotation
        px, py = apply_rotation(np.array([x]), np.array([y]), cx, cy, np.array([om]), DT_SECONDS)
        
        # Create anti-event
        predicted_events.append([px[0], py[0], 1.0 - p, t + DT_SECONDS])
    
    predicted_events = np.array(predicted_events)
    
    ax3.scatter(predicted_events[:, 0], predicted_events[:, 1], 
               c=predicted_events[:, 2], cmap='RdYlBu', s=3, alpha=0.7)
    ax3.set_title('Step 3: Predicted Anti-Events\n(x\', y\', 1-p, t+Δt)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    # Plot 4: Combined view showing transformation
    ax4.scatter(demo_events[:, 0], demo_events[:, 1], 
               c='blue', s=2, alpha=0.5, label='Original Events')
    ax4.scatter(predicted_events[:, 0], predicted_events[:, 1], 
               c='red', s=2, alpha=0.5, label='Predicted Anti-Events')
    
    # Draw some example transformation arrows
    for i in range(0, len(demo_events), 50):  # Every 50th event
        ax4.annotate('', xy=(predicted_events[i, 0], predicted_events[i, 1]),
                    xytext=(demo_events[i, 0], demo_events[i, 1]),
                    arrowprops=dict(arrowstyle='->', lw=0.5, alpha=0.3, color='gray'))
    
    ax4.set_title('Step 4: Transformation Visualization\nBlue→Red: Original→Predicted', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X coordinate')
    ax4.set_ylabel('Y coordinate')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()
    
    plt.suptitle('Forward Prediction Algorithm Demonstration\nUsing Real Event Camera Data', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_animated_demonstration():
    """Create an animated version showing the algorithm in action"""
    print("Creating animated demonstration...")
    
    # Load data
    real_events = load_event_data_fast(REAL_EVENTS_FILE)
    timestamps, center_x, center_y, omega = load_tracker_series(TRACKER_CSV_FILE, source=OMEGA_SOURCE)
    
    # Use first 100 events for smooth animation
    demo_events = real_events[:100]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    ax.invert_yaxis()
    ax.set_title('Forward Prediction Algorithm Animation', fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)
    
    # Initialize empty plots
    original_scatter = ax.scatter([], [], c='blue', s=20, alpha=0.7, label='Original Event')
    predicted_scatter = ax.scatter([], [], c='red', s=20, alpha=0.7, label='Predicted Anti-Event')
    arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0), 
                      arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    ax.legend()
    
    def animate(frame):
        if frame >= len(demo_events):
            return original_scatter, predicted_scatter, arrow
        
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
        
        # Update arrow
        arrow.xy = (px[0], py[0])
        arrow.xytext = (x, y)
        
        # Update title with current info
        ax.set_title(f'Forward Prediction Algorithm Animation\nEvent {frame+1}/{len(demo_events)}: '
                    f'ω={om:.2f} rad/s, Δt={DT_SECONDS*1000:.1f}ms', 
                    fontsize=12, fontweight='bold')
        
        return original_scatter, predicted_scatter, arrow
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(demo_events), 
                                 interval=200, blit=False, repeat=True)
    
    return fig, anim

def main():
    """Main function to create all visualizations"""
    print("Creating algorithm flow visualizations...")
    
    # Create output directory
    output_dir = "./algorithm_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Static flow diagram
    print("1. Creating static flow diagram...")
    fig1 = create_algorithm_flow_diagram()
    fig1.savefig(os.path.join(output_dir, "algorithm_flow_diagram.png"), 
                 dpi=300, bbox_inches='tight')
    print("   Saved: algorithm_flow_diagram.png")
    
    # 2. Visual demonstration with real data
    print("2. Creating visual demonstration...")
    fig2 = create_visual_demonstration()
    fig2.savefig(os.path.join(output_dir, "algorithm_visualization.png"), 
                 dpi=300, bbox_inches='tight')
    print("   Saved: algorithm_visualization.png")
    
    # 3. Animated demonstration
    print("3. Creating animated demonstration...")
    fig3, anim = create_animated_demonstration()
    
    # Save animation as GIF
    try:
        anim.save(os.path.join(output_dir, "algorithm_animation.gif"), 
                 writer='pillow', fps=5, dpi=100)
        print("   Saved: algorithm_animation.gif")
    except Exception as e:
        print(f"   Could not save GIF: {e}")
        print("   Animation will be displayed instead")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("Files created:")
    print("  - algorithm_flow_diagram.png (static flow chart)")
    print("  - algorithm_visualization.png (4-step demonstration)")
    print("  - algorithm_animation.gif (animated version)")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
