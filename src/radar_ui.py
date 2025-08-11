"""
Professional radar user interface with real-time updates
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, CheckButtons
from matplotlib.patches import Circle, Wedge
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional
import time

class RadarHUD:
    """Professional radar heads-up display"""
    
    def __init__(self, max_range_km=200, update_interval=100):
        self.max_range_km = max_range_km
        self.update_interval = update_interval
        
        # Display settings
        self.show_trails = True
        self.show_range_rings = True
        self.show_bearing_lines = True
        self.show_target_ids = True
        self.show_velocity_vectors = True
        self.trail_length = 10
        
        # Color scheme
        self.colors = {
            'background': '#000000',
            'grid': '#00FF00',
            'sweep': '#FFFF00',
            'aircraft': '#FF0000',
            'ship': '#0000FF', 
            'weather': '#00FFFF',
            'unknown': '#FFFFFF',
            'text': '#00FF00'
        }
        
        # Data storage
        self.target_trails = {}
        self.performance_data = {
            'times': [],
            'track_counts': [],
            'detection_counts': [],
            'processing_times': []
        }
        
        self.setup_display()
    
    def setup_display(self):
        """Set up the radar display interface"""
        # Create figure with dark background
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10), facecolor='black')
        
        # Main radar display (large, left side)
        self.ax_radar = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=3, projection='polar')
        
        # Information panels (right side)
        self.ax_targets = plt.subplot2grid((3, 4), (0, 2), colspan=2)
        self.ax_performance = plt.subplot2grid((3, 4), (1, 2), colspan=2)
        self.ax_controls = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        
        self.setup_radar_display()
        self.setup_info_panels()
        
    def setup_radar_display(self):
        """Configure the main radar display"""
        ax = self.ax_radar
        
        # Radar display configuration
        ax.set_facecolor('black')
        ax.set_theta_zero_location('N')  # North at top
        ax.set_theta_direction(-1)       # Clockwise
        ax.set_ylim(0, self.max_range_km)
        ax.set_title('RADAR DISPLAY - REAL-TIME TRACKING', 
                    color=self.colors['text'], fontsize=14, weight='bold', pad=20)
        
        # Range rings
        if self.show_range_rings:
            for r in range(50, self.max_range_km + 1, 50):
                circle = Circle((0, 0), r, fill=False, color=self.colors['grid'], 
                              alpha=0.3, linewidth=0.8, transform=ax.transData._b)
                ax.add_patch(circle)
                
                # Range labels
                ax.text(0, r, f'{r}km', color=self.colors['grid'], 
                       fontsize=8, ha='center', alpha=0.7)
        
        # Bearing lines
        if self.show_bearing_lines:
            for angle in range(0, 360, 30):
                theta_rad = np.radians(angle)
                ax.plot([theta_rad, theta_rad], [0, self.max_range_km], 
                       color=self.colors['grid'], alpha=0.3, linewidth=0.8)
                
                # Major bearing labels
                if angle % 90 == 0:
                    labels = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
                    ax.text(theta_rad, self.max_range_km * 1.05, labels[angle], 
                           color=self.colors['text'], ha='center', va='center', 
                           fontsize=12, weight='bold')
        
        # Initialize sweep line
        self.sweep_line = None
        self.target_plots = {}
        
    def setup_info_panels(self):
        """Set up information and control panels"""
        # Target information panel
        ax = self.ax_targets
        ax.set_facecolor('black')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('TARGET INFORMATION', color=self.colors['text'], fontsize=12, weight='bold')
        ax.axis('off')
        
        # Performance panel
        ax = self.ax_performance
        ax.set_facecolor('black')
        ax.set_title('SYSTEM PERFORMANCE', color=self.colors['text'], fontsize=12, weight='bold')
        
        # Controls panel
        ax = self.ax_controls
        ax.set_facecolor('black')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('SYSTEM CONTROLS', color=self.colors['text'], fontsize=12, weight='bold')
        ax.axis('off')
        
    def update_display(self, radar_data: Dict):
        """Update the entire radar display"""
        current_time = time.time()
        
        # Update radar sweep
        self.update_radar_sweep(radar_data)
        
        # Update target information
        self.update_target_info(radar_data)
        
        # Update performance metrics
        self.update_performance_panel(radar_data)
        
        # Update trails
        self.update_target_trails(radar_data)
        
    def update_radar_sweep(self, radar_data: Dict):
        """Update the main radar display"""
        ax = self.ax_radar
        
        # Clear previous target plots
        for plot in self.target_plots.values():
            if plot in ax.collections or plot in ax.lines:
                plot.remove()
        self.target_plots.clear()
        
        # Clear and redraw sweep line
        if self.sweep_line:
            self.sweep_line.remove()
        
        sweep_angle = radar_data.get('sweep_angle', 0)
        sweep_rad = np.radians(sweep_angle)
        self.sweep_line = ax.plot([sweep_rad, sweep_rad], [0, self.max_range_km], 
                                 color=self.colors['sweep'], linewidth=3, alpha=0.8)[0]
        
        # Plot confirmed tracks
        confirmed_tracks = radar_data.get('confirmed_tracks', [])
        
        for track in confirmed_tracks:
            self.plot_track(track)
        
        # Update status text
        status_text = f"TRACKS: {len(confirmed_tracks)} | "
        status_text += f"SWEEP: {sweep_angle:.0f}° | "
        status_text += f"TIME: {radar_data.get('current_time', 0):.1f}s"
        
        ax.text(0, self.max_range_km * 1.15, status_text,
               ha='center', va='center', color=self.colors['text'], 
               fontsize=10, weight='bold')
    
    def plot_track(self, track):
        """Plot a single track on the radar display"""
        ax = self.ax_radar
        
        # Convert position to polar coordinates
        x, y = track.state.x, track.state.y
        range_km = np.sqrt(x*x + y*y)
        bearing_rad = np.arctan2(x, y)
        
        # Choose color based on classification
        color = self.colors.get(track.classification, self.colors['unknown'])
        
        # Plot target symbol
        symbol = 'o' if track.classification == 'aircraft' else \
                's' if track.classification == 'ship' else \
                '*' if track.classification == 'weather' else 'D'
        
        plot = ax.plot(bearing_rad, range_km, symbol, 
                      color=color, markersize=10, alpha=0.9, 
                      markeredgecolor='white', markeredgewidth=1)[0]
        
        self.target_plots[track.id] = plot
        
        # Add target ID label
        if self.show_target_ids:
            ax.text(bearing_rad, range_km + 5, track.id, 
                   color=color, ha='center', va='bottom', 
                   fontsize=8, weight='bold')
        
        # Add velocity vector
        if self.show_velocity_vectors and track.state.speed_kmh > 10:
            # Scale velocity for display
            vel_scale = 0.1  # km/h to display units
            vel_x = track.state.vx * vel_scale
            vel_y = track.state.vy * vel_scale
            
            # Convert to polar for display
            vel_range = np.sqrt(vel_x*vel_x + vel_y*vel_y)
            vel_bearing = np.arctan2(vel_x, vel_y)
            
            if vel_range > 1:  # Only show if significant velocity
                ax.arrow(bearing_rad, range_km, 
                        vel_bearing - bearing_rad, vel_range,
                        head_width=2, head_length=3, 
                        fc=color, ec=color, alpha=0.6)
        
        # Update trail
        if self.show_trails:
            self.update_trail(track.id, bearing_rad, range_km, color)
    
