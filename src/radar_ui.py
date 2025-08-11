"""
Professional radar user interface with real-time updates - COMPLETE VERSION
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, CheckButtons
from matplotlib.patches import Circle, Wedge
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
        
        # Color scheme - Professional radar colors
        self.colors = {
            'background': '#000000',
            'grid': '#00FF00',           # Bright green grid
            'sweep': '#FFFF00',          # Yellow sweep line
            'sweep_sector': '#FF0000',   # Red sweep sector
            'aircraft': '#FF0000',       # Red aircraft
            'ship': '#0080FF',          # Blue ships
            'weather': '#00FFFF',        # Cyan weather
            'unknown': '#FFFFFF',        # White unknown
            'text': '#00FF00',          # Green text
            'highlight': '#FFFFFF'       # White highlights
        }
        
        # Data storage
        self.target_trails = {}
        self.sweep_patches = []
        self.performance_data = {
            'times': [],
            'track_counts': [],
            'detection_counts': [],
            'processing_times': []
        }
        
        self.setup_display()
    
    def setup_display(self):
        """Set up the radar display interface with proper spacing"""
        # Create figure with better proportions
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 12), facecolor='black')
        
        # Use a simple 2x2 grid instead of complex subplot2grid
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1],
                                  hspace=0.3, wspace=0.3)
        
        # Main radar display (large, left side)
        self.ax_radar = self.fig.add_subplot(gs[:, 0], projection='polar')
        
        # Information panels (right side)
        self.ax_targets = self.fig.add_subplot(gs[0, 1])
        self.ax_performance = self.fig.add_subplot(gs[1, 1])
        
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
                    color=self.colors['text'], fontsize=18, weight='bold', pad=40)
        
        # Range rings with proper styling
        if self.show_range_rings:
            theta_full = np.linspace(0, 2*np.pi, 360)
            for r in range(25, self.max_range_km + 1, 25):  # Every 25km
                ax.plot(theta_full, np.full_like(theta_full, r), 
                       color=self.colors['grid'], alpha=0.4, linewidth=0.8)
                
                # Range labels
                ax.text(0, r, f'{r}', color=self.colors['grid'], 
                       fontsize=9, ha='center', alpha=0.8)
        
        # Bearing lines
        if self.show_bearing_lines:
            for angle in range(0, 360, 15):  # Every 15 degrees
                theta_rad = np.radians(angle)
                alpha = 0.6 if angle % 30 == 0 else 0.3  # Major/minor lines
                linewidth = 1.0 if angle % 30 == 0 else 0.5
                
                ax.plot([theta_rad, theta_rad], [0, self.max_range_km], 
                       color=self.colors['grid'], alpha=alpha, linewidth=linewidth)
                
                # Major bearing labels
                if angle % 90 == 0:
                    labels = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
                    ax.text(theta_rad, self.max_range_km * 1.08, labels[angle], 
                           color=self.colors['text'], ha='center', va='center', 
                           fontsize=16, weight='bold')
                elif angle % 30 == 0:
                    ax.text(theta_rad, self.max_range_km * 1.05, f'{angle}°', 
                           color=self.colors['text'], ha='center', va='center', 
                           fontsize=10)
        
        # Initialize storage
        self.target_plots = {}
        
    def setup_info_panels(self):
        """Set up information panels with proper spacing"""
        # Target information panel
        ax = self.ax_targets
        ax.set_facecolor('black')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('TARGET INFORMATION', color=self.colors['text'], 
                    fontsize=14, weight='bold', pad=20)
        ax.axis('off')
        
        # Performance panel
        ax = self.ax_performance
        ax.set_facecolor('black')
        ax.set_title('SYSTEM PERFORMANCE', color=self.colors['text'], 
                    fontsize=14, weight='bold', pad=20)
        
    def update_display(self, radar_data: Dict):
        """Update the entire radar display"""
        # Update radar sweep with effects
        self.update_radar_sweep(radar_data)
        
        # Update target information
        self.update_target_info(radar_data)
        
        # Update performance metrics
        self.update_performance_panel(radar_data)
        
        # Update target trails
        self.update_target_trails(radar_data)
        
    def update_radar_sweep(self, radar_data: Dict):
        """Update the main radar display with full sweep effects"""
        ax = self.ax_radar
        
        # Clear previous dynamic elements
        for plot in self.target_plots.values():
            if hasattr(plot, 'remove'):
                plot.remove()
        self.target_plots.clear()
        
        # Clear previous sweep elements
        for patch in self.sweep_patches:
            if hasattr(patch, 'remove'):
                patch.remove()
        self.sweep_patches.clear()
        
        sweep_angle = radar_data.get('sweep_angle', 0)
        sweep_width = 20  # degrees
        
        # Create sweep sector (the red highlighted area)
        sweep_start_deg = sweep_angle - sweep_width/2
        sweep_end_deg = sweep_angle + sweep_width/2
        
        sweep_wedge = Wedge((0, 0), self.max_range_km, 
                           sweep_start_deg, sweep_end_deg,
                           facecolor=self.colors['sweep_sector'], alpha=0.15, 
                           edgecolor=self.colors['sweep_sector'], linewidth=2)
        ax.add_patch(sweep_wedge)
        self.sweep_patches.append(sweep_wedge)
        
        # Add the bright sweep line
        sweep_rad = np.radians(sweep_angle)
        sweep_line = ax.plot([sweep_rad, sweep_rad], [0, self.max_range_km], 
                            color=self.colors['sweep'], linewidth=4, alpha=1.0)[0]
        self.sweep_patches.append(sweep_line)
        
        # Add sweep beam gradient effect
        for i in range(5):
            fade_angle = sweep_angle - i * 2
            fade_rad = np.radians(fade_angle)
            fade_alpha = 0.8 - (i * 0.15)
            if fade_alpha > 0:
                fade_line = ax.plot([fade_rad, fade_rad], [0, self.max_range_km], 
                                   color=self.colors['sweep'], linewidth=2, alpha=fade_alpha)[0]
                self.sweep_patches.append(fade_line)
        
        # Plot confirmed tracks with effects
        confirmed_tracks = radar_data.get('confirmed_tracks', [])
        
        for track in confirmed_tracks:
            self.plot_track_with_effects(track, sweep_angle, sweep_width)
        
        # Update status text with better formatting
        num_tracks = len(confirmed_tracks)
        current_time = radar_data.get('current_time', 0)
        
        status_text = f"TRACKS: {num_tracks:2d} | SWEEP: {sweep_angle:3.0f}° | TIME: {current_time:6.1f}s"
        
        ax.text(0, self.max_range_km * 1.18, status_text,
               ha='center', va='center', color=self.colors['highlight'], 
               fontsize=14, weight='bold', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))

    def plot_track_with_effects(self, track, sweep_angle, sweep_width):
        """Plot a track with full radar effects"""
        ax = self.ax_radar
        
        # Convert position to polar coordinates
        x, y = track.state.x, track.state.y
        range_km = np.sqrt(x*x + y*y)
        bearing_rad = np.arctan2(x, y)
        bearing_deg = np.degrees(bearing_rad) % 360
        
        # Check if target is in sweep beam
        angle_diff = abs(((bearing_deg - sweep_angle + 180) % 360) - 180)
        in_beam = angle_diff <= sweep_width / 2
        recently_detected = angle_diff <= sweep_width  # Wider for trail effects
        
        # Get base color
        base_color = self.colors.get(track.classification, self.colors['unknown'])
        
        # Enhanced effects if in beam or recently detected
        if in_beam:
            # Bright, large symbol when actively being scanned
            color = base_color
            alpha = 1.0
            markersize = 16
            edge_color = self.colors['highlight']
            edge_width = 3
            
            # Add pulsing detection ring
            for ring_size in [1.05, 1.1, 1.15]:
                detection_ring = Circle((0, 0), range_km * ring_size, fill=False, 
                                      color=base_color, alpha=0.4/ring_size, linewidth=2,
                                      linestyle='--')
                ax.add_patch(detection_ring)
                self.sweep_patches.append(detection_ring)
                
        elif recently_detected:
            # Fade effect for recently scanned targets
            color = base_color
            alpha = 0.8
            markersize = 14
            edge_color = 'lightgray'
            edge_width = 2
        else:
            # Normal display for other targets
            color = base_color
            alpha = 0.6
            markersize = 12
            edge_color = 'gray'
            edge_width = 1
        
        # Choose symbol based on classification
        symbols = {
            'aircraft': 'o',
            'ship': 's', 
            'weather': '*',
            'unknown': 'D'
        }
        symbol = symbols.get(track.classification, 'D')
        
        # Plot target symbol
        plot = ax.plot(bearing_rad, range_km, symbol, 
                      color=color, markersize=markersize, alpha=alpha, 
                      markeredgecolor=edge_color, markeredgewidth=edge_width)[0]
        
        self.target_plots[track.id] = plot
        
        # Add target ID label
        if self.show_target_ids:
            label_color = self.colors['highlight'] if in_beam else color
            label_size = 12 if in_beam else 10
            label_weight = 'bold' if in_beam else 'normal'
            
            ax.text(bearing_rad, range_km + 10, track.id, 
                   color=label_color, ha='center', va='bottom', 
                   fontsize=label_size, weight=label_weight)
        
        # Add velocity vector for fast-moving targets
        if self.show_velocity_vectors and track.state.speed_kmh > 100:
            vel_scale = 0.3
            speed_ms = track.state.speed_kmh / 3.6  # Convert to m/s
            vel_length = speed_ms * vel_scale
            
            # Calculate velocity direction
            vel_bearing = np.radians(track.state.heading_deg)
            vel_end_x = range_km * np.cos(bearing_rad) + vel_length * np.sin(vel_bearing)
            vel_end_y = range_km * np.sin(bearing_rad) + vel_length * np.cos(vel_bearing)
            vel_end_range = np.sqrt(vel_end_x**2 + vel_end_y**2)
            vel_end_bearing = np.arctan2(vel_end_y, vel_end_x)
            
            if vel_end_range <= self.max_range_km:
                # Draw velocity vector
                vel_line = ax.plot([bearing_rad, vel_end_bearing], 
                                  [range_km, vel_end_range],
                                  color=color, linewidth=3, alpha=0.8)[0]
                self.sweep_patches.append(vel_line)
                
                # Add arrowhead
                arrow_size = 5
                ax.annotate('', xy=(vel_end_bearing, vel_end_range), 
                           xytext=(bearing_rad, range_km),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))

    def update_target_trails(self, radar_data: Dict):
        """Update target trails for all tracks"""
        confirmed_tracks = radar_data.get('confirmed_tracks', [])
        
        # Clean up trails for tracks that no longer exist
        active_track_ids = {track.id for track in confirmed_tracks}
        trails_to_remove = [tid for tid in self.target_trails.keys() if tid not in active_track_ids]
        
        for tid in trails_to_remove:
            if tid in self.target_trails:
                # Remove trail graphics
                trail_data = self.target_trails[tid]
                if 'plots' in trail_data:
                    for plot in trail_data['plots']:
                        if hasattr(plot, 'remove'):
                            plot.remove()
                del self.target_trails[tid]
        
        # Update trails for active tracks
        if self.show_trails:
            for track in confirmed_tracks:
                self.update_trail(track)

    def update_trail(self, track):
        """Update trail for a single track"""
        ax = self.ax_radar
        track_id = track.id
        
        # Convert position
        x, y = track.state.x, track.state.y
        range_km = np.sqrt(x*x + y*y)
        bearing_rad = np.arctan2(x, y)
        
        # Initialize trail if needed
        if track_id not in self.target_trails:
            self.target_trails[track_id] = {
                'positions': [],
                'plots': [],
                'color': self.colors.get(track.classification, self.colors['unknown'])
            }
        
        trail = self.target_trails[track_id]
        
        # Add current position
        trail['positions'].append((bearing_rad, range_km))
        
        # Limit trail length
        if len(trail['positions']) > self.trail_length:
            trail['positions'] = trail['positions'][-self.trail_length:]
        
        # Remove old trail plots
        for plot in trail['plots']:
            if hasattr(plot, 'remove'):
                plot.remove()
        trail['plots'].clear()
        
        # Draw trail
        if len(trail['positions']) > 1:
            positions = trail['positions']
            for i in range(len(positions) - 1):
                alpha = (i + 1) / len(positions) * 0.5  # Fade trail
                trail_line = ax.plot([positions[i][0], positions[i+1][0]], 
                                   [positions[i][1], positions[i+1][1]], 
                                   color=trail['color'], alpha=alpha, linewidth=2)[0]
                trail['plots'].append(trail_line)

    def update_target_info(self, radar_data: Dict):
        """Update target information panel with proper spacing"""
        ax = self.ax_targets
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('TARGET INFORMATION', color=self.colors['text'], 
                    fontsize=14, weight='bold', pad=20)
        ax.axis('off')
        
        confirmed_tracks = radar_data.get('confirmed_tracks', [])
        
        if not confirmed_tracks:
            ax.text(0.5, 0.5, 'NO CONFIRMED TRACKS', 
                   ha='center', va='center', color=self.colors['text'], 
                   fontsize=14, weight='bold')
            return
        
        # Sort tracks by range (closest first)
        sorted_tracks = sorted(confirmed_tracks, key=lambda t: t.state.range_km)
        
        # Display tracks with much better spacing
        y_start = 0.9
        track_height = 0.28  # More space per track
        
        for i, track in enumerate(sorted_tracks[:3]):  # Only show 3 closest tracks
            y_pos = y_start - (i * track_height)
            
            # Track header with classification color
            header = f"{track.id} - {track.classification.upper()}"
            header_color = self.colors.get(track.classification, self.colors['unknown'])
            ax.text(0.05, y_pos, header, color=header_color, 
                   fontsize=13, weight='bold')
            
            # Track details with proper line spacing
            details = [
                f"Range: {track.state.range_km:.1f} km",
                f"Bearing: {track.state.bearing_deg:.1f}°", 
                f"Speed: {track.state.speed_kmh:.0f} km/h",
                f"Heading: {track.state.heading_deg:.0f}°",
                f"Quality: {track.quality_score:.2f}"
            ]
            
            for j, detail in enumerate(details):
                detail_y = y_pos - 0.04 - (j * 0.035)  # Proper spacing
                ax.text(0.1, detail_y, detail, 
                       color=self.colors['text'], fontsize=11)
            
            if y_pos - track_height < 0.1:  # Check if we have room
                if i < len(sorted_tracks) - 1:
                    remaining = len(sorted_tracks) - i - 1
                    ax.text(0.05, 0.05, f"... and {remaining} more tracks", 
                           color=self.colors['text'], fontsize=10, style='italic')
                break
    
    def update_performance_panel(self, radar_data: Dict):
        """Update performance metrics panel"""
        ax = self.ax_performance
        ax.clear()
        ax.set_facecolor('black')
        ax.set_title('SYSTEM PERFORMANCE', color=self.colors['text'], 
                    fontsize=14, weight='bold', pad=20)
        
        # Current metrics
        track_count = len(radar_data.get('confirmed_tracks', []))
        detection_count = len(radar_data.get('processed_detections', []))
        processing_time = radar_data.get('processing_time', 0) * 1000
        current_time = radar_data.get('current_time', 0)
        
        # Store performance data for trending
        self.performance_data['times'].append(current_time)
        self.performance_data['track_counts'].append(track_count)
        self.performance_data['detection_counts'].append(detection_count)
        self.performance_data['processing_times'].append(processing_time)
        
        # Keep only recent data (last 60 points)
        if len(self.performance_data['times']) > 60:
            for key in self.performance_data:
                self.performance_data[key] = self.performance_data[key][-60:]
        
        # Performance status
        status = "OPTIMAL" if processing_time < 10 else "NOMINAL" if processing_time < 50 else "DEGRADED"
        status_color = self.colors['text'] if processing_time < 10 else 'yellow' if processing_time < 50 else 'red'
        
        # Display metrics
        metrics_text = f"""SYSTEM STATUS: {status}

CURRENT METRICS:
  Confirmed Tracks: {track_count:2d}
  Active Detections: {detection_count:2d}
  Processing Time: {processing_time:5.1f} ms
  System Uptime: {current_time:6.1f} s

PERFORMANCE:
  Real-time Capable: {'YES' if processing_time < 100 else 'NO'}
  Update Rate: 10.0 Hz
  Sweep Rate: 60 RPM
  Detection Range: {self.max_range_km} km

RADAR STATUS:
  Transmitter: ACTIVE
  Receiver: ACTIVE  
  Signal Processor: ACTIVE
  Tracker: ACTIVE"""

        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               va='top', ha='left', color=self.colors['text'], 
               fontsize=10, family='monospace')
        
        # Add status indicator
        ax.text(0.95, 0.95, f"● {status}", transform=ax.transAxes,
               va='top', ha='right', color=status_color, 
               fontsize=12, weight='bold')
        
        ax.axis('off')

# Test function with more realistic data
def test_radar_ui():
    """Test the radar UI with simulated data"""
    print("Testing Complete Professional Radar UI")
    print("=" * 40)
    
    # Create radar HUD
    hud = RadarHUD(max_range_km=200)
    
    # Mock track class with more realistic data
    class MockTrack:
        def __init__(self, track_id, x, y, vx, vy, classification):
            self.id = track_id
            self.classification = classification
            self.quality_score = np.random.uniform(0.7, 0.95)
            self.age = np.random.uniform(10, 120)
            
            # Mock state with realistic values
            speed_kmh = np.sqrt(vx*vx + vy*vy) * 3.6
            self.state = type('State', (), {
                'x': x, 'y': y, 'vx': vx, 'vy': vy,
                'range_km': np.sqrt(x*x + y*y),
                'bearing_deg': np.degrees(np.arctan2(x, y)) % 360,
                'speed_kmh': speed_kmh,
                'heading_deg': np.degrees(np.arctan2(vx, vy)) % 360
            })()
    
    # Create realistic mock tracks
    tracks = [
        MockTrack("AC001", 75, 100, 150, -50, "aircraft"),    # Commercial airliner
        MockTrack("SH001", -80, 60, -8, 5, "ship"),          # Cargo ship  
        MockTrack("AC002", 120, -80, 200, 100, "aircraft"),  # Fighter jet
        MockTrack("WX001", 50, 150, 20, -10, "weather"),     # Storm cell
        MockTrack("AC003", -40, 130, 180, -30, "aircraft"),  # Regional aircraft
    ]
    
    # Simulate realistic radar data
    radar_data = {
        'confirmed_tracks': tracks,
        'processed_detections': list(range(len(tracks) + 2)),  # Some extra detections
        'sweep_angle': 75,  # Currently scanning northeast
        'current_time': 1847.3,  # System uptime
        'processing_time': 0.0035  # 3.5ms processing time
    }
    
    # Update display
    hud.update_display(radar_data)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Complete professional radar UI test finished!")
if __name__ == "__main__":
    test_radar_ui()