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
        
        #display settings
        self.show_trails = True
        self.show_range_rings = True
        self.show_bearing_lines = True
        self.show_target_ids = True
        self.show_velocity_vectors = True
        self.trail_length = 10
        
        #color scheme
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
        
        #data storage
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
        #create figure with dark background
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10), facecolor='black')
        
        #main radar display (large, left side)
        self.ax_radar = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=3, projection='polar')
        
        #information panels (right side)
        self.ax_targets = plt.subplot2grid((3, 4), (0, 2), colspan=2)
        self.ax_performance = plt.subplot2grid((3, 4), (1, 2), colspan=2)
        self.ax_controls = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        
        self.setup_radar_display()
        self.setup_info_panels()
        
    def setup_radar_display(self):
        """Configure the main radar display"""
        ax = self.ax_radar
        
        #radar display configuration
        ax.set_facecolor('black')
        ax.set_theta_zero_location('N')  #north at top
        ax.set_theta_direction(-1)       #clockwise
        ax.set_ylim(0, self.max_range_km)
        ax.set_title('RADAR DISPLAY - REAL-TIME TRACKING', 
                    color=self.colors['text'], fontsize=14, weight='bold', pad=20)
        
        #range rings
        if self.show_range_rings:
            for r in range(50, self.max_range_km + 1, 50):
                circle = Circle((0, 0), r, fill=False, color=self.colors['grid'], 
                              alpha=0.3, linewidth=0.8, transform=ax.transData._b)
                ax.add_patch(circle)
                
                #range labels
                ax.text(0, r, f'{r}km', color=self.colors['grid'], 
                       fontsize=8, ha='center', alpha=0.7)
        
        #bearing lines
        if self.show_bearing_lines:
            for angle in range(0, 360, 30):
                theta_rad = np.radians(angle)
                ax.plot([theta_rad, theta_rad], [0, self.max_range_km], 
                       color=self.colors['grid'], alpha=0.3, linewidth=0.8)
                
                #major bearing labels
                if angle % 90 == 0:
                    labels = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
                    ax.text(theta_rad, self.max_range_km * 1.05, labels[angle], 
                           color=self.colors['text'], ha='center', va='center', 
                           fontsize=12, weight='bold')
        
        #initialize sweep line
        self.sweep_line = None
        self.target_plots = {}
        
    def setup_info_panels(self):
        """Set up information and control panels"""
        #target information panel
        ax = self.ax_targets
        ax.set_facecolor('black')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('TARGET INFORMATION', color=self.colors['text'], fontsize=12, weight='bold')
        ax.axis('off')
        
        #performance panel
        ax = self.ax_performance
        ax.set_facecolor('black')
        ax.set_title('SYSTEM PERFORMANCE', color=self.colors['text'], fontsize=12, weight='bold')
        
        #controls panel
        ax = self.ax_controls
        ax.set_facecolor('black')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('SYSTEM CONTROLS', color=self.colors['text'], fontsize=12, weight='bold')
        ax.axis('off')
        
    def update_display(self, radar_data: Dict):
        """Update the entire radar display"""
        current_time = time.time()
        
        #update radar sweep
        self.update_radar_sweep(radar_data)
        
        #update target information
        self.update_target_info(radar_data)
        
        #update performance metrics
        self.update_performance_panel(radar_data)
        
        #update trails
        self.update_target_trails(radar_data)
        
    def update_radar_sweep(self, radar_data: Dict):
        """Update the main radar display"""
        ax = self.ax_radar
        
        #clear previous target plots
        for plot in self.target_plots.values():
            if plot in ax.collections or plot in ax.lines:
                plot.remove()
        self.target_plots.clear()
        
        #clear and redraw sweep line
        if self.sweep_line:
            self.sweep_line.remove()
        
        sweep_angle = radar_data.get('sweep_angle', 0)
        sweep_rad = np.radians(sweep_angle)
        self.sweep_line = ax.plot([sweep_rad, sweep_rad], [0, self.max_range_km], 
                                 color=self.colors['sweep'], linewidth=3, alpha=0.8)[0]
        
        #plot confirmed tracks
        confirmed_tracks = radar_data.get('confirmed_tracks', [])
        
        for track in confirmed_tracks:
            self.plot_track(track)
        
        #update status text
        status_text = f"TRACKS: {len(confirmed_tracks)} | "
        status_text += f"SWEEP: {sweep_angle:.0f}° | "
        status_text += f"TIME: {radar_data.get('current_time', 0):.1f}s"
        
        ax.text(0, self.max_range_km * 1.15, status_text,
               ha='center', va='center', color=self.colors['text'], 
               fontsize=10, weight='bold')
    
    def plot_track(self, track):
        """Plot a single track on the radar display"""
        ax = self.ax_radar
        
        #convert position to polar coordinates
        x, y = track.state.x, track.state.y
        range_km = np.sqrt(x*x + y*y)
        bearing_rad = np.arctan2(x, y)
        
        #choose color based on classification
        color = self.colors.get(track.classification, self.colors['unknown'])
        
        #plot target symbol
        symbol = 'o' if track.classification == 'aircraft' else \
                's' if track.classification == 'ship' else \
                '*' if track.classification == 'weather' else 'D'
        
        plot = ax.plot(bearing_rad, range_km, symbol, 
                      color=color, markersize=10, alpha=0.9, 
                      markeredgecolor='white', markeredgewidth=1)[0]
        
        self.target_plots[track.id] = plot
        
        #add target ID label
        if self.show_target_ids:
            ax.text(bearing_rad, range_km + 5, track.id, 
                   color=color, ha='center', va='bottom', 
                   fontsize=8, weight='bold')
        
        #add velocity vector
        if self.show_velocity_vectors and track.state.speed_kmh > 10:
            #scale velocity for display
            vel_scale = 0.1  #km/h to display units
            vel_x = track.state.vx * vel_scale
            vel_y = track.state.vy * vel_scale
            
            #convert to polar for display
            vel_range = np.sqrt(vel_x*vel_x + vel_y*vel_y)
            vel_bearing = np.arctan2(vel_x, vel_y)
            
            if vel_range > 1:  #only show if significant velocity
                ax.arrow(bearing_rad, range_km, 
                        vel_bearing - bearing_rad, vel_range,
                        head_width=2, head_length=3, 
                        fc=color, ec=color, alpha=0.6)
        
        #update trail
        if self.show_trails:
            self.update_trail(track.id, bearing_rad, range_km, color)
    
    def update_trail(self, track_id: str, bearing: float, range_km: float, color: str):
        """Update target trail"""
        if track_id not in self.target_trails:
            self.target_trails[track_id] = {'bearings': [], 'ranges': [], 'color': color}
        
        trail = self.target_trails[track_id]
        trail['bearings'].append(bearing)
        trail['ranges'].append(range_km)
        
        #limit trail length
        if len(trail['bearings']) > self.trail_length:
            trail['bearings'] = trail['bearings'][-self.trail_length:]
            trail['ranges'] = trail['ranges'][-self.trail_length:]
        
        #plot trail
        if len(trail['bearings']) > 1:
            ax = self.ax_radar
            for i in range(len(trail['bearings']) - 1):
                alpha = (i + 1) / len(trail['bearings']) * 0.5  #fade trail
                ax.plot([trail['bearings'][i], trail['bearings'][i+1]], 
                       [trail['ranges'][i], trail['ranges'][i+1]], 
                       color=color, alpha=alpha, linewidth=1)
    
    def update_target_info(self, radar_data: Dict):
        """Update target information panel"""
        ax = self.ax_targets
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('TARGET INFORMATION', color=self.colors['text'], fontsize=12, weight='bold')
        ax.axis('off')
        
        confirmed_tracks = radar_data.get('confirmed_tracks', [])
        
        if not confirmed_tracks:
            ax.text(0.5, 0.5, 'NO CONFIRMED TRACKS', 
                   ha='center', va='center', color=self.colors['text'], 
                   fontsize=12, weight='bold')
            return
        
        #display top 5 tracks
        y_pos = 0.9
        for i, track in enumerate(confirmed_tracks[:5]):
            #track header
            header = f"{track.id} - {track.classification.upper()}"
            ax.text(0.05, y_pos, header, color=self.colors[track.classification], 
                   fontsize=10, weight='bold')
            
            #track details
            details = [
                f"Range: {track.state.range_km:.1f} km",
                f"Bearing: {track.state.bearing_deg:.1f}°", 
                f"Speed: {track.state.speed_kmh:.0f} km/h",
                f"Heading: {track.state.heading_deg:.0f}°",
                f"Quality: {track.quality_score:.2f}",
                f"Age: {track.age:.1f}s"
            ]
            
            for j, detail in enumerate(details):
                ax.text(0.1, y_pos - 0.02 * (j + 1), detail, 
                       color=self.colors['text'], fontsize=8)
            
            y_pos -= 0.16
            
            if y_pos < 0.1:
                break
    
    def update_performance_panel(self, radar_data: Dict):
        """Update performance metrics panel"""
        ax = self.ax_performance
        ax.clear()
        ax.set_facecolor('black')
        ax.set_title('SYSTEM PERFORMANCE', color=self.colors['text'], fontsize=12, weight='bold')
        
        #store performance data
        current_time = radar_data.get('current_time', 0)
        track_count = len(radar_data.get('confirmed_tracks', []))
        detection_count = len(radar_data.get('processed_detections', []))
        processing_time = radar_data.get('processing_time', 0) * 1000  #convert to ms
        
        self.performance_data['times'].append(current_time)
        self.performance_data['track_counts'].append(track_count)
        self.performance_data['detection_counts'].append(detection_count)
        self.performance_data['processing_times'].append(processing_time)
        
        #keep only recent data (last 60 seconds)
        if len(self.performance_data['times']) > 60:
            for key in self.performance_data:
                self.performance_data[key] = self.performance_data[key][-60:]
        
        #plot performance metrics
        if len(self.performance_data['times']) > 1:
            times = self.performance_data['times']
            
            #tracks and detections
            ax.plot(times, self.performance_data['track_counts'], 
                   'r-', linewidth=2, label='Tracks')
            ax.plot(times, self.performance_data['detection_counts'], 
                   'b-', linewidth=2, label='Detections')
            
            ax.set_xlabel('Time (s)', color=self.colors['text'])
            ax.set_ylabel('Count', color=self.colors['text'])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            #set text colors
            ax.tick_params(colors=self.colors['text'])
            ax.xaxis.label.set_color(self.colors['text'])
            ax.yaxis.label.set_color(self.colors['text'])
        
        #add current metrics text
        metrics_text = f"Current Metrics:\n"
        metrics_text += f"Tracks: {track_count}\n"
        metrics_text += f"Detections: {detection_count}\n"
        metrics_text += f"Processing: {processing_time:.1f}ms\n"
        
        if self.performance_data['processing_times']:
            avg_processing = np.mean(self.performance_data['processing_times'][-10:])
            metrics_text += f"Avg Processing: {avg_processing:.1f}ms"
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               va='top', ha='left', color=self.colors['text'], 
               fontsize=9, family='monospace')

#test the radar UI
def test_radar_ui():
    """Test the radar UI with simulated data"""
    print("Testing Radar UI")
    print("=" * 30)
    
    #create radar HUD
    hud = RadarHUD(max_range_km=200)
    
    #simulate some track data
    class MockTrack:
        def __init__(self, track_id, x, y, vx, vy, classification):
            self.id = track_id
            self.classification = classification
            self.quality_score = 0.85
            self.age = 30.0
            
            #mock state
            self.state = type('State', (), {
                'x': x, 'y': y, 'vx': vx, 'vy': vy,
                'range_km': np.sqrt(x*x + y*y),
                'bearing_deg': np.degrees(np.arctan2(x, y)) % 360,
                'speed_kmh': np.sqrt(vx*vx + vy*vy) * 3.6,
                'heading_deg': np.degrees(np.arctan2(vx, vy)) % 360
            })()
    
    #create mock tracks
    tracks = [
        MockTrack("TRK_001", 75, 100, 10, -5, "aircraft"),
        MockTrack("TRK_002", -50, 80, -3, 2, "ship"),
        MockTrack("TRK_003", 120, -60, 15, 8, "aircraft")
    ]
    
    #simulate radar data
    radar_data = {
        'confirmed_tracks': tracks,
        'processed_detections': [1, 2, 3],  #mock detections
        'sweep_angle': 45,
        'current_time': 120.5,
        'processing_time': 0.002
    }
    
    #update display
    hud.update_display(radar_data)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Radar UI test complete!")

if __name__ == "__main__":
    test_radar_ui()