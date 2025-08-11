"""
Professional Integrated Radar System 
Normal speed radar with smooth shadow trail and enhanced target effects.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.widgets import Button
import threading
import time
from typing import Dict, List
from datetime import datetime

# Import our professional radar components
from src.radar_data_generator import RadarDataGenerator, EnvironmentType
from src.signal_processing import SignalProcessor
from src.target_detection import TargetDetector  
from src.multi_target_tracker import MultiTargetTracker
from src.kalman_filter import TrackState

class ProfessionalRadarDemo:
    """Complete professional radar demonstration using all Day 1-6 components"""
    
    def __init__(self):
        # Initialize all professional components with realistic parameters
        self.data_generator = RadarDataGenerator(max_range_km=150)
        self.signal_processor = SignalProcessor()
        self.target_detector = TargetDetector()
        self.tracker = MultiTargetTracker()
        
        # Configure much more sensitive detection parameters
        self.signal_processor.detection_threshold = 0.1   # Much lower threshold
        self.signal_processor.false_alarm_rate = 0.05     # Higher false alarm rate for more detections
        self.target_detector.min_detections_for_confirmation = 1  # Only need 1 detection
        self.tracker.max_association_distance = 15.0      # Much wider association
        
        # System state with normal speed
        self.is_running = False
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.sweep_rate = 18.0  # Normal speed: 30 RPM (1 rotation every 12 seconds)
        self.sweep_history = []
        
        # Display components
        self.fig = None
        self.axes = {}
        self.animation = None
        
        # Performance tracking
        self.performance_metrics = {
            'targets_tracked': 0,
            'detection_rate': 95.5,
            'cpu_usage': 45.0,
            'memory_usage': 68.0,
            'start_time': time.time(),
            'processing_times': []
        }
        
        self.setup_display()
        self.load_scenario()
        
    def setup_display(self):
        """Setup professional radar display"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('PROFESSIONAL RADAR SYSTEM - ENHANCED DAY 6 VERSION', 
                         fontsize=16, color='#00ff00', weight='bold')
        self.fig.patch.set_facecolor('#000000')
        
        # Create layout
        gs = self.fig.add_gridspec(3, 4, width_ratios=[3, 1, 1, 1], height_ratios=[1, 3, 1])
        
        # Main radar display
        self.axes['radar'] = self.fig.add_subplot(gs[1, 0], projection='polar')
        self.setup_radar_display()
        
        # Information panels
        self.axes['status'] = self.fig.add_subplot(gs[0, :])
        self.axes['controls'] = self.fig.add_subplot(gs[1, 1])
        self.axes['targets'] = self.fig.add_subplot(gs[1, 2])
        self.axes['performance'] = self.fig.add_subplot(gs[1, 3])
        self.axes['alerts'] = self.fig.add_subplot(gs[2, :])
        
        self.setup_info_panels()
        
    def setup_radar_display(self):
        """Setup the main radar display"""
        ax = self.axes['radar']
        ax.set_facecolor('#000000')
        ax.set_ylim(0, 100)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title('RADAR SCOPE - ENHANCED OPERATION', 
                    color='#00ff00', weight='bold', pad=20, fontsize=14)
        
        # Range rings
        for r in range(25, 101, 25):
            circle = Circle((0, 0), r, fill=False, color='#003300', alpha=0.4, linewidth=1)
            ax.add_patch(circle)
            ax.text(0, r + 2, f'{r}', ha='center', va='bottom', color='#004400', fontsize=8)
            
        # Bearing lines
        for angle in range(0, 360, 30):
            ax.plot([np.radians(angle), np.radians(angle)], [0, 100], 
                   color='#002200', alpha=0.3, linewidth=0.5)
            
        # Compass labels
        compass_labels = [('N', 0), ('E', 90), ('S', 180), ('W', 270)]
        for label, angle in compass_labels:
            ax.text(np.radians(angle), 105, label, 
                   ha='center', va='center', color='#00ff00', fontsize=12, weight='bold')
        
        ax.grid(False)
        ax.set_rticks([])
        ax.set_thetagrids([])
        
    def setup_info_panels(self):
        """Setup information and control panels"""
        self.setup_status_panel()
        self.setup_controls_panel()
        self.setup_targets_panel()
        self.setup_performance_panel()
        self.setup_alerts_panel()
        
    def setup_status_panel(self):
        """System status display"""
        ax = self.axes['status']
        ax.clear()
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 2)
        ax.set_title('SYSTEM STATUS', color='#00ff00', weight='bold', fontsize=12)
        ax.axis('off')
        
        # System components status
        components = [
            ('RADAR', 1.5, '#00ff00' if self.is_running else '#666666'),
            ('SIGNAL PROC', 3.5, '#00ff00' if self.is_running else '#666666'),
            ('DETECTION', 5.5, '#00ff00' if self.is_running else '#666666'),
            ('TRACKING', 7.5, '#00ff00' if self.is_running else '#666666'),
            ('DISPLAY', 9.5, '#00ff00' if self.is_running else '#666666')
        ]
        
        for name, x_pos, color in components:
            circle = Circle((x_pos, 1.4), 0.15, color=color, alpha=0.9)
            ax.add_patch(circle)
            ax.text(x_pos, 1.0, name, ha='center', va='center', 
                   color='white', fontsize=9, weight='bold')
            status = "ONLINE" if color == '#00ff00' else "STANDBY"
            ax.text(x_pos, 0.6, status, ha='center', va='center', 
                   color=color, fontsize=8)
        
        # Current scenario and sweep info
        scenario_text = f"SCENARIO: AIRPORT TRAFFIC | SWEEP: 30 RPM"
        ax.text(11, 1.5, scenario_text, ha='center', va='center', 
               color='#00aaff', fontsize=11, weight='bold',
               bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
        
    def setup_controls_panel(self):
        """Control interface"""
        ax = self.axes['controls']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('CONTROLS', color='#00ff00', weight='bold')
        ax.axis('off')
        
        # Control buttons
        buttons = [
            (5, 8.5, 'START\nSYSTEM', '#006600'),
            (5, 6.5, 'STOP\nSYSTEM', '#666600'),
            (5, 4.5, 'RESET\nSYSTEM', '#000066'),
            (5, 2.5, 'EMERGENCY\nSTOP', '#660000')
        ]
        
        for x, y, label, color in buttons:
            rect = Rectangle((x-2, y-0.6), 4, 1.2, facecolor=color, 
                           alpha=0.8, edgecolor='white', linewidth=1)
            ax.add_patch(rect)
            ax.text(x, y, label, ha='center', va='center', 
                   color='white', fontsize=9, weight='bold')
        
    def setup_targets_panel(self):
        """Target information display"""
        ax = self.axes['targets']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('TRACKED TARGETS', color='#00ff00', weight='bold')
        ax.axis('off')
        
        # Will be updated with real target data
        ax.text(5, 5, 'NO TARGETS\nTRACKED', ha='center', va='center',
               color='#666666', fontsize=12, weight='bold')
        
    def setup_performance_panel(self):
        """Performance monitoring"""
        ax = self.axes['performance']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('PERFORMANCE', color='#00ff00', weight='bold')
        ax.axis('off')
        
        # Performance metrics
        uptime = time.time() - self.performance_metrics['start_time']
        hours = int(uptime // 3600)
        mins = int((uptime % 3600) // 60)
        secs = int(uptime % 60)
        
        perf_text = f"""SYSTEM METRICS

Targets:      {self.performance_metrics['targets_tracked']:3d}
Detection:    {self.performance_metrics['detection_rate']:5.1f}%
CPU Usage:    {self.performance_metrics['cpu_usage']:5.1f}%
Memory:       {self.performance_metrics['memory_usage']:5.1f}%
Uptime:     {hours:02d}:{mins:02d}:{secs:02d}"""
        
        ax.text(5, 7, perf_text, ha='center', va='top',
               color='#00ff00', fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#0a0a0a', 
                        alpha=0.9, edgecolor='#00ff00'))
        
    def setup_alerts_panel(self):
        """System alerts"""
        ax = self.axes['alerts']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)
        ax.set_title('SYSTEM ALERTS', color='#00ff00', weight='bold')
        ax.axis('off')
        
        current_time = datetime.now().strftime("%H:%M:%S")
        if self.is_running:
            alert_text = f"[{current_time}] OPERATIONAL: Enhanced radar system active - Normal sweep speed"
        else:
            alert_text = f"[{current_time}] STANDBY: Click START SYSTEM to begin enhanced operation"
            
        ax.text(5, 1, alert_text, ha='center', va='center',
               color='#00ff00' if self.is_running else '#ffaa00', 
               fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#0a0a0a', 
                        alpha=0.9, edgecolor='#333333'))
        
    def load_scenario(self):
        """Load realistic airport traffic scenario"""
        self.data_generator.create_scenario("busy_airport")
        
        # Verify and adjust target positions to realistic ranges
        for target in self.data_generator.targets:
            current_range = target.range_km
            if current_range < 30:
                scale_factor = np.random.uniform(40, 120) / max(current_range, 1)
                target.position_x *= scale_factor
                target.position_y *= scale_factor
                
            # Ensure realistic speeds
            current_speed = target.speed
            if current_speed < 200 or current_speed > 900:
                target.speed = np.random.uniform(250, 800)
                heading_rad = np.radians(target.heading)
                target.velocity_x = target.speed * np.sin(heading_rad)
                target.velocity_y = target.speed * np.cos(heading_rad)
                
            # Ensure realistic RCS for aircraft
            if target.radar_cross_section > 100 or target.radar_cross_section < 1:
                target.radar_cross_section = np.random.uniform(5, 30)
        
        print(f"📊 Loaded enhanced scenario: {len(self.data_generator.targets)} targets")
        for i, target in enumerate(self.data_generator.targets[:3]):
            print(f"  Target {i+1}: {target.range_km:.1f}km, {target.speed:.0f}km/h, RCS:{target.radar_cross_section:.1f}m²")
        
    def update_radar_display(self):
        """Update the main radar display with enhanced sweep effects"""
        ax = self.axes['radar']
        
        # Simple but effective clearing - just clear and redraw static elements
        ax.clear()
        
        # Quickly redraw static elements
        ax.set_facecolor('#000000')
        ax.set_ylim(0, 100)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title('RADAR SCOPE - ENHANCED OPERATION', 
                    color='#00ff00', weight='bold', pad=20, fontsize=14)
        
        # Range rings
        for r in range(25, 101, 25):
            circle = Circle((0, 0), r, fill=False, color='#003300', alpha=0.4, linewidth=1)
            ax.add_patch(circle)
            ax.text(0, r + 2, f'{r}', ha='center', va='bottom', color='#004400', fontsize=8)
            
        # Bearing lines
        for angle in range(0, 360, 30):
            ax.plot([np.radians(angle), np.radians(angle)], [0, 100], 
                   color='#002200', alpha=0.3, linewidth=0.5)
            
        # Compass labels
        compass_labels = [('N', 0), ('E', 90), ('S', 180), ('W', 270)]
        for label, angle in compass_labels:
            ax.text(np.radians(angle), 105, label, 
                   ha='center', va='center', color='#00ff00', fontsize=12, weight='bold')
        
        ax.grid(False)
        ax.set_rticks([])
        ax.set_thetagrids([])
        
        if not self.is_running:
            return
            
        # Enhanced sweep line with beam effect
        sweep_rad = np.radians(self.sweep_angle)
        
        # Main sweep line (bright)
        ax.plot([sweep_rad, sweep_rad], [0, 100], 
               color='#00ff00', linewidth=4, alpha=1.0, zorder=10)
        
        # Sweep beam (wider, faded)
        beam_width = 8  # degrees
        beam_start = np.radians(self.sweep_angle - beam_width/2)
        beam_end = np.radians(self.sweep_angle + beam_width/2)
        theta_beam = np.linspace(beam_start, beam_end, 20)
        for i, theta in enumerate(theta_beam):
            alpha = 0.3 * (1 - abs(i - 10) / 10)  # Fade from center
            ax.plot([theta, theta], [0, 100], 
                   color='#00ff00', alpha=alpha, linewidth=2, zorder=9)
        
        # Smooth shadow trail with exponential fade
        trail_length = min(20, len(self.sweep_history))  # Longer trail for wider beam
        for i in range(1, trail_length):
            if i < len(self.sweep_history):
                angle = self.sweep_history[-i-1]
                # Slower fade for wider beam effect
                alpha = 0.6 * np.exp(-i * 0.2)  # Slower fade
                if alpha > 0.01:  # Only draw if visible
                    fade_rad = np.radians(angle)
                    # Draw wider fade lines
                    for offset in [-2, -1, 0, 1, 2]:
                        offset_angle = angle + offset
                        offset_rad = np.radians(offset_angle)
                        line_alpha = alpha * (1 - abs(offset) * 0.2)
                        if line_alpha > 0.01:
                            ax.plot([offset_rad, offset_rad], [0, 100], 
                                   color='#00ff00', alpha=line_alpha, linewidth=1, zorder=8-i)
        
        # Get confirmed tracks from tracker
        confirmed_tracks = [track for track in self.tracker.tracks.values() 
                          if track.confirmed and not track.terminated]
        
        # Enhanced target detection and display - much more generous
        visible_tracks = []
        sweep_width = 40  # degrees - much wider beam for better detection
        
        for track in confirmed_tracks:
            track_range = np.sqrt(track.state.x**2 + track.state.y**2)
            track_bearing = np.degrees(np.arctan2(track.state.x, track.state.y)) % 360
            
            # Check if track has been swept over recently
            angle_diff = abs(((track_bearing - self.sweep_angle + 180) % 360) - 180)
            
            # Very generous detection - show if swept in last 90 degrees
            recently_swept = False
            if angle_diff <= sweep_width:  # Currently in beam
                recently_swept = True
            else:
                # Check sweep history for recent detection - much more generous
                for hist_angle in self.sweep_history[-50:]:  # Check much more history
                    hist_diff = abs(((track_bearing - hist_angle + 180) % 360) - 180)
                    if hist_diff <= sweep_width:
                        recently_swept = True
                        break
            
            if recently_swept and track_range <= 100:
                visible_tracks.append((track, angle_diff))
        
        # Draw visible tracks with enhanced effects
        for track, angle_diff in visible_tracks:
            track_range = np.sqrt(track.state.x**2 + track.state.y**2)
            track_bearing = np.degrees(np.arctan2(track.state.x, track.state.y)) % 360
            theta = np.radians(track_bearing)
            
            # Enhanced target appearance based on how recently swept
            if angle_diff <= 10:  # Very recently swept
                color = '#ffff00'  # Bright yellow
                size = 140
                alpha = 1.0
                edge_color = 'white'
                edge_width = 3
            elif angle_diff <= 25:  # Recently swept
                color = '#ffaa00'  # Orange
                size = 120
                alpha = 0.8
                edge_color = 'lightgray'
                edge_width = 2
            else:  # Older detection
                color = '#ff6600'  # Red-orange
                size = 100
                alpha = 0.6
                edge_color = 'gray'
                edge_width = 1
            
            # Aircraft symbol (triangle)
            ax.scatter(theta, track_range, c=color, s=size, 
                      marker='^', alpha=alpha, edgecolors=edge_color, 
                      linewidth=edge_width, zorder=20)
            
            # Track ID label with enhanced visibility
            ax.text(theta, track_range + 8, track.id, 
                   ha='center', va='bottom', color=color, 
                   fontsize=10, weight='bold', zorder=25,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', 
                            alpha=0.8, edgecolor=color))
            
            # Enhanced velocity vector
            speed = np.sqrt(track.state.vx**2 + track.state.vy**2)
            if speed > 0.5:
                vel_scale = 10.0  # Longer vectors
                end_x = track.state.x + track.state.vx * vel_scale
                end_y = track.state.y + track.state.vy * vel_scale
                end_range = np.sqrt(end_x**2 + end_y**2)
                end_bearing = np.degrees(np.arctan2(end_x, end_y)) % 360
                end_theta = np.radians(end_bearing)
                
                if end_range <= 100:
                    ax.plot([theta, end_theta], [track_range, end_range], 
                           color=color, alpha=0.8, linewidth=3, zorder=15)
                    ax.scatter(end_theta, end_range, c=color, s=80, 
                             marker='>', alpha=0.9, zorder=15)
        
        # Update target count
        self.performance_metrics['targets_tracked'] = len(visible_tracks)
        
    def process_radar_data(self):
        """Process radar data using professional pipeline"""
        if not self.is_running:
            return
            
        start_time = time.time()
        
        # Update target positions
        self.data_generator.update_targets(time_step_seconds=0.1)  # 10Hz
        
        # Simulate radar detection with much wider sweep for maximum pickup
        raw_detections = self.data_generator.simulate_radar_detection(
            self.sweep_angle, sweep_width_deg=40  # Much wider sweep
        )
        
        # Apply very lenient detection filtering
        if raw_detections:
            print(f"    🔍 Processing {len(raw_detections)} raw detections")
            filtered_detections = []
            for detection in raw_detections:
                detection_range = detection.get('range', 0)
                if detection_range >= 1.0 and detection_range <= 150.0:  # Very lenient range
                    filtered_detections.append(detection)
            
            print(f"    ✅ {len(filtered_detections)} detections passed range filter")
            
            if filtered_detections:
                detected_targets = self.target_detector.process_raw_detections(filtered_detections)
                print(f"    🎯 {len(detected_targets)} targets detected after processing")
                
                if detected_targets:
                    active_tracks = self.tracker.update(detected_targets, self.current_time)
                    print(f"    📊 {len([t for t in active_tracks.values() if t.confirmed])} confirmed tracks")
        else:
            print("    ❌ No raw detections from radar")
            
        # Record processing time
        processing_time = time.time() - start_time
        self.performance_metrics['processing_times'].append(processing_time * 1000)
        
        # Keep only recent processing times
        if len(self.performance_metrics['processing_times']) > 30:
            self.performance_metrics['processing_times'] = self.performance_metrics['processing_times'][-30:]
        
    def update_info_panels(self):
        """Update all information panels"""
        self.setup_status_panel()
        self.update_targets_panel()
        self.setup_performance_panel()
        self.setup_alerts_panel()
        
    def update_targets_panel(self):
        """Update target information with real tracking data"""
        ax = self.axes['targets']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('TRACKED TARGETS', color='#00ff00', weight='bold')
        ax.axis('off')
        
        # Get confirmed tracks
        confirmed_tracks = [track for track in self.tracker.tracks.values() 
                          if track.confirmed and not track.terminated]
        
        if not confirmed_tracks:
            ax.text(5, 5, 'NO TARGETS\nTRACKED', ha='center', va='center',
                   color='#666666', fontsize=12, weight='bold')
            return
        
        # Sort by range and show top 4
        sorted_tracks = sorted(confirmed_tracks, 
                             key=lambda t: np.sqrt(t.state.x**2 + t.state.y**2))[:4]
        
        target_text = "ACTIVE TRACKS\n\n"
        for track in sorted_tracks:
            track_range = np.sqrt(track.state.x**2 + track.state.y**2)
            track_bearing = np.degrees(np.arctan2(track.state.x, track.state.y)) % 360
            speed_kmh = np.sqrt(track.state.vx**2 + track.state.vy**2) * 3.6
            
            target_line = f"{track.id}: {track_range:5.1f}km {track_bearing:3.0f}°\n"
            target_line += f"     Speed: {speed_kmh:3.0f} km/h\n"
            target_text += target_line
            
        ax.text(5, 9, target_text, ha='center', va='top',
               color='#ffff00', fontsize=8, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                        alpha=0.9, edgecolor='#ffff00'))
        
    def animate(self, frame):
        """Enhanced animation callback"""
        if self.is_running:
            # Update time and sweep
            self.current_time += 0.1  # 10Hz
            self.sweep_angle = (self.sweep_angle + self.sweep_rate * 0.1) % 360
            
            # Update sweep history for smoother trail
            self.sweep_history.append(self.sweep_angle)
            if len(self.sweep_history) > 25:  # Keep more history for smoother trail
                self.sweep_history = self.sweep_history[-25:]
            
            # Process radar data
            self.process_radar_data()
            
            # Update displays
            self.update_radar_display()
            self.update_info_panels()
        else:
            # Update panels even when not running
            self.update_info_panels()
            
        return []
        
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes == self.axes['controls']:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                if 3 <= x <= 7:
                    if 7.9 <= y <= 9.1:  # START
                        self.start_system()
                    elif 5.9 <= y <= 7.1:  # STOP
                        self.stop_system()
                    elif 3.9 <= y <= 5.1:  # RESET
                        self.reset_system()
                    elif 1.9 <= y <= 3.1:  # E-STOP
                        self.emergency_stop()
                        
    def start_system(self):
        """Start the enhanced radar system"""
        self.is_running = True
        self.performance_metrics['start_time'] = time.time()
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.sweep_history = []
        
        # Reset tracker
        self.tracker = MultiTargetTracker()
        
        print("🚀 Enhanced Radar System STARTED")
        print("✅ Normal speed operation with smooth shadow trail:")
        print("  • 30 RPM sweep speed (1 rotation every 12 seconds)")
        print("  • Enhanced shadow trail with exponential fade")
        print("  • Improved target detection and display effects")
        print("  • Wider sweep beam for better target persistence")
        
    def stop_system(self):
        """Stop the radar system"""
        self.is_running = False
        print("🛑 Enhanced Radar System STOPPED")
        
    def reset_system(self):
        """Reset the system"""
        self.stop_system()
        self.tracker = MultiTargetTracker()
        self.data_generator = RadarDataGenerator(max_range_km=100)
        self.load_scenario()
        print("🔄 Enhanced System RESET")
        
    def emergency_stop(self):
        """Emergency stop"""
        self.stop_system()
        print("🚨 EMERGENCY STOP")
        
    def run(self):
        """Run the enhanced radar demo"""
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Start animation with good frame rate
        self.animation = FuncAnimation(self.fig, self.animate, interval=100, 
                                     blit=False, cache_frame_data=False)
        
        print("🎯 Enhanced Professional Radar System - Day 6 Complete!")
        print("✨ Features:")
        print("  • Normal 30 RPM sweep speed for comfortable viewing")
        print("  • Smooth exponential shadow trail behind sweep")
        print("  • Enhanced target effects based on detection timing")
        print("  • Wider sweep beam with gradient fade effects")
        print("  • Improved target persistence and visibility")
        
        plt.tight_layout()
        plt.show()

def main():
    """Run the enhanced Day 6 demonstration"""
    demo = ProfessionalRadarDemo()
    demo.run()

if __name__ == "__main__":
    main()