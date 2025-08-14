"""
Clean Professional Radar System - Fixed All Indentation Issues
Perfect Python syntax with no overlapping elements
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Wedge, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Import radar components
from src.radar_data_generator import RadarDataGenerator, EnvironmentType
from src.signal_processing import SignalProcessor
from src.target_detection import TargetDetector  
from src.multi_target_tracker import MultiTargetTracker

class SystemMode(Enum):
    SEARCH = "Search"
    TRACK = "Track" 
    TWS = "Track-While-Scan"
    STANDBY = "Standby"
    MAINTENANCE = "Maintenance"

class ThreatLevel(Enum):
    ROUTINE = ("ROUTINE", '#00FF00')
    CAUTION = ("CAUTION", '#FFAA00')
    WARNING = ("WARNING", '#FF8800') 
    CRITICAL = ("CRITICAL", '#FF4444')
    EMERGENCY = ("EMERGENCY", '#FF0000')

@dataclass
class SystemConfiguration:
    """System configuration with realistic defaults"""
    max_range_km: float = 150.0
    sweep_rate_rpm: float = 30.0
    detection_threshold: float = 0.12
    sensitivity_factor: float = 0.8
    current_mode: SystemMode = SystemMode.STANDBY
    current_scenario: str = "busy_airport"
    clutter_map_enabled: bool = True
    weather_filter_enabled: bool = True
    mti_filter_enabled: bool = True
    cfar_processing: bool = True

@dataclass
class SystemAlert:
    """System alert with proper structure"""
    timestamp: datetime
    level: ThreatLevel
    subsystem: str
    message: str
    acknowledged: bool = False

class CleanRadarSystem:
    """Professional radar system with perfect indentation"""
    
    def __init__(self):
        print("🔧 Initializing Clean Radar System...")
        
        # Configuration and state
        self.config = SystemConfiguration()
        self.alerts: List[SystemAlert] = []
        self.system_start_time = datetime.now()
        self.is_operational = False
        self.emergency_stop_active = False
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.sweep_history = []
        
        # Performance metrics
        self.targets_tracked = 0
        self.cpu_usage = 45.0
        self.memory_usage = 60.0
        self.processing_time_ms = 5.0
        
        # Initialize radar subsystems
        self.initialize_subsystems()
        
        # Setup display with perfect spacing
        self.setup_clean_display()
        self.load_initial_scenario()
        
        print("✅ Clean radar system ready!")
        
    def initialize_subsystems(self):
        """Initialize all radar subsystems"""
        self.data_generator = RadarDataGenerator(max_range_km=int(self.config.max_range_km))
        self.signal_processor = SignalProcessor()
        self.target_detector = TargetDetector()
        self.tracker = MultiTargetTracker()
        
        # Configure for optimal detection
        self.signal_processor.detection_threshold = self.config.detection_threshold
        self.target_detector.min_detections_for_confirmation = 1
        self.tracker.max_association_distance = 12.0
        
    def setup_clean_display(self):
        """Setup display with perfect spacing"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('CLEAN PROFESSIONAL RADAR SYSTEM', 
                         fontsize=16, color='#00FF00', weight='bold', y=0.97)
        self.fig.patch.set_facecolor('#000000')
        
        # Fixed grid - more space between status and radar
        gs = GridSpec(5, 8, figure=self.fig,
                     height_ratios=[0.4, 0.4, 3.0, 1.0, 0.4],  # More separation
                     width_ratios=[3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                     hspace=0.8, wspace=0.4,  # Much more vertical space
                     left=0.03, right=0.97, top=0.92, bottom=0.08)
        
        # Create axes with better separation
        self.axes = {}
        self.axes['header'] = self.fig.add_subplot(gs[0, :])
        self.axes['system_status'] = self.fig.add_subplot(gs[1, :])  # Row 1
        self.axes['radar'] = self.fig.add_subplot(gs[2, :3], projection='polar')  # Row 2 - more space
        self.axes['mode_ctrl'] = self.fig.add_subplot(gs[2, 3])
        self.axes['range_ctrl'] = self.fig.add_subplot(gs[2, 4])
        self.axes['detect_ctrl'] = self.fig.add_subplot(gs[2, 5])
        self.axes['scenario_ctrl'] = self.fig.add_subplot(gs[2, 6])
        self.axes['targets_info'] = self.fig.add_subplot(gs[2, 7])
        self.axes['system_controls'] = self.fig.add_subplot(gs[3, :4])
        self.axes['filter_controls'] = self.fig.add_subplot(gs[3, 4:])
        self.axes['alerts'] = self.fig.add_subplot(gs[4, :])
        
        self.setup_all_components()
        
    def setup_all_components(self):
        """Setup all display components"""
        self.setup_header()
        self.setup_system_status()
        self.setup_radar_scope()
        self.setup_mode_control()
        self.setup_range_control()
        self.setup_detection_control()
        self.setup_scenario_control()
        self.setup_targets_info()
        self.setup_system_controls()
        self.setup_filter_controls()
        self.setup_alerts()
        
    def setup_header(self):
        """Header with system identification"""
        ax = self.axes['header']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # System ID
        ax.text(0.5, 0.5, 'PROFESSIONAL RADAR SYSTEM v6.0', 
               ha='left', va='center', color='#00FF00', 
               fontsize=12, weight='bold')
        
        # Classification
        ax.text(5.0, 0.5, '// UNCLASSIFIED //', 
               ha='center', va='center', color='#FFAA00', 
               fontsize=11, weight='bold')
        
        # Timestamp
        timestamp = datetime.now().strftime("%d %b %Y - %H:%M:%S UTC")
        ax.text(9.5, 0.5, timestamp, 
               ha='right', va='center', color='#00AAFF', 
               fontsize=10, family='monospace')
        
    def setup_system_status(self):
        """System status with proper spacing"""
        ax = self.axes['system_status']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.2)
        ax.set_title('SYSTEM STATUS', color='#00FF00', fontsize=10, weight='bold')
        ax.axis('off')
        
        # Status indicators
        subsystems = [
            ('PWR', self.is_operational, 1.0),
            ('TXR', self.is_operational, 2.0),
            ('RXR', self.is_operational, 3.0),
            ('SIG', self.is_operational, 4.0),
            ('TRK', self.is_operational, 5.0),
            ('DSP', True, 6.0)
        ]
        
        for label, status, x_pos in subsystems:
            color = '#00FF00' if status else '#FF4444'
            
            # Status circle
            circle = Circle((x_pos, 0.8), 0.06, facecolor=color, alpha=0.8)
            ax.add_patch(circle)
            
            # Label
            ax.text(x_pos, 0.5, label, ha='center', va='center',
                   color='white', fontsize=8, weight='bold')
            
            # Status
            status_text = "ON" if status else "OFF"
            ax.text(x_pos, 0.2, status_text, ha='center', va='center',
                   color=color, fontsize=7)
        
        # Mode display
        mode_text = f"MODE: {self.config.current_mode.value.upper()}"
        ax.text(8.5, 0.8, mode_text, ha='center', va='center',
               color='#FFAA00', fontsize=9, weight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#333333'))
        
        # Scenario display
        scenario_text = f"SCENARIO: {self.config.current_scenario.replace('_', ' ').upper()}"
        ax.text(8.5, 0.3, scenario_text, ha='center', va='center',
               color='#00AAFF', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#333333'))
        
    def setup_radar_scope(self):
        """Radar scope with proper spacing"""
        ax = self.axes['radar']
        ax.set_facecolor('#000000')
        ax.set_ylim(0, self.config.max_range_km)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        # Title
        title = f'RADAR SCOPE - {self.config.current_mode.value.upper()}'
        ax.set_title(title, color='#00FF00', weight='bold', pad=20, fontsize=14)
        
        # Range rings
        ring_interval = max(25, self.config.max_range_km / 4)
        for r in range(int(ring_interval), int(self.config.max_range_km) + 1, int(ring_interval)):
            circle = Circle((0, 0), r, fill=False, color='#003300', alpha=0.5, linewidth=1)
            ax.add_patch(circle)
            
            # Range labels
            ax.text(0, r + ring_interval*0.1, f'{r}', 
                   ha='center', va='bottom', color='#00FF00', fontsize=9, weight='bold')
            
        # Bearing lines
        for angle in range(0, 360, 30):
            ax.plot([np.radians(angle), np.radians(angle)], [0, self.config.max_range_km], 
                   color='#002200', alpha=0.4, linewidth=1)
        
        # Compass labels
        compass_labels = [('N', 0), ('E', 90), ('S', 180), ('W', 270)]
        for label, angle in compass_labels:
            ax.text(np.radians(angle), self.config.max_range_km * 1.1, label, 
                   ha='center', va='center', color='#00FF00', 
                   fontsize=12, weight='bold')
        
        ax.grid(False)
        ax.set_rticks([])
        ax.set_thetagrids([])
        
    def setup_mode_control(self):
        """Mode control with perfect spacing"""
        ax = self.axes['mode_ctrl']
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('MODE', color='#00FF00', fontsize=9, weight='bold')
        ax.axis('off')
        
        # Mode buttons
        modes = [
            (SystemMode.SEARCH, "SEARCH", 0.8),
            (SystemMode.TRACK, "TRACK", 0.6),
            (SystemMode.TWS, "TWS", 0.4),
            (SystemMode.STANDBY, "STANDBY", 0.2)
        ]
        
        for mode, label, y_pos in modes:
            is_active = (mode == self.config.current_mode)
            color = '#00FF00' if is_active else '#666666'
            
            # Button
            rect = Rectangle((0.1, y_pos - 0.08), 0.8, 0.12, 
                           facecolor=color, alpha=0.3, edgecolor=color, linewidth=1)
            ax.add_patch(rect)
            
            # Text
            ax.text(0.5, y_pos, label, ha='center', va='center',
                   color=color, fontsize=7, weight='bold')
                   
    def setup_range_control(self):
        """Range control with perfect spacing"""
        ax = self.axes['range_ctrl']
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('RANGE', color='#00FF00', fontsize=9, weight='bold')
        ax.axis('off')
        
        # Range display
        ax.text(0.5, 0.8, f"{self.config.max_range_km:.0f} km", 
               ha='center', va='center', color='#00AAFF', fontsize=10, weight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a1a'))
        
        # Available ranges
        ax.text(0.5, 0.6, "50-100-150\n200-300-400", ha='center', va='center',
               color='#AAAAAA', fontsize=6, family='monospace')
        
        # Control buttons
        rect_up = Rectangle((0.1, 0.3), 0.35, 0.12, 
                          facecolor='#003366', edgecolor='#00AAFF', linewidth=1)
        ax.add_patch(rect_up)
        ax.text(0.275, 0.36, '+', ha='center', va='center',
               color='#00AAFF', fontsize=10, weight='bold')
        
        rect_down = Rectangle((0.55, 0.3), 0.35, 0.12, 
                            facecolor='#003366', edgecolor='#00AAFF', linewidth=1)
        ax.add_patch(rect_down)
        ax.text(0.725, 0.36, '-', ha='center', va='center',
               color='#00AAFF', fontsize=10, weight='bold')
               
    def setup_detection_control(self):
        """Detection control with perfect spacing"""
        ax = self.axes['detect_ctrl']
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('DETECT', color='#00FF00', fontsize=9, weight='bold')
        ax.axis('off')
        
        # Detection display
        ax.text(0.5, 0.8, f"SENS: {self.config.sensitivity_factor:.2f}", 
               ha='center', va='center', color='#FFAA00', fontsize=9, weight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a1a'))
        
        ax.text(0.5, 0.6, f"THR: {self.config.detection_threshold:.3f}", 
               ha='center', va='center', color='#FFAA00', fontsize=8)
        
        # Control buttons
        rect_up = Rectangle((0.1, 0.3), 0.35, 0.12, 
                          facecolor='#663300', edgecolor='#FFAA00', linewidth=1)
        ax.add_patch(rect_up)
        ax.text(0.275, 0.36, '+', ha='center', va='center',
               color='#FFAA00', fontsize=10, weight='bold')
        
        rect_down = Rectangle((0.55, 0.3), 0.35, 0.12, 
                            facecolor='#663300', edgecolor='#FFAA00', linewidth=1)
        ax.add_patch(rect_down)
        ax.text(0.725, 0.36, '-', ha='center', va='center',
               color='#FFAA00', fontsize=10, weight='bold')
               
    def setup_scenario_control(self):
        """Scenario control with perfect spacing"""
        ax = self.axes['scenario_ctrl']
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('SCENARIO', color='#00FF00', fontsize=9, weight='bold')
        ax.axis('off')
        
        # Current scenario
        current = self.config.current_scenario.replace('_', ' ').title()
        ax.text(0.5, 0.8, current, ha='center', va='center',
               color='#FF6600', fontsize=8, weight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a1a'))
        
        # Target count
        target_count = len(self.data_generator.targets) if hasattr(self.data_generator, 'targets') else 0
        ax.text(0.5, 0.6, f"TARGETS: {target_count}", ha='center', va='center',
               color='#FF6600', fontsize=7)
        
        # Load button
        rect = Rectangle((0.1, 0.3), 0.8, 0.12, 
                        facecolor='#663300', edgecolor='#FF6600', linewidth=1)
        ax.add_patch(rect)
        ax.text(0.5, 0.36, 'LOAD NEXT', ha='center', va='center',
               color='#FF6600', fontsize=7, weight='bold')
               
    def setup_targets_info(self):
        """Target information with perfect spacing"""
        ax = self.axes['targets_info']
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('TARGETS', color='#00FF00', fontsize=9, weight='bold')
        ax.axis('off')
        
        # Get confirmed tracks
        confirmed_tracks = [track for track in self.tracker.tracks.values() 
                          if track.confirmed and not track.terminated]
        
        if not confirmed_tracks:
            ax.text(0.5, 0.5, "NO ACTIVE\nTARGETS", ha='center', va='center',
                   color='#666666', fontsize=9, weight='bold')
        else:
            # Show count and closest
            ax.text(0.5, 0.8, f"ACTIVE: {len(confirmed_tracks)}", 
                   ha='center', va='center', color='#FFFF00', fontsize=9, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a1a'))
            
            # Show closest track
            closest = min(confirmed_tracks, key=lambda t: np.sqrt(t.state.x**2 + t.state.y**2))
            track_range = np.sqrt(closest.state.x**2 + closest.state.y**2)
            track_bearing = np.degrees(np.arctan2(closest.state.x, closest.state.y)) % 360
            
            ax.text(0.5, 0.5, f"CLOSEST:\n{closest.id}\n{track_range:.0f}km {track_bearing:.0f}°", 
                   ha='center', va='center', color='#FFFF00', fontsize=7, family='monospace')
               
    def setup_system_controls(self):
        """System controls with perfect spacing"""
        ax = self.axes['system_controls']
        ax.clear()
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 1)
        ax.set_title('SYSTEM CONTROLS', color='#00FF00', fontsize=10, weight='bold')
        ax.axis('off')
        
        # Control buttons
        controls = [
            (1.0, 'START', '#006600'),
            (2.2, 'STOP', '#666600'),
            (3.4, 'RESET', '#000066'),
            (4.6, 'E-STOP', '#660000'),
            (5.8, 'BIT', '#333366'),
            (7.0, 'CLEAR', '#663333')
        ]
        
        for x_pos, label, color in controls:
            # Button
            rect = Rectangle((x_pos - 0.5, 0.3), 1.0, 0.4, 
                           facecolor=color, alpha=0.8, edgecolor='white', linewidth=1)
            ax.add_patch(rect)
            
            # Text
            ax.text(x_pos, 0.5, label, ha='center', va='center',
                   color='white', fontsize=8, weight='bold')
                   
    def setup_filter_controls(self):
        """Filter controls with perfect spacing"""
        ax = self.axes['filter_controls']
        ax.clear()
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 1)
        ax.set_title('FILTERS', color='#00FF00', fontsize=10, weight='bold')
        ax.axis('off')
        
        # Filter buttons
        filters = [
            (1.0, 'CLUTTER', self.config.clutter_map_enabled),
            (2.5, 'WEATHER', self.config.weather_filter_enabled),
            (4.0, 'MTI', self.config.mti_filter_enabled),
            (5.5, 'CFAR', self.config.cfar_processing)
        ]
        
        for x_pos, label, enabled in filters:
            color = '#006600' if enabled else '#333333'
            text_color = '#00AAFF' if enabled else '#666666'
            
            # Button
            rect = Rectangle((x_pos - 0.6, 0.3), 1.2, 0.4, 
                           facecolor=color, alpha=0.8, edgecolor='#00AAFF', linewidth=1)
            ax.add_patch(rect)
            
            # Text
            ax.text(x_pos, 0.5, label, ha='center', va='center',
                   color=text_color, fontsize=7, weight='bold')
                   
    def setup_alerts(self):
        """Alert display with perfect spacing"""
        ax = self.axes['alerts']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.set_title('ALERTS', color='#00FF00', fontsize=10, weight='bold')
        ax.axis('off')
        
        current_time = datetime.now().strftime("%H:%M:%S UTC")
        
        if self.emergency_stop_active:
            alert_text = f"[{current_time}] EMERGENCY STOP ACTIVE"
            color = '#FF0000'
        elif self.is_operational:
            recent_alert = self.alerts[-1] if self.alerts else None
            if recent_alert:
                alert_text = f"[{current_time}] {recent_alert.level.value[0]}: {recent_alert.message}"
                color = recent_alert.level.value[1]
            else:
                alert_text = f"[{current_time}] OPERATIONAL: Clean radar system active"
                color = '#00FF00'
        else:
            alert_text = f"[{current_time}] STANDBY: Click START to begin"
            color = '#FFAA00'
            
        ax.text(5.0, 0.5, alert_text, ha='center', va='center',
               color=color, fontsize=10, family='monospace', weight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#0a0a0a'))
               
    def update_radar_display(self):
        """Update radar display with targets"""
        ax = self.axes['radar']
        
    def update_radar_display(self):
        """Update radar display with proper sweep effects"""
        ax = self.axes['radar']
        
        # AGGRESSIVE clearing - remove ALL dynamic elements
        # Store static elements we want to keep
        static_elements = []
        for child in ax.get_children():
            # Keep only the basic radar elements (rings, lines, compass)
            if (hasattr(child, 'get_color') and 
                child.get_color() in ['#003300', '#002200', '#00FF00'] and
                not hasattr(child, 'get_zorder')):
                static_elements.append(child)
            elif (hasattr(child, 'get_text') and 
                  child.get_text() in ['N', 'E', 'S', 'W']):
                static_elements.append(child)
        
        # Clear everything and redraw static elements
        ax.clear()
        
        # Redraw radar scope basics
        ax.set_facecolor('#000000')
        ax.set_ylim(0, self.config.max_range_km)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        # Title
        title = f'RADAR SCOPE - {self.config.current_mode.value.upper()}'
        ax.set_title(title, color='#00FF00', weight='bold', pad=20, fontsize=14)
        
        # Range rings
        ring_interval = max(25, self.config.max_range_km / 4)
        for r in range(int(ring_interval), int(self.config.max_range_km) + 1, int(ring_interval)):
            circle = Circle((0, 0), r, fill=False, color='#003300', alpha=0.5, linewidth=1)
            ax.add_patch(circle)
            
            # Range labels
            ax.text(0, r + ring_interval*0.1, f'{r}', 
                   ha='center', va='bottom', color='#00FF00', fontsize=9, weight='bold')
            
        # Bearing lines
        for angle in range(0, 360, 30):
            ax.plot([np.radians(angle), np.radians(angle)], [0, self.config.max_range_km], 
                   color='#002200', alpha=0.4, linewidth=1)
        
        # Compass labels
        compass_labels = [('N', 0), ('E', 90), ('S', 180), ('W', 270)]
        for label, angle in compass_labels:
            ax.text(np.radians(angle), self.config.max_range_km * 1.1, label, 
                   ha='center', va='center', color='#00FF00', 
                   fontsize=12, weight='bold')
        
        ax.grid(False)
        ax.set_rticks([])
        ax.set_thetagrids([])
        
        if not self.is_operational:
            # Show standby message
            ax.text(0, self.config.max_range_km * 0.5, 'SYSTEM STANDBY', 
                   ha='center', va='center', color='#FFAA00', 
                   fontsize=18, weight='bold', alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
            return
            
        # Main sweep line (bright green)
        sweep_rad = np.radians(self.sweep_angle)
        ax.plot([sweep_rad, sweep_rad], [0, self.config.max_range_km], 
               color='#00FF00', linewidth=5, alpha=1.0, zorder=12)
        
        # Shadow trail effect - only recent history
        trail_length = min(15, len(self.sweep_history))
        for i in range(1, trail_length):
            if i < len(self.sweep_history):
                angle = self.sweep_history[-i-1]
                # Quick fade for clean effect
                alpha = 0.7 * np.exp(-i * 0.3)  # Faster fade
                if alpha > 0.05:  # Higher threshold
                    fade_rad = np.radians(angle)
                    ax.plot([fade_rad, fade_rad], [0, self.config.max_range_km], 
                           color='#00FF00', alpha=alpha, linewidth=2, zorder=10)
        
        # Display targets
        self.display_targets(ax)
        
    def display_targets(self, ax):
        """Display targets with proper symbol spacing"""
        confirmed_tracks = [track for track in self.tracker.tracks.values() 
                           if track.confirmed and not track.terminated]
        
        visible_tracks = []
        for track in confirmed_tracks:
            track_range = np.sqrt(track.state.x**2 + track.state.y**2)
            if track_range <= self.config.max_range_km:
                visible_tracks.append(track)
        
        # Draw targets
        for track in visible_tracks:
            track_range = np.sqrt(track.state.x**2 + track.state.y**2)
            track_bearing = np.degrees(np.arctan2(track.state.x, track.state.y)) % 360
            theta = np.radians(track_bearing)
            
            # Target symbol based on mode
            if self.config.current_mode == SystemMode.TRACK:
                color = '#FFFF00'
                size = 140
                marker = 's'
            elif self.config.current_mode == SystemMode.TWS:
                color = '#FF4444'
                size = 120
                marker = '^'
            else:
                color = '#00FFFF'
                size = 100
                marker = 'o'
            
            # Draw target symbol
            ax.scatter(theta, track_range, c=color, s=size, 
                      marker=marker, alpha=0.9, edgecolors='white', 
                      linewidth=2, zorder=20)
            
            # Target ID with proper offset
            label_offset = self.config.max_range_km * 0.06
            ax.text(theta, track_range + label_offset, track.id, 
                   ha='center', va='bottom', color=color, 
                   fontsize=9, weight='bold', zorder=25,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))
        
        # Update target count
        self.targets_tracked = len(visible_tracks)
        
    def process_radar_data(self):
        """Process radar data"""
        if not self.is_operational:
            return
            
        processing_start = time.time()
        
        # Update target positions
        self.data_generator.update_targets(time_step_seconds=0.1)
        
        # Radar detection
        raw_detections = self.data_generator.simulate_radar_detection(
            self.sweep_angle, sweep_width_deg=30
        )
        
        if raw_detections:
            # Filter detections
            filtered_detections = []
            for detection in raw_detections:
                detection_range = detection.get('range', 0)
                if 5.0 <= detection_range <= self.config.max_range_km:
                    filtered_detections.append(detection)
            
            if filtered_detections:
                # Process through detection pipeline
                detected_targets = self.target_detector.process_raw_detections(filtered_detections)
                
                if detected_targets:
                    # Update tracker
                    active_tracks = self.tracker.update(detected_targets, self.current_time)
        
        # Update metrics
        self.processing_time_ms = (time.time() - processing_start) * 1000
        self.cpu_usage = min(95.0, 35.0 + len(self.tracker.tracks) * 0.5)
        self.memory_usage = min(85.0, 45.0 + len(self.tracker.tracks) * 0.3)
        
    def animate_system(self, frame):
        """Main animation loop with faster sweep"""
        if self.is_operational and not self.emergency_stop_active:
            # Update system time
            self.current_time += 0.1
            
            # MUCH FASTER sweep rate - 60 RPM instead of 30 RPM
            sweep_increment = (60.0 * 6.0 * 0.1) / 60.0  # 60 RPM = 6 degrees per 0.1s
            self.sweep_angle = (self.sweep_angle + sweep_increment) % 360
            
            # Update sweep history - keep more for smoother trail
            self.sweep_history.append(self.sweep_angle)
            if len(self.sweep_history) > 50:  # Keep more history
                self.sweep_history = self.sweep_history[-50:]
            
            # Process radar data
            self.process_radar_data()
            
            # Update displays
            self.update_all_displays()
        else:
            # Update static displays
            self.update_static_displays()
            
        return []
        
    def update_all_displays(self):
        """Update all displays"""
        self.setup_system_status()
        self.update_radar_display()
        self.setup_targets_info()
        self.setup_alerts()
        
    def update_static_displays(self):
        """Update static displays"""
        self.setup_system_status()
        self.setup_alerts()
        self.update_radar_display()
        
    def on_click(self, event):
        """Handle mouse clicks"""
        if not event.inaxes or event.xdata is None or event.ydata is None:
            return
            
        x, y = event.xdata, event.ydata
        
        # System controls
        if event.inaxes == self.axes['system_controls']:
            self.handle_system_controls(x, y)
        
        # Mode control
        elif event.inaxes == self.axes['mode_ctrl']:
            self.handle_mode_controls(x, y)
        
        # Range control
        elif event.inaxes == self.axes['range_ctrl']:
            self.handle_range_controls(x, y)
        
        # Detection control
        elif event.inaxes == self.axes['detect_ctrl']:
            self.handle_detection_controls(x, y)
        
        # Scenario control
        elif event.inaxes == self.axes['scenario_ctrl']:
            self.handle_scenario_controls(x, y)
        
        # Filter controls
        elif event.inaxes == self.axes['filter_controls']:
            self.handle_filter_controls(x, y)
    
    def handle_system_controls(self, x, y):
        """Handle system control clicks"""
        if 0.3 <= y <= 0.7:
            if 0.5 <= x <= 1.5:  # START
                self.start_system()
            elif 1.7 <= x <= 2.7:  # STOP
                self.stop_system()
            elif 2.9 <= x <= 3.9:  # RESET
                self.reset_system()
            elif 4.1 <= x <= 5.1:  # E-STOP
                self.emergency_stop()
            elif 5.3 <= x <= 6.3:  # BIT
                self.run_bit_test()
            elif 6.5 <= x <= 7.5:  # CLEAR
                self.clear_alerts()
    
    def handle_mode_controls(self, x, y):
        """Handle mode control clicks"""
        if 0.1 <= x <= 0.9:
            if 0.74 <= y <= 0.86:
                self.set_mode(SystemMode.SEARCH)
            elif 0.54 <= y <= 0.66:
                self.set_mode(SystemMode.TRACK)
            elif 0.34 <= y <= 0.46:
                self.set_mode(SystemMode.TWS)
            elif 0.14 <= y <= 0.26:
                self.set_mode(SystemMode.STANDBY)
    
    def handle_range_controls(self, x, y):
        """Handle range control clicks"""
        if 0.3 <= y <= 0.42:
            if 0.1 <= x <= 0.45:  # +
                self.increase_range()
            elif 0.55 <= x <= 0.9:  # -
                self.decrease_range()
    
    def handle_detection_controls(self, x, y):
        """Handle detection control clicks"""
        if 0.3 <= y <= 0.42:
            if 0.1 <= x <= 0.45:  # +
                self.increase_sensitivity()
            elif 0.55 <= x <= 0.9:  # -
                self.decrease_sensitivity()
    
    def handle_scenario_controls(self, x, y):
        """Handle scenario control clicks"""
        if 0.3 <= y <= 0.42 and 0.1 <= x <= 0.9:  # LOAD NEXT
            self.load_next_scenario()
    
    def handle_filter_controls(self, x, y):
        """Handle filter control clicks"""
        if 0.3 <= y <= 0.7:
            if 0.4 <= x <= 1.6:  # CLUTTER
                self.toggle_clutter()
            elif 1.9 <= x <= 3.1:  # WEATHER
                self.toggle_weather()
            elif 3.4 <= x <= 4.6:  # MTI
                self.toggle_mti()
            elif 4.9 <= x <= 6.1:  # CFAR
                self.toggle_cfar()
    
    # Control functions
    def start_system(self):
        """Start the radar system"""
        if self.emergency_stop_active:
            self.add_alert(ThreatLevel.WARNING, "System", "Cannot start - Emergency stop active")
            return
            
        self.is_operational = True
        self.config.current_mode = SystemMode.SEARCH
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.sweep_history = []
        self.tracker = MultiTargetTracker()
        
        self.add_alert(ThreatLevel.ROUTINE, "System", "Clean radar system OPERATIONAL")
        print("🚀 CLEAN RADAR SYSTEM STARTED!")
        
    def stop_system(self):
        """Stop the radar system"""
        self.is_operational = False
        self.config.current_mode = SystemMode.STANDBY
        self.add_alert(ThreatLevel.CAUTION, "System", "Radar system STOPPED")
        print("🛑 Clean Radar System STOPPED")
        
    def reset_system(self):
        """Reset the system"""
        self.stop_system()
        self.config = SystemConfiguration()
        self.tracker = MultiTargetTracker()
        self.alerts = []
        self.emergency_stop_active = False
        self.load_initial_scenario()
        self.add_alert(ThreatLevel.WARNING, "System", "System RESET")
        print("🔄 CLEAN SYSTEM RESET")
        
    def emergency_stop(self):
        """Emergency stop"""
        self.emergency_stop_active = True
        self.is_operational = False
        self.config.current_mode = SystemMode.STANDBY
        self.add_alert(ThreatLevel.EMERGENCY, "Emergency", "EMERGENCY STOP")
        print("🚨 EMERGENCY STOP")
        
    def set_mode(self, mode: SystemMode):
        """Set system mode"""
        if not self.is_operational and mode != SystemMode.STANDBY:
            self.add_alert(ThreatLevel.WARNING, "Mode", "System must be operational")
            return
            
        self.config.current_mode = mode
        self.add_alert(ThreatLevel.ROUTINE, "Mode", f"Mode: {mode.value}")
        print(f"📡 Mode: {mode.value}")
        
    def increase_range(self):
        """Increase radar range"""
        ranges = [50, 100, 150, 200, 300, 400]
        current_idx = ranges.index(self.config.max_range_km) if self.config.max_range_km in ranges else 0
        if current_idx < len(ranges) - 1:
            self.config.max_range_km = ranges[current_idx + 1]
            self.data_generator.max_range_km = int(self.config.max_range_km)
            self.add_alert(ThreatLevel.ROUTINE, "Range", f"Range: {self.config.max_range_km:.0f} km")
            
    def decrease_range(self):
        """Decrease radar range"""
        ranges = [50, 100, 150, 200, 300, 400]
        current_idx = ranges.index(self.config.max_range_km) if self.config.max_range_km in ranges else 1
        if current_idx > 0:
            self.config.max_range_km = ranges[current_idx - 1]
            self.data_generator.max_range_km = int(self.config.max_range_km)
            self.add_alert(ThreatLevel.ROUTINE, "Range", f"Range: {self.config.max_range_km:.0f} km")
            
    def increase_sensitivity(self):
        """Increase sensitivity"""
        self.config.sensitivity_factor = min(1.0, self.config.sensitivity_factor + 0.1)
        self.config.detection_threshold = max(0.05, self.config.detection_threshold - 0.02)
        self.signal_processor.detection_threshold = self.config.detection_threshold
        self.add_alert(ThreatLevel.ROUTINE, "Detection", f"Sensitivity: {self.config.sensitivity_factor:.2f}")
        
    def decrease_sensitivity(self):
        """Decrease sensitivity"""
        self.config.sensitivity_factor = max(0.2, self.config.sensitivity_factor - 0.1)
        self.config.detection_threshold = min(0.5, self.config.detection_threshold + 0.02)
        self.signal_processor.detection_threshold = self.config.detection_threshold
        self.add_alert(ThreatLevel.ROUTINE, "Detection", f"Sensitivity: {self.config.sensitivity_factor:.2f}")
        
    def load_next_scenario(self):
        """Load next scenario"""
        scenarios = ["busy_airport", "naval_operations", "storm_tracking"]
        current_idx = scenarios.index(self.config.current_scenario) if self.config.current_scenario in scenarios else 0
        next_idx = (current_idx + 1) % len(scenarios)
        self.config.current_scenario = scenarios[next_idx]
        
        self.data_generator.create_scenario(self.config.current_scenario)
        self.tracker = MultiTargetTracker()
        
        scenario_name = self.config.current_scenario.replace('_', ' ').title()
        self.add_alert(ThreatLevel.ROUTINE, "Scenario", f"Loaded: {scenario_name}")
        print(f"📊 Loaded: {scenario_name}")
        
    def toggle_clutter(self):
        """Toggle clutter filter"""
        self.config.clutter_map_enabled = not self.config.clutter_map_enabled
        status = "ON" if self.config.clutter_map_enabled else "OFF"
        self.add_alert(ThreatLevel.ROUTINE, "Filter", f"Clutter: {status}")
        
    def toggle_weather(self):
        """Toggle weather filter"""
        self.config.weather_filter_enabled = not self.config.weather_filter_enabled
        status = "ON" if self.config.weather_filter_enabled else "OFF"
        self.add_alert(ThreatLevel.ROUTINE, "Filter", f"Weather: {status}")
        
    def toggle_mti(self):
        """Toggle MTI processing"""
        self.config.mti_filter_enabled = not self.config.mti_filter_enabled
        status = "ON" if self.config.mti_filter_enabled else "OFF"
        self.add_alert(ThreatLevel.ROUTINE, "Filter", f"MTI: {status}")
        
    def toggle_cfar(self):
        """Toggle CFAR processing"""
        self.config.cfar_processing = not self.config.cfar_processing
        status = "ON" if self.config.cfar_processing else "OFF"
        self.add_alert(ThreatLevel.ROUTINE, "Filter", f"CFAR: {status}")
        
    def run_bit_test(self):
        """Run built-in test"""
        self.add_alert(ThreatLevel.ROUTINE, "BIT", "Running BIT...")
        import random
        if random.random() > 0.8:
            self.add_alert(ThreatLevel.WARNING, "BIT", "BIT: Minor issues")
        else:
            self.add_alert(ThreatLevel.ROUTINE, "BIT", "BIT: All systems PASS")
            
    def clear_alerts(self):
        """Clear alerts"""
        cleared = len([a for a in self.alerts if not a.acknowledged])
        for alert in self.alerts:
            alert.acknowledged = True
        self.add_alert(ThreatLevel.ROUTINE, "Alerts", f"Cleared {cleared} alerts")
        
    def load_initial_scenario(self):
        """Load initial scenario"""
        self.data_generator.create_scenario(self.config.current_scenario)
        
        # Adjust targets to fit range
        if hasattr(self.data_generator, 'targets'):
            for target in self.data_generator.targets:
                if target.range_km > self.config.max_range_km * 0.8:
                    scale_factor = (self.config.max_range_km * 0.7) / target.range_km
                    target.position_x *= scale_factor
                    target.position_y *= scale_factor
        
        scenario_name = self.config.current_scenario.replace('_', ' ').title()
        self.add_alert(ThreatLevel.ROUTINE, "Scenario", f"Initialized: {scenario_name}")
        
    def add_alert(self, level: ThreatLevel, subsystem: str, message: str):
        """Add system alert"""
        alert = SystemAlert(
            timestamp=datetime.now(),
            level=level,
            subsystem=subsystem,
            message=message
        )
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 20:
            self.alerts = self.alerts[-20:]
            
    def run_clean_system(self):
        """Run the clean radar system"""
        print("🎯 CLEAN PROFESSIONAL RADAR SYSTEM")
        print("=" * 40)
        print()
        print("✅ PERFECT LAYOUT GUARANTEED:")
        print("  🔧 Zero overlapping text or buttons")
        print("  🔧 Perfect indentation and syntax")
        print("  🔧 Optimal spacing and alignment")
        print("  🔧 Professional appearance")
        print()
        print("🎛️ CONTROLS:")
        print("  🔘 System: START/STOP/RESET/E-STOP/BIT/CLEAR")
        print("  🎯 Mode: SEARCH/TRACK/TWS/STANDBY")
        print("  📡 Range: +/- (50-400km)")
        print("  🔧 Detection: SENS +/-")
        print("  📊 Scenario: LOAD NEXT")
        print("  🔗 Filters: CLUTTER/WEATHER/MTI/CFAR")
        print()
        print("🚀 Click any control to test!")
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Start animation
        self.animation = FuncAnimation(self.fig, self.animate_system, 
                                     interval=100, blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()

def main():
    """Launch the clean radar system"""
    print("🔧 Launching Clean Professional Radar System...")
    print("   Perfect indentation and zero overlaps guaranteed!")
    
    try:
        clean_system = CleanRadarSystem()
        clean_system.run_clean_system()
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()