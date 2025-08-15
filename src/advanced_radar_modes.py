"""
Advanced Radar Modes System - Day 7 Task 1
===========================================
Implements professional radar operating modes:
- Search Mode: Wide-area scanning for new target detection
- Track Mode: Focused tracking of confirmed targets
- Track-While-Scan: Hybrid mode for operational flexibility

Features:
• Mode-specific sweep patterns and timing
• Adaptive detection parameters per mode
• Professional mode switching logic
• Optimized performance for each operating mode
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Wedge, FancyBboxPatch
from matplotlib.widgets import Button
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

# Import radar components
try:
    from src.radar_data_generator import RadarDataGenerator
    from src.signal_processing import SignalProcessor
    from src.target_detection import TargetDetector
    from src.multi_target_tracker import MultiTargetTracker
    print("✅ All radar components imported successfully")
except ImportError as e:
    print(f"⚠️  Warning: Some components not found: {e}")

class RadarMode(Enum):
    """Professional radar operating modes"""
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    TRACK_WHILE_SCAN = "TWS"
    STANDBY = "STANDBY"

@dataclass
class ModeConfiguration:
    """Configuration parameters for each radar mode"""
    sweep_rate_rpm: float           # Antenna rotation speed
    sweep_width_deg: float          # Beam width for detection
    detection_threshold: float      # Signal strength threshold
    max_range_km: float            # Maximum detection range
    dwell_time_ms: float           # Time spent on each bearing
    priority_sectors: List[Tuple[float, float]] = field(default_factory=list)  # (start_deg, end_deg)
    track_update_rate_hz: float = 1.0    # Rate for track updates
    
class RadarModeManager:
    """
    Manages radar operating modes and their specific behaviors
    
    This class implements the core logic for different radar modes,
    each optimized for specific operational requirements.
    """
    
    def __init__(self):
        # Define mode configurations
        self.mode_configs = {
            RadarMode.SEARCH: ModeConfiguration(
                sweep_rate_rpm=30.0,        # Standard search speed
                sweep_width_deg=30.0,       # Wide beam for coverage
                detection_threshold=0.12,   # Sensitive for new targets
                max_range_km=200.0,         # Maximum range
                dwell_time_ms=50.0,         # Quick scan
                track_update_rate_hz=0.5    # Slow track updates
            ),
            
            RadarMode.TRACK: ModeConfiguration(
                sweep_rate_rpm=60.0,        # Fast for accuracy
                sweep_width_deg=10.0,       # Narrow beam for precision
                detection_threshold=0.08,   # Very sensitive
                max_range_km=150.0,         # Focused range
                dwell_time_ms=100.0,        # Longer dwell for accuracy
                track_update_rate_hz=5.0    # Fast track updates
            ),
            
            RadarMode.TRACK_WHILE_SCAN: ModeConfiguration(
                sweep_rate_rpm=45.0,        # Balanced speed
                sweep_width_deg=20.0,       # Medium beam
                detection_threshold=0.10,   # Balanced sensitivity
                max_range_km=175.0,         # Extended range
                dwell_time_ms=75.0,         # Balanced dwell
                track_update_rate_hz=2.0    # Balanced updates
            ),
            
            RadarMode.STANDBY: ModeConfiguration(
                sweep_rate_rpm=0.0,         # No rotation
                sweep_width_deg=0.0,        # No beam
                detection_threshold=1.0,    # No detection
                max_range_km=0.0,           # No range
                dwell_time_ms=0.0,          # No dwell
                track_update_rate_hz=0.0    # No updates
            )
        }
        
        self.current_mode = RadarMode.SEARCH
        self.mode_start_time = time.time()
        self.sector_priorities = []
        
    def set_mode(self, new_mode: RadarMode) -> bool:
        """
        Switch radar to new operating mode
        
        Args:
            new_mode: Target radar mode
            
        Returns:
            True if mode switch successful
        """
        if new_mode == self.current_mode:
            return True
            
        old_mode = self.current_mode
        self.current_mode = new_mode
        self.mode_start_time = time.time()
        
        print(f"🔄 Mode change: {old_mode.value} → {new_mode.value}")
        self._log_mode_characteristics()
        
        return True
    
    def _log_mode_characteristics(self):
        """Log current mode characteristics"""
        config = self.get_current_config()
        print(f"   • Sweep rate: {config.sweep_rate_rpm} RPM")
        print(f"   • Beam width: {config.sweep_width_deg}°")
        print(f"   • Range: {config.max_range_km} km")
        print(f"   • Threshold: {config.detection_threshold:.3f}")
        
    def get_current_config(self) -> ModeConfiguration:
        """Get configuration for current mode"""
        return self.mode_configs[self.current_mode]
    
    def get_sweep_parameters(self, current_time: float) -> Dict:
        """
        Get sweep parameters based on current mode and time
        
        Args:
            current_time: Current system time
            
        Returns:
            Dictionary with sweep parameters
        """
        config = self.get_current_config()
        
        if self.current_mode == RadarMode.STANDBY:
            return {
                'sweep_rate': 0.0,
                'sweep_width': 0.0,
                'dwell_multiplier': 0.0,
                'range_limit': 0.0
            }
        
        # Calculate adaptive parameters
        mode_duration = current_time - self.mode_start_time
        
        # Track mode focuses on known target areas
        if self.current_mode == RadarMode.TRACK and self.sector_priorities:
            dwell_multiplier = 2.0  # Extra time on priority sectors
        else:
            dwell_multiplier = 1.0
            
        return {
            'sweep_rate': config.sweep_rate_rpm * 6.0,  # Convert to degrees/second
            'sweep_width': config.sweep_width_deg,
            'dwell_multiplier': dwell_multiplier,
            'range_limit': config.max_range_km,
            'detection_threshold': config.detection_threshold
        }
    
    def add_priority_sector(self, start_bearing: float, end_bearing: float):
        """Add priority sector for focused scanning"""
        self.sector_priorities.append((start_bearing % 360, end_bearing % 360))
        print(f"📍 Priority sector added: {start_bearing:.1f}° - {end_bearing:.1f}°")
    
    def is_priority_sector(self, bearing: float) -> bool:
        """Check if bearing is in a priority sector"""
        bearing = bearing % 360
        for start, end in self.sector_priorities:
            if start <= end:
                if start <= bearing <= end:
                    return True
            else:  # Wraps around 360°
                if bearing >= start or bearing <= end:
                    return True
        return False
    
    def get_mode_display_properties(self) -> Dict:
        """Get display properties for current mode"""
        mode_colors = {
            RadarMode.SEARCH: {
                'sweep_color': '#00ff00',
                'beam_alpha': 0.2,
                'trail_alpha': 0.1,
                'target_color': '#ffff00',
                'mode_text': 'SEARCH'
            },
            RadarMode.TRACK: {
                'sweep_color': '#ff4400',
                'beam_alpha': 0.4,
                'trail_alpha': 0.3,
                'target_color': '#ff0000',
                'mode_text': 'TRACK'
            },
            RadarMode.TRACK_WHILE_SCAN: {
                'sweep_color': '#00ffff',
                'beam_alpha': 0.3,
                'trail_alpha': 0.2,
                'target_color': '#ff8800',
                'mode_text': 'TWS'
            },
            RadarMode.STANDBY: {
                'sweep_color': '#404040',
                'beam_alpha': 0.1,
                'trail_alpha': 0.05,
                'target_color': '#808080',
                'mode_text': 'STANDBY'
            }
        }
        
        return mode_colors[self.current_mode]

class AdvancedRadarSystem:
    """
    Advanced Radar System with Multiple Operating Modes
    
    This system demonstrates professional radar capabilities with
    multiple operating modes suitable for different tactical situations.
    """
    
    def __init__(self):
        print("🚀 Initializing Advanced Radar System...")
        
        # Core components
        self.data_generator = RadarDataGenerator(max_range_km=200)
        self.signal_processor = SignalProcessor()
        self.target_detector = TargetDetector()
        self.tracker = MultiTargetTracker()
        self.mode_manager = RadarModeManager()
        
        # System state
        self.is_running = False
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.sweep_history = []
        self.target_trails = {}
        
        # Display components
        self.fig = None
        self.axes = {}
        self.animation = None
        
        # Performance metrics
        self.metrics = {
            'mode_switches': 0,
            'detections_by_mode': {mode: 0 for mode in RadarMode},
            'tracks_by_mode': {mode: 0 for mode in RadarMode},
            'avg_processing_time': 0.0,
            'frame_rate': 0.0
        }
        
        self.setup_display()
        self.load_demo_scenario()
        
    def setup_display(self):
        """Setup advanced radar display with mode indicators"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor('black')
        
        # Create layout
        gs = self.fig.add_gridspec(3, 5, height_ratios=[2, 2, 1], width_ratios=[3, 1, 1, 1, 1])
        
        # Main radar display
        self.axes['radar'] = self.fig.add_subplot(gs[:2, 0], projection='polar')
        self.setup_radar_scope()
        
        # Mode control panel
        self.axes['modes'] = self.fig.add_subplot(gs[0, 1])
        self.axes['status'] = self.fig.add_subplot(gs[1, 1])
        self.axes['targets'] = self.fig.add_subplot(gs[0, 2])
        self.axes['performance'] = self.fig.add_subplot(gs[1, 2])
        self.axes['controls'] = self.fig.add_subplot(gs[0, 3])
        self.axes['parameters'] = self.fig.add_subplot(gs[1, 3])
        self.axes['alerts'] = self.fig.add_subplot(gs[0, 4])
        self.axes['history'] = self.fig.add_subplot(gs[1, 4])
        self.axes['info'] = self.fig.add_subplot(gs[2, :])
        
        # Style panels
        for name, ax in self.axes.items():
            if name != 'radar':
                ax.set_facecolor('#001133')
                for spine in ax.spines.values():
                    spine.set_color('#00ff00')
                    spine.set_linewidth(1)
                ax.tick_params(colors='#00ff00', labelsize=8)
        
        # Title
        self.fig.suptitle('ADVANCED RADAR SYSTEM - MULTIPLE OPERATING MODES', 
                         fontsize=18, color='#00ff00', weight='bold', y=0.95)
                         
    def setup_radar_scope(self):
        """Configure the main radar PPI scope"""
        ax = self.axes['radar']
        ax.set_facecolor('black')
        ax.set_ylim(0, 200)
        ax.set_title('RADAR PPI SCOPE\nMulti-Mode Operation', 
                    color='#00ff00', pad=20, fontsize=14, weight='bold')
        
        # Range rings
        for r in [50, 100, 150, 200]:
            circle = Circle((0, 0), r, fill=False, color='#00ff00', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
            ax.text(np.pi/4, r-5, f'{r}km', color='#00ff00', fontsize=10, ha='center')
        
        # Bearing lines
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            ax.plot([rad, rad], [0, 200], color='#00ff00', alpha=0.2, linewidth=0.5)
            ax.text(rad, 210, f'{angle}°', color='#00ff00', fontsize=9, ha='center')
        
        # Configure polar display
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.grid(True, color='#00ff00', alpha=0.2)
        ax.set_rticks([])
        ax.set_thetagrids([])
        
    def load_demo_scenario(self):
        """Load comprehensive scenario for mode demonstration"""
        print("📡 Loading advanced mode demonstration scenario...")
        
        # High-traffic scenario for mode testing
        aircraft_data = [
            (-80, 120, 90, 450),    # East-bound commercial
            (60, 140, 180, 520),    # South-bound heavy
            (-120, -80, 45, 380),   # Northeast light
            (90, -60, 315, 420),    # Northwest medium
            (-40, 160, 270, 480),   # West-bound fast
            (140, 40, 225, 360),    # Southwest slow
            (-60, -140, 0, 400),    # North-bound commercial
            (100, 100, 270, 340),   # West-bound light
            (20, 180, 180, 600),    # South-bound military
            (-100, 60, 135, 280)    # Southeast civilian
        ]
        
        for x, y, heading, speed in aircraft_data:
            self.data_generator.add_aircraft(x, y, heading, speed)
            
        # Naval vessels for track mode testing
        ship_data = [
            (-120, -160, 45, 25),   # Naval patrol
            (80, -180, 315, 18),    # Cargo vessel
            (-60, -170, 90, 12),    # Fishing fleet
            (130, -140, 225, 22),   # Coast guard
            (40, -190, 0, 15)       # Research vessel
        ]
        
        for x, y, heading, speed in ship_data:
            self.data_generator.add_ship(x, y, heading, speed)
            
        # Weather for TWS mode testing
        self.data_generator.add_weather_returns(-80, 60, 40)   # Storm system
        self.data_generator.add_weather_returns(100, 150, 30)  # Rain shower
        self.data_generator.add_weather_returns(-40, -120, 25) # Squall line
        
        total_targets = len(self.data_generator.targets)
        aircraft_count = sum(1 for t in self.data_generator.targets if t.target_type == 'aircraft')
        ship_count = sum(1 for t in self.data_generator.targets if t.target_type == 'ship')
        weather_count = sum(1 for t in self.data_generator.targets if t.target_type == 'weather')
        
        print(f"✅ Advanced scenario loaded: {total_targets} targets")
        print(f"   • {aircraft_count} aircraft (commercial, military, civilian)")
        print(f"   • {ship_count} ships (naval, commercial, research)")
        print(f"   • {weather_count} weather phenomena")
        
    def animate(self, frame):
        """Main animation with mode-specific behaviors"""
        if not self.is_running:
            self.update_static_displays()
            return []
            
        start_time = time.time()
        
        # Update system time
        self.current_time += 0.1
        
        # Get mode-specific parameters
        sweep_params = self.mode_manager.get_sweep_parameters(self.current_time)
        
        # Update sweep based on current mode
        if sweep_params['sweep_rate'] > 0:
            self.sweep_angle = (self.sweep_angle + sweep_params['sweep_rate'] * 0.1) % 360
        
        # Update target positions
        self.data_generator.update_targets(0.1)
        
        # Mode-specific detection processing
        self.process_mode_specific_detection(sweep_params)
        
        # Update all displays
        self.update_radar_display()
        self.update_all_panels()
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.metrics['avg_processing_time'] = (
            self.metrics['avg_processing_time'] * 0.9 + processing_time * 0.1
        )
        self.metrics['frame_rate'] = 1.0 / max(processing_time, 0.001)
        
        return []
    
    def process_mode_specific_detection(self, sweep_params):
        """Process detections based on current radar mode"""
        if sweep_params['sweep_rate'] == 0:  # Standby mode
            return
            
        # Get detections with mode-specific parameters
        detections = self.data_generator.simulate_radar_detection(
            self.sweep_angle,
            sweep_width_deg=sweep_params['sweep_width']
        )
        
        if not detections:
            return
            
        # Filter by mode range limit
        filtered_detections = [
            d for d in detections 
            if d.get('range', 0) <= sweep_params['range_limit']
        ]
        
        if not filtered_detections:
            return
            
        # Update signal processor threshold
        self.signal_processor.detection_threshold = sweep_params['detection_threshold']
        
        # Process through pipeline
        targets = self.target_detector.process_raw_detections(filtered_detections)
        
        if targets:
            # Update tracker with mode-specific parameters
            active_tracks = self.tracker.update(targets, self.current_time)
            
            # Update metrics
            current_mode = self.mode_manager.current_mode
            self.metrics['detections_by_mode'][current_mode] += len(filtered_detections)
            self.metrics['tracks_by_mode'][current_mode] = len(self.tracker.get_confirmed_tracks())
    
    def update_radar_display(self):
        """Update radar display with mode-specific appearance"""
        ax = self.axes['radar']
        ax.clear()
        self.setup_radar_scope()
        
        # Get mode display properties
        display_props = self.mode_manager.get_mode_display_properties()
        
        # Draw mode-specific sweep beam
        if self.mode_manager.current_mode != RadarMode.STANDBY:
            sweep_rad = np.radians(self.sweep_angle)
            config = self.mode_manager.get_current_config()
            beam_width = np.radians(config.sweep_width_deg)
            
            # Main beam with mode-specific color
            beam = Wedge((0, 0), config.max_range_km,
                        np.degrees(sweep_rad - beam_width/2),
                        np.degrees(sweep_rad + beam_width/2),
                        alpha=display_props['beam_alpha'], 
                        color=display_props['sweep_color'])
            ax.add_patch(beam)
            
            # Bright sweep line
            ax.plot([sweep_rad, sweep_rad], [0, config.max_range_km], 
                   color=display_props['sweep_color'], linewidth=3, alpha=0.9)
        
        # Mode-specific sweep trail
        if len(self.sweep_history) > 0:
            trail_length = min(len(self.sweep_history), 30)
            for i, (angle, timestamp) in enumerate(self.sweep_history[-trail_length:]):
                age_factor = (i + 1) / trail_length
                alpha = display_props['trail_alpha'] * age_factor
                trail_rad = np.radians(angle)
                max_range = self.mode_manager.get_current_config().max_range_km
                ax.plot([trail_rad, trail_rad], [0, max_range], 
                       color=display_props['sweep_color'], linewidth=1, alpha=alpha)
        
        # Add current sweep to history
        self.sweep_history.append((self.sweep_angle, self.current_time))
        if len(self.sweep_history) > 60:
            self.sweep_history = self.sweep_history[-60:]
        
        # Display tracks with mode-specific styling
        confirmed_tracks = self.tracker.get_confirmed_tracks()
        for track in confirmed_tracks:
            self.draw_mode_specific_track(ax, track, display_props)
        
        # Mode indicator
        ax.text(0.02, 0.98, f'MODE: {display_props["mode_text"]}', 
               transform=ax.transAxes, color=display_props['sweep_color'], 
               fontsize=14, weight='bold', verticalalignment='top')
        
        # Sweep angle display
        ax.text(0.02, 0.92, f'AZ: {self.sweep_angle:06.2f}°', 
               transform=ax.transAxes, color='#00ff00', fontsize=12,
               verticalalignment='top', fontfamily='monospace')
    
    def draw_mode_specific_track(self, ax, track, display_props):
        """Draw track with mode-specific styling"""
        # Convert to polar coordinates
        range_km = np.sqrt(track.state.x**2 + track.state.y**2)
        bearing_rad = np.arctan2(track.state.x, track.state.y)
        
        max_range = self.mode_manager.get_current_config().max_range_km
        if range_km > max_range:
            return
            
        # Mode-specific track symbol
        if self.mode_manager.current_mode == RadarMode.TRACK:
            marker = 's'  # Square for track mode
            size = 150
        elif self.mode_manager.current_mode == RadarMode.TRACK_WHILE_SCAN:
            marker = '^'  # Triangle for TWS
            size = 120
        else:  # Search mode
            marker = 'o'  # Circle for search
            size = 100
        
        # Track symbol
        ax.scatter(bearing_rad, range_km, s=size, c=display_props['target_color'], 
                  marker=marker, alpha=0.9, edgecolors='white', linewidths=2, zorder=20)
        
        # Track information
        info_text = f'T{track.id[-3:]}\n{track.state.speed_kmh:.0f}kt'
        ax.text(bearing_rad, range_km + 8, info_text, color=display_props['target_color'], 
               fontsize=9, ha='center', va='bottom', weight='bold')
        
        # Velocity vector for track and TWS modes
        if self.mode_manager.current_mode in [RadarMode.TRACK, RadarMode.TRACK_WHILE_SCAN]:
            speed = np.sqrt(track.state.vx**2 + track.state.vy**2)
            if speed > 0.5:
                vel_scale = max_range * 0.08
                end_x = track.state.x + track.state.vx * vel_scale
                end_y = track.state.y + track.state.vy * vel_scale
                end_range = np.sqrt(end_x**2 + end_y**2)
                end_bearing = np.arctan2(end_x, end_y)
                
                if end_range <= max_range:
                    ax.annotate('', xy=(end_bearing, end_range),
                               xytext=(bearing_rad, range_km),
                               arrowprops=dict(arrowstyle='->', 
                                             color=display_props['target_color'], 
                                             lw=2, alpha=0.8))
    
    def update_all_panels(self):
        """Update all information panels"""
        self.update_mode_panel()
        self.update_status_panel()
        self.update_targets_panel()
        self.update_performance_panel()
        self.update_controls_panel()
        self.update_parameters_panel()
        self.update_alerts_panel()
        self.update_history_panel()
        self.update_info_bar()
    
    def update_static_displays(self):
        """Update displays when system is stopped"""
        self.update_controls_panel()
        self.update_info_bar()
    
    def update_mode_panel(self):
        """Update radar mode control panel"""
        ax = self.axes['modes']
        ax.clear()
        ax.set_title('RADAR MODES', color='#00ff00', fontsize=11, weight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Mode buttons
        modes = [
            (RadarMode.SEARCH, (1, 7.5, 8, 1.5)),
            (RadarMode.TRACK, (1, 5.5, 8, 1.5)),
            (RadarMode.TRACK_WHILE_SCAN, (1, 3.5, 8, 1.5)),
            (RadarMode.STANDBY, (1, 1.5, 8, 1.5))
        ]
        
        for mode, (x, y, w, h) in modes:
            # Highlight current mode
            if mode == self.mode_manager.current_mode:
                color = '#006600'
                text_color = '#00ff00'
            else:
                color = '#333333'
                text_color = '#888888'
                
            rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='#00ff00', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, mode.value, ha='center', va='center',
                   color=text_color, fontsize=10, weight='bold')
        
        ax.axis('off')
    
    def update_status_panel(self):
        """Update system status panel"""
        ax = self.axes['status']
        ax.clear()
        ax.set_title('SYSTEM STATUS', color='#00ff00', fontsize=11, weight='bold')
        
        config = self.mode_manager.get_current_config()
        
        status_text = f"""
STATUS: {'ACTIVE' if self.is_running else 'STANDBY'}
MODE: {self.mode_manager.current_mode.value}
RANGE: {config.max_range_km:.0f} km
SWEEP: {config.sweep_rate_rpm:.0f} RPM
BEAM: {config.sweep_width_deg:.0f}°

THRESHOLD: {config.detection_threshold:.3f}
UPTIME: {self.current_time:.0f}s
        """.strip()
        
        ax.text(0.05, 0.95, status_text, transform=ax.transAxes,
               color='#00ff00', fontsize=9, verticalalignment='top',
               fontfamily='monospace')
        ax.axis('off')
    
    def update_targets_panel(self):
        """Update targets panel"""
        ax = self.axes['targets']
        ax.clear()
        ax.set_title('ACTIVE TARGETS', color='#00ff00', fontsize=11, weight='bold')
        
        tracks = self.tracker.get_confirmed_tracks()
        
        if tracks:
            targets_text = f"CONFIRMED: {len(tracks)}\n\n"
            for i, track in enumerate(tracks[:5]):
                range_km = np.sqrt(track.state.x**2 + track.state.y**2)
                bearing = np.degrees(np.arctan2(track.state.x, track.state.y)) % 360
                targets_text += f"T{track.id[-3:]}: {track.classification[:4].upper()}\n"
                targets_text += f"  {range_km:.1f}km @ {bearing:.0f}°\n"
                targets_text += f"  {track.state.speed_kmh:.0f}kt\n"
                if i < 4:
                    targets_text += "\n"
        else:
            targets_text = "NO CONFIRMED\nTARGETS"
            
        ax.text(0.05, 0.95, targets_text, transform=ax.transAxes,
               color='#00ff00', fontsize=8, verticalalignment='top',
               fontfamily='monospace')
        ax.axis('off')
    
    def update_performance_panel(self):
        """Update performance metrics panel"""
        ax = self.axes['performance']
        ax.clear()
        ax.set_title('PERFORMANCE', color='#00ff00', fontsize=11, weight='bold')
        
        current_mode = self.mode_manager.current_mode
        
        perf_text = f"""
FRAME RATE: {self.metrics['frame_rate']:.1f} FPS
PROC TIME: {self.metrics['avg_processing_time']*1000:.1f}ms

MODE STATS:
  Detections: {self.metrics['detections_by_mode'][current_mode]}
  Tracks: {self.metrics['tracks_by_mode'][current_mode]}
  
EFFICIENCY: {'OPTIMAL' if self.metrics['frame_rate'] > 30 else 'GOOD'}
        """.strip()
        
        ax.text(0.05, 0.95, perf_text, transform=ax.transAxes,
               color='#00ff00', fontsize=9, verticalalignment='top',
               fontfamily='monospace')
        ax.axis('off')
    
    def update_controls_panel(self):
        """Update system controls panel"""
        ax = self.axes['controls']
        ax.clear()
        ax.set_title('SYSTEM CONTROL', color='#00ff00', fontsize=11, weight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Control buttons
        buttons = [
            ('START', (1, 7.5, 8, 1.5), '#006600' if not self.is_running else '#333333'),
            ('STOP', (1, 5.5, 8, 1.5), '#660000' if self.is_running else '#333333'),
            ('RESET', (1, 3.5, 8, 1.5), '#444444'),
            ('AUTO', (1, 1.5, 8, 1.5), '#004466')
        ]
        
        for label, (x, y, w, h), color in buttons:
            rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='#00ff00', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                   color='#00ff00', fontsize=10, weight='bold')
        
        ax.axis('off')
    
    def update_parameters_panel(self):
        """Update radar parameters panel"""
        ax = self.axes['parameters']
        ax.clear()
        ax.set_title('PARAMETERS', color='#00ff00', fontsize=11, weight='bold')
        
        config = self.mode_manager.get_current_config()
        
        param_text = f"""
SENSITIVITY: {'HIGH' if config.detection_threshold < 0.1 else 'MEDIUM'}
RANGE GATE: {config.max_range_km:.0f} km
BEAM WIDTH: {config.sweep_width_deg:.0f}°
DWELL TIME: {config.dwell_time_ms:.0f} ms

UPDATE RATE: {config.track_update_rate_hz:.1f} Hz
PRIORITY SECTORS: {len(self.mode_manager.sector_priorities)}
        """.strip()
        
        ax.text(0.05, 0.95, param_text, transform=ax.transAxes,
               color='#00ff00', fontsize=9, verticalalignment='top',
               fontfamily='monospace')
        ax.axis('off')
    
    def update_alerts_panel(self):
        """Update system alerts panel"""
        ax = self.axes['alerts']
        ax.clear()
        ax.set_title('ALERTS', color='#00ff00', fontsize=11, weight='bold')
        
        # Generate dynamic alerts based on system state
        alerts = []
        
        if self.is_running:
            tracks = self.tracker.get_confirmed_tracks()
            if len(tracks) > 15:
                alerts.append("⚠️ HIGH TRAFFIC")
            if self.metrics['frame_rate'] < 20:
                alerts.append("⚠️ PERFORMANCE")
            if self.mode_manager.current_mode == RadarMode.TRACK and len(tracks) == 0:
                alerts.append("ℹ️ NO TRACK TARGETS")
        else:
            alerts.append("ℹ️ SYSTEM STANDBY")
        
        if not alerts:
            alerts.append("✅ ALL NORMAL")
        
        alert_text = "\n".join(alerts)
        
        ax.text(0.05, 0.95, alert_text, transform=ax.transAxes,
               color='#00ff00', fontsize=9, verticalalignment='top',
               fontfamily='monospace')
        ax.axis('off')
    
    def update_history_panel(self):
        """Update mode history panel"""
        ax = self.axes['history']
        ax.clear()
        ax.set_title('MODE HISTORY', color='#00ff00', fontsize=11, weight='bold')
        
        mode_duration = self.current_time - self.mode_manager.mode_start_time
        
        history_text = f"""
CURRENT MODE:
{self.mode_manager.current_mode.value}

DURATION: {mode_duration:.1f}s
SWITCHES: {self.metrics['mode_switches']}

EFFICIENCY:
Search: {self.metrics['detections_by_mode'][RadarMode.SEARCH]}
Track: {self.metrics['tracks_by_mode'][RadarMode.TRACK]}
TWS: {self.metrics['tracks_by_mode'][RadarMode.TRACK_WHILE_SCAN]}
        """.strip()
        
        ax.text(0.05, 0.95, history_text, transform=ax.transAxes,
               color='#00ff00', fontsize=8, verticalalignment='top',
               fontfamily='monospace')
        ax.axis('off')
    
    def update_info_bar(self):
        """Update bottom information bar"""
        ax = self.axes['info']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        
        mode_name = self.mode_manager.current_mode.value
        config = self.mode_manager.get_current_config()
        tracks = len(self.tracker.get_confirmed_tracks())
        
        info_text = (f"DAY 7 ADVANCED MODES: {mode_name} Mode Active | "
                    f"Range: {config.max_range_km:.0f}km | "
                    f"Sweep: {config.sweep_rate_rpm:.0f}RPM | "
                    f"Beam: {config.sweep_width_deg:.0f}° | "
                    f"Tracks: {tracks} | "
                    f"Time: {self.current_time:.1f}s | "
                    f"FPS: {self.metrics['frame_rate']:.1f}")
        
        ax.text(5, 0.5, info_text, ha='center', va='center',
               color='#00ff00', fontsize=11, weight='bold')
        ax.axis('off')
    
    def on_click(self, event):
        """Handle mouse clicks for mode switching and controls"""
        if event.inaxes == self.axes['modes']:
            # Mode switching
            x, y = event.xdata, event.ydata
            if x is not None and y is not None and 1 <= x <= 9:
                if 7.5 <= y <= 9:
                    self.switch_mode(RadarMode.SEARCH)
                elif 5.5 <= y <= 7:
                    self.switch_mode(RadarMode.TRACK)
                elif 3.5 <= y <= 5:
                    self.switch_mode(RadarMode.TRACK_WHILE_SCAN)
                elif 1.5 <= y <= 3:
                    self.switch_mode(RadarMode.STANDBY)
                    
        elif event.inaxes == self.axes['controls']:
            # System controls
            x, y = event.xdata, event.ydata
            if x is not None and y is not None and 1 <= x <= 9:
                if 7.5 <= y <= 9:  # START
                    self.start_system()
                elif 5.5 <= y <= 7:  # STOP
                    self.stop_system()
                elif 3.5 <= y <= 5:  # RESET
                    self.reset_system()
                elif 1.5 <= y <= 3:  # AUTO
                    self.auto_mode_cycle()
    
    def switch_mode(self, new_mode: RadarMode):
        """Switch to new radar mode"""
        if new_mode != self.mode_manager.current_mode:
            self.mode_manager.set_mode(new_mode)
            self.metrics['mode_switches'] += 1
            
            # Adjust system parameters for new mode
            config = self.mode_manager.get_current_config()
            self.signal_processor.detection_threshold = config.detection_threshold
            self.tracker.max_association_distance = config.max_range_km * 0.05
            
            # Add priority sectors for track mode
            if new_mode == RadarMode.TRACK:
                tracks = self.tracker.get_confirmed_tracks()
                if tracks:
                    for track in tracks[:3]:  # Focus on first 3 tracks
                        bearing = np.degrees(np.arctan2(track.state.x, track.state.y))
                        self.mode_manager.add_priority_sector(bearing - 15, bearing + 15)
    
    def start_system(self):
        """Start the advanced radar system"""
        if not self.is_running:
            self.is_running = True
            self.current_time = 0.0
            self.sweep_angle = 0.0
            print(f"🚀 Advanced Radar System STARTED in {self.mode_manager.current_mode.value} mode")
    
    def stop_system(self):
        """Stop the radar system"""
        if self.is_running:
            self.is_running = False
            print("🛑 Advanced Radar System STOPPED")
    
    def reset_system(self):
        """Reset the system"""
        self.stop_system()
        self.tracker = MultiTargetTracker()
        self.mode_manager.set_mode(RadarMode.SEARCH)
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.target_trails = {}
        self.mode_manager.sector_priorities = []
        print("🔄 Advanced System RESET to Search mode")
    
    def auto_mode_cycle(self):
        """Automatically cycle through modes for demonstration"""
        if not self.is_running:
            self.start_system()
            
        modes = [RadarMode.SEARCH, RadarMode.TRACK_WHILE_SCAN, RadarMode.TRACK]
        current_index = modes.index(self.mode_manager.current_mode) if self.mode_manager.current_mode in modes else -1
        next_mode = modes[(current_index + 1) % len(modes)]
        self.switch_mode(next_mode)
        
        print(f"🔄 Auto mode cycle: {next_mode.value}")
    
    def run_demo(self):
        """Run the advanced radar modes demonstration"""
        print("\n" + "="*70)
        print("🎯 DAY 7 TASK 1: ADVANCED RADAR MODES DEMONSTRATION")
        print("="*70)
        print("\nThis demonstration showcases professional radar operating modes:")
        print("✅ SEARCH Mode: Wide-area scanning for new target detection")
        print("✅ TRACK Mode: Focused tracking of confirmed targets")
        print("✅ TWS Mode: Track-While-Scan hybrid operation")
        print("✅ STANDBY Mode: System standby with minimal power")
        print("\n🎛️  Interactive Controls:")
        print("  • Click mode buttons (left panel) to switch radar modes")
        print("  • Use system controls (START/STOP/RESET/AUTO)")
        print("  • AUTO button cycles through modes automatically")
        print("  • Observe how each mode changes radar behavior")
        print("\n🔍 Mode Characteristics:")
        print("  • SEARCH: 30 RPM, 30° beam, 200km range (green)")
        print("  • TRACK: 60 RPM, 10° beam, 150km range (red)")
        print("  • TWS: 45 RPM, 20° beam, 175km range (cyan)")
        print("  • STANDBY: No sweep, minimal processing (gray)")
        print("\n💡 Watch how targets appear differently in each mode!")
        print("="*70)
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Start animation
        self.animation = FuncAnimation(self.fig, self.animate, interval=100,
                                     blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
        
        print("\n🎉 Advanced Radar Modes demonstration complete!")
        print("✅ Professional multi-mode radar system operational")

def main():
    """Run the advanced radar modes demonstration"""
    try:
        system = AdvancedRadarSystem()
        system.run_demo()
    except Exception as e:
        print(f"❌ Error running advanced modes demo: {e}")
        print("Make sure all radar components are available")

if __name__ == "__main__":
    main()