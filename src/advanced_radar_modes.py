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
    
    