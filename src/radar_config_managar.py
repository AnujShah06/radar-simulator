"""
Professional Radar Configuration Manager - Day 7 Task 2
=======================================================
Real-time radar parameter adjustment and configuration management system.
Provides operator controls for live system tuning and operational presets.

Features:
• Real-time parameter adjustment (range, sensitivity, filters)
• Interactive control interface with sliders and toggles
• Configuration presets for different operational scenarios
• Live parameter validation with safety limits
• Performance impact monitoring
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Wedge, FancyBboxPatch
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
import json

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
    TWS = "TWS"
    WEATHER = "WEATHER"
    STANDBY = "STANDBY"

class ConfigPreset(Enum):
    """Predefined configuration presets"""
    AIRPORT_CONTROL = "Airport Control"
    NAVAL_SURVEILLANCE = "Naval Surveillance"
    MILITARY_DEFENSE = "Military Defense"
    WEATHER_MONITORING = "Weather Monitoring"
    COASTAL_PATROL = "Coastal Patrol"
    CUSTOM = "Custom"

@dataclass
class RadarConfiguration:
    """Complete radar system configuration"""
    # Detection parameters
    max_range_km: float = 200.0
    min_range_km: float = 5.0
    detection_threshold: float = 0.08
    false_alarm_rate: float = 0.05
    
    # Sweep parameters
    sweep_rate_rpm: float = 30.0
    beam_width_deg: float = 2.0
    antenna_gain_db: float = 35.0
    
    # Tracking parameters
    max_association_distance: float = 10.0
    min_hits_for_confirmation: int = 1
    max_missed_detections: int = 15
    track_aging_time: float = 45.0
    
    # Filter settings
    clutter_rejection: bool = True
    weather_filtering: bool = True
    moving_target_indicator: bool = True
    sea_clutter_suppression: bool = False
    
    # Display settings
    trail_length_sec: float = 30.0
    update_rate_hz: float = 10.0
    brightness: float = 1.0
    contrast: float = 1.0
    
    # Alert settings
    proximity_alert_range: float = 50.0
    speed_alert_threshold: float = 1000.0  # km/h
    enable_audio_alerts: bool = True
    
    # System settings
    transmitter_power_kw: float = 100.0
    receiver_sensitivity_dbm: float = -110.0
    current_mode: RadarMode = RadarMode.SEARCH
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'max_range_km': self.max_range_km,
            'min_range_km': self.min_range_km,
            'detection_threshold': self.detection_threshold,
            'false_alarm_rate': self.false_alarm_rate,
            'sweep_rate_rpm': self.sweep_rate_rpm,
            'beam_width_deg': self.beam_width_deg,
            'antenna_gain_db': self.antenna_gain_db,
            'max_association_distance': self.max_association_distance,
            'min_hits_for_confirmation': self.min_hits_for_confirmation,
            'max_missed_detections': self.max_missed_detections,
            'track_aging_time': self.track_aging_time,
            'clutter_rejection': self.clutter_rejection,
            'weather_filtering': self.weather_filtering,
            'moving_target_indicator': self.moving_target_indicator,
            'sea_clutter_suppression': self.sea_clutter_suppression,
            'trail_length_sec': self.trail_length_sec,
            'update_rate_hz': self.update_rate_hz,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'proximity_alert_range': self.proximity_alert_range,
            'speed_alert_threshold': self.speed_alert_threshold,
            'enable_audio_alerts': self.enable_audio_alerts,
            'transmitter_power_kw': self.transmitter_power_kw,
            'receiver_sensitivity_dbm': self.receiver_sensitivity_dbm,
            'current_mode': self.current_mode.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RadarConfiguration':
        """Create configuration from dictionary"""
        config = cls()
        for key, value in data.items():
            if key == 'current_mode':
                config.current_mode = RadarMode(value)
            elif hasattr(config, key):
                setattr(config, key, value)
        return config

class ConfigurationManager:
    """Manages radar configuration presets and validation"""
    
    def __init__(self):
        self.presets = self._create_default_presets()
        self.current_config = RadarConfiguration()
        self.validation_rules = self._create_validation_rules()
        
    def _create_default_presets(self) -> Dict[ConfigPreset, RadarConfiguration]:
        """Create default configuration presets"""
        presets = {}
        
        # Airport Control - High precision, moderate range
        airport = RadarConfiguration(
            max_range_km=150.0,
            min_range_km=2.0,
            detection_threshold=0.06,
            sweep_rate_rpm=60.0,
            beam_width_deg=1.5,
            max_association_distance=5.0,
            clutter_rejection=True,
            weather_filtering=True,
            proximity_alert_range=20.0,
            transmitter_power_kw=50.0,
            current_mode=RadarMode.TWS
        )
        presets[ConfigPreset.AIRPORT_CONTROL] = airport
        
        # Naval Surveillance - Long range, sea clutter rejection
        naval = RadarConfiguration(
            max_range_km=300.0,
            min_range_km=10.0,
            detection_threshold=0.10,
            sweep_rate_rpm=20.0,
            beam_width_deg=3.0,
            max_association_distance=15.0,
            sea_clutter_suppression=True,
            moving_target_indicator=True,
            proximity_alert_range=100.0,
            transmitter_power_kw=200.0,
            current_mode=RadarMode.SEARCH
        )
        presets[ConfigPreset.NAVAL_SURVEILLANCE] = naval
        
        # Military Defense - Maximum performance
        military = RadarConfiguration(
            max_range_km=400.0,
            min_range_km=5.0,
            detection_threshold=0.04,
            sweep_rate_rpm=90.0,
            beam_width_deg=1.0,
            max_association_distance=20.0,
            min_hits_for_confirmation=1,
            max_missed_detections=20,
            clutter_rejection=True,
            moving_target_indicator=True,
            proximity_alert_range=200.0,
            speed_alert_threshold=500.0,
            transmitter_power_kw=500.0,
            antenna_gain_db=45.0,
            current_mode=RadarMode.TRACK
        )
        presets[ConfigPreset.MILITARY_DEFENSE] = military
        
        # Weather Monitoring - Specialized for meteorological detection
        weather = RadarConfiguration(
            max_range_km=250.0,
            min_range_km=1.0,
            detection_threshold=0.15,
            sweep_rate_rpm=15.0,
            beam_width_deg=4.0,
            max_association_distance=25.0,
            clutter_rejection=False,
            weather_filtering=False,
            moving_target_indicator=False,
            transmitter_power_kw=150.0,
            current_mode=RadarMode.WEATHER
        )
        presets[ConfigPreset.WEATHER_MONITORING] = weather
        
        # Coastal Patrol - Balanced for mixed environment
        coastal = RadarConfiguration(
            max_range_km=180.0,
            min_range_km=3.0,
            detection_threshold=0.08,
            sweep_rate_rpm=40.0,
            beam_width_deg=2.5,
            max_association_distance=12.0,
            sea_clutter_suppression=True,
            weather_filtering=True,
            proximity_alert_range=75.0,
            transmitter_power_kw=100.0,
            current_mode=RadarMode.TWS
        )
        presets[ConfigPreset.COASTAL_PATROL] = coastal
        
        return presets
    
    def _create_validation_rules(self) -> Dict[str, Tuple[float, float]]:
        """Create parameter validation rules (min, max)"""
        return {
            'max_range_km': (10.0, 500.0),
            'min_range_km': (0.1, 50.0),
            'detection_threshold': (0.01, 1.0),
            'false_alarm_rate': (0.001, 0.5),
            'sweep_rate_rpm': (5.0, 120.0),
            'beam_width_deg': (0.5, 10.0),
            'antenna_gain_db': (20.0, 60.0),
            'max_association_distance': (1.0, 50.0),
            'min_hits_for_confirmation': (1, 10),
            'max_missed_detections': (3, 50),
            'track_aging_time': (10.0, 300.0),
            'trail_length_sec': (5.0, 120.0),
            'update_rate_hz': (1.0, 30.0),
            'brightness': (0.1, 2.0),
            'contrast': (0.1, 2.0),
            'proximity_alert_range': (5.0, 500.0),
            'speed_alert_threshold': (50.0, 5000.0),
            'transmitter_power_kw': (10.0, 1000.0),
            'receiver_sensitivity_dbm': (-130.0, -80.0)
        }
    
    def validate_parameter(self, param_name: str, value: float) -> Tuple[bool, str]:
        """Validate a parameter value"""
        if param_name not in self.validation_rules:
            return True, ""
            
        min_val, max_val = self.validation_rules[param_name]
        if value < min_val:
            return False, f"{param_name} must be >= {min_val}"
        if value > max_val:
            return False, f"{param_name} must be <= {max_val}"
            
        return True, ""
    
    def apply_preset(self, preset: ConfigPreset) -> RadarConfiguration:
        """Apply a configuration preset"""
        if preset in self.presets:
            self.current_config = RadarConfiguration(**self.presets[preset].__dict__)
            print(f"🎛️  Applied preset: {preset.value}")
            return self.current_config
        else:
            print(f"❌ Preset {preset.value} not found")
            return self.current_config
    
    def save_preset(self, name: str, config: RadarConfiguration):
        """Save current configuration as custom preset"""
        # In a real system, this would save to file
        print(f"💾 Saved configuration as '{name}'")
    
    def get_config_summary(self, config: RadarConfiguration) -> str:
        """Get human-readable configuration summary"""
        return f"""
Range: {config.min_range_km:.1f}-{config.max_range_km:.1f}km
Sensitivity: {config.detection_threshold:.3f}
Sweep: {config.sweep_rate_rpm:.0f} RPM
Beam: {config.beam_width_deg:.1f}°
Power: {config.transmitter_power_kw:.0f}kW
Mode: {config.current_mode.value}
        """.strip()

class ConfigurableRadarSystem:
    """
    Configurable Radar System with Real-Time Parameter Adjustment
    
    This system allows operators to adjust radar parameters in real-time
    and apply operational presets for different mission requirements.
    """
    
    def __init__(self):
        print("🎛️  Initializing Configurable Radar System...")
        
        # Core components
        self.data_generator = RadarDataGenerator(max_range_km=200)
        self.signal_processor = SignalProcessor()
        self.target_detector = TargetDetector()
        self.tracker = MultiTargetTracker()
        
        # Configuration management
        self.config_manager = ConfigurationManager()
        self.current_config = self.config_manager.current_config
        
        # Apply initial configuration
        self.apply_configuration(self.current_config)
        
        # System state
        self.is_running = False
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.sweep_history = []
        self.target_trails = {}
        self.config_changed = False
        
        # Performance metrics
        self.metrics = {
            'confirmed_tracks': 0,
            'total_detections': 0,
            'false_alarms': 0,
            'config_changes': 0,
            'avg_processing_time': 0.0,
            'frame_rate': 0.0,
            'power_consumption': 0.0,
            'detection_range_actual': 0.0
        }
        
        # UI components
        self.fig = None
        self.axes = {}
        self.sliders = {}
        self.checkboxes = {}
        self.radio_buttons = {}
        self.animation = None
        
        self.setup_configurable_display()
        self.load_demo_scenario()
        
    def apply_configuration(self, config: RadarConfiguration):
        """Apply configuration to all radar components"""
        # Update signal processor
        self.signal_processor.detection_threshold = config.detection_threshold
        self.signal_processor.false_alarm_rate = config.false_alarm_rate
        
        # Update target detector
        self.target_detector.min_detections_for_confirmation = config.min_hits_for_confirmation
        self.target_detector.association_distance_threshold = config.max_association_distance
        
        # Update tracker
        self.tracker.max_association_distance = config.max_association_distance
        self.tracker.min_hits_for_confirmation = config.min_hits_for_confirmation
        self.tracker.max_missed_detections = config.max_missed_detections
        self.tracker.max_track_age_without_update = config.track_aging_time
        
        # Update data generator range
        self.data_generator.max_range_km = config.max_range_km
        
        # Calculate derived parameters
        self.calculate_performance_metrics(config)
        
        self.config_changed = True
        self.metrics['config_changes'] += 1
        
        print(f"🔧 Configuration applied:")
        print(f"   • Range: {config.min_range_km:.1f}-{config.max_range_km:.1f}km")
        print(f"   • Threshold: {config.detection_threshold:.3f}")
        print(f"   • Power: {config.transmitter_power_kw:.0f}kW")
        
    def calculate_performance_metrics(self, config: RadarConfiguration):
        """Calculate performance metrics based on configuration"""
        # Simplified radar range equation: R = (P * G^2 * σ * λ^2) / ((4π)^3 * S_min)
        # Estimate actual detection range based on parameters
        power_factor = config.transmitter_power_kw / 100.0  # Normalized to 100kW
        gain_factor = (10 ** (config.antenna_gain_db / 10)) / (10 ** (35.0 / 10))  # Normalized to 35dB
        sensitivity_factor = 1.0 / config.detection_threshold
        
        self.metrics['detection_range_actual'] = config.max_range_km * power_factor * gain_factor * sensitivity_factor * 0.3
        self.metrics['power_consumption'] = config.transmitter_power_kw * 1.2  # Include cooling, etc.
        
    def setup_configurable_display(self):
        """Setup configurable radar display with control panels"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.patch.set_facecolor('black')
        
        # Create complex layout for configuration interface
        gs = self.fig.add_gridspec(4, 6, height_ratios=[3, 1, 1, 1], width_ratios=[3, 1, 1, 1, 1, 1])
        
        # Main radar display
        self.axes['radar'] = self.fig.add_subplot(gs[0, :3], projection='polar')
        self.setup_radar_scope()
        
        # Configuration panels
        self.axes['presets'] = self.fig.add_subplot(gs[0, 3])
        self.axes['detection'] = self.fig.add_subplot(gs[0, 4])
        self.axes['tracking'] = self.fig.add_subplot(gs[0, 5])
        
        # Slider panels
        self.axes['range_sliders'] = self.fig.add_subplot(gs[1, :3])
        self.axes['sensitivity_sliders'] = self.fig.add_subplot(gs[2, :3])
        self.axes['power_sliders'] = self.fig.add_subplot(gs[3, :3])
        
        # Status panels
        self.axes['status'] = self.fig.add_subplot(gs[1, 3])
        self.axes['performance'] = self.fig.add_subplot(gs[1, 4])
        self.axes['alerts'] = self.fig.add_subplot(gs[1, 5])
        
        # Filter controls
        self.axes['filters'] = self.fig.add_subplot(gs[2, 3])
        self.axes['display'] = self.fig.add_subplot(gs[2, 4])
        self.axes['system'] = self.fig.add_subplot(gs[2, 5])
        
        # Control buttons
        self.axes['controls'] = self.fig.add_subplot(gs[3, 3:])
        
        # Style all panels
        for name, ax in self.axes.items():
            if name not in ['radar', 'range_sliders', 'sensitivity_sliders', 'power_sliders']:
                ax.set_facecolor('#001122')
                for spine in ax.spines.values():
                    spine.set_color('#00ff00')
                    spine.set_linewidth(1)
                ax.tick_params(colors='#00ff00', labelsize=8)
        
        # Setup interactive controls
        self.setup_sliders()
        self.setup_preset_buttons()
        self.setup_filter_controls()
        
        # Title
        self.fig.suptitle('CONFIGURABLE RADAR SYSTEM - REAL-TIME PARAMETER ADJUSTMENT', 
                         fontsize=18, color='#00ff00', weight='bold', y=0.95)
    
    def setup_radar_scope(self):
        """Configure the main radar PPI scope"""
        ax = self.axes['radar']
        ax.set_facecolor('black')
        ax.set_ylim(0, self.current_config.max_range_km)
        ax.set_title('CONFIGURABLE RADAR PPI SCOPE\nReal-Time Parameter Control', 
                    color='#00ff00', pad=20, fontsize=14, weight='bold')
        
        # Dynamic range rings based on configuration
        max_range = self.current_config.max_range_km
        ring_interval = max_range / 4
        
        for i in range(1, 5):
            r = ring_interval * i
            circle = Circle((0, 0), r, fill=False, color='#00ff00', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
            ax.text(np.pi/4, r-ring_interval*0.1, f'{r:.0f}km', color='#00ff00', fontsize=10, ha='center')
        
        # Bearing lines
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            ax.plot([rad, rad], [0, max_range], color='#00ff00', alpha=0.2, linewidth=0.5)
            ax.text(rad, max_range*1.05, f'{angle}°', color='#00ff00', fontsize=9, ha='center')
        
        # Configure polar display
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.grid(True, color='#00ff00', alpha=0.2)
        ax.set_rticks([])
        ax.set_thetagrids([])
    
    def setup_sliders(self):
        """Setup parameter adjustment sliders"""
        # Range sliders
        ax_range = self.axes['range_sliders']
        ax_range.set_title('RANGE CONTROLS', color='#00ff00', fontsize=11, weight='bold')
        
        # Max range slider
        slider_ax1 = plt.axes([0.1, 0.65, 0.4, 0.03], facecolor='#001122')
        self.sliders['max_range'] = Slider(
            slider_ax1, 'Max Range (km)', 50, 500, 
            valinit=self.current_config.max_range_km, 
            color='#00ff00', track_color='#003300'
        )
        
        # Min range slider
        slider_ax2 = plt.axes([0.1, 0.6, 0.4, 0.03], facecolor='#001122')
        self.sliders['min_range'] = Slider(
            slider_ax2, 'Min Range (km)', 0.1, 50, 
            valinit=self.current_config.min_range_km,
            color='#00ff00', track_color='#003300'
        )
        
        # Sensitivity sliders
        ax_sens = self.axes['sensitivity_sliders']
        ax_sens.set_title('SENSITIVITY CONTROLS', color='#00ff00', fontsize=11, weight='bold')
        
        # Detection threshold slider
        slider_ax3 = plt.axes([0.1, 0.45, 0.4, 0.03], facecolor='#001122')
        self.sliders['threshold'] = Slider(
            slider_ax3, 'Detection Threshold', 0.01, 0.5, 
            valinit=self.current_config.detection_threshold,
            color='#ffff00', track_color='#333300'
        )
        
        # False alarm rate slider
        slider_ax4 = plt.axes([0.1, 0.4, 0.4, 0.03], facecolor='#001122')
        self.sliders['false_alarm'] = Slider(
            slider_ax4, 'False Alarm Rate', 0.001, 0.2, 
            valinit=self.current_config.false_alarm_rate,
            color='#ffff00', track_color='#333300'
        )
        
        # Power sliders
        ax_power = self.axes['power_sliders']
        ax_power.set_title('POWER & SWEEP CONTROLS', color='#00ff00', fontsize=11, weight='bold')
        
        # Transmitter power slider
        slider_ax5 = plt.axes([0.1, 0.25, 0.4, 0.03], facecolor='#001122')
        self.sliders['power'] = Slider(
            slider_ax5, 'TX Power (kW)', 10, 500, 
            valinit=self.current_config.transmitter_power_kw,
            color='#ff4400', track_color='#330000'
        )
        
        # Sweep rate slider
        slider_ax6 = plt.axes([0.1, 0.2, 0.4, 0.03], facecolor='#001122')
        self.sliders['sweep_rate'] = Slider(
            slider_ax6, 'Sweep Rate (RPM)', 5, 120, 
            valinit=self.current_config.sweep_rate_rpm,
            color='#ff4400', track_color='#330000'
        )
        
        # Connect slider events
        for slider_name, slider in self.sliders.items():
            slider.on_changed(lambda val, name=slider_name: self.on_slider_change(name, val))
        
        # Hide slider panel axes
        for ax_name in ['range_sliders', 'sensitivity_sliders', 'power_sliders']:
            self.axes[ax_name].axis('off')
    
    def setup_preset_buttons(self):
        """Setup configuration preset buttons"""
        ax = self.axes['presets']
        ax.clear()
        ax.set_title('CONFIG PRESETS', color='#00ff00', fontsize=11, weight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        presets = [
            (ConfigPreset.AIRPORT_CONTROL, (0.5, 8, 9, 1.2), '#004400'),
            (ConfigPreset.NAVAL_SURVEILLANCE, (0.5, 6.5, 9, 1.2), '#000044'),
            (ConfigPreset.MILITARY_DEFENSE, (0.5, 5, 9, 1.2), '#440000'),
            (ConfigPreset.WEATHER_MONITORING, (0.5, 3.5, 9, 1.2), '#404000'),
            (ConfigPreset.COASTAL_PATROL, (0.5, 2, 9, 1.2), '#004440'),
            (ConfigPreset.CUSTOM, (0.5, 0.5, 9, 1.2), '#404040')
        ]
        
        for preset, (x, y, w, h), color in presets:
            rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='#00ff00', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, preset.value, ha='center', va='center',
                   color='#00ff00', fontsize=8, weight='bold')
        
        ax.axis('off')
    
    def setup_filter_controls(self):
        """Setup filter and option controls"""
        # Filter controls
        ax_filters = self.axes['filters']
        ax_filters.clear()
        ax_filters.set_title('FILTERS', color='#00ff00', fontsize=11, weight='bold')
        
        filter_text = f"""
CLUTTER REJECT: {'ON' if self.current_config.clutter_rejection else 'OFF'}
WEATHER FILTER: {'ON' if self.current_config.weather_filtering else 'OFF'}
MTI: {'ON' if self.current_config.moving_target_indicator else 'OFF'}
SEA CLUTTER: {'ON' if self.current_config.sea_clutter_suppression else 'OFF'}
        """.strip()
        
        ax_filters.text(0.05, 0.95, filter_text, transform=ax_filters.transAxes,
                       color='#00ff00', fontsize=9, verticalalignment='top',
                       fontfamily='monospace')
        ax_filters.axis('off')
        
        # Display controls
        ax_display = self.axes['display']
        ax_display.clear()
        ax_display.set_title('DISPLAY', color='#00ff00', fontsize=11, weight='bold')
        
        display_text = f"""
BRIGHTNESS: {self.current_config.brightness:.1f}
CONTRAST: {self.current_config.contrast:.1f}
TRAIL: {self.current_config.trail_length_sec:.0f}s
UPDATE: {self.current_config.update_rate_hz:.1f}Hz
        """.strip()
        
        ax_display.text(0.05, 0.95, display_text, transform=ax_display.transAxes,
                       color='#00ff00', fontsize=9, verticalalignment='top',
                       fontfamily='monospace')
        ax_display.axis('off')
    
    def on_slider_change(self, slider_name: str, value: float):
        """Handle slider value changes"""
        # Validate parameter
        param_map = {
            'max_range': 'max_range_km',
            'min_range': 'min_range_km',
            'threshold': 'detection_threshold',
            'false_alarm': 'false_alarm_rate',
            'power': 'transmitter_power_kw',
            'sweep_rate': 'sweep_rate_rpm'
        }
        
        if slider_name in param_map:
            param_name = param_map[slider_name]
            is_valid, error_msg = self.config_manager.validate_parameter(param_name, value)
            
            if is_valid:
                # Update configuration
                setattr(self.current_config, param_name, value)
                self.apply_configuration(self.current_config)
                
                # Special handling for range changes
                if slider_name in ['max_range', 'min_range']:
                    self.setup_radar_scope()  # Redraw scope with new range
                    
                print(f"🎛️  {param_name}: {value:.3f}")
            else:
                print(f"❌ {error_msg}")
    
    