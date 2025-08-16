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
    
