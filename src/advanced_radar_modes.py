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
    
