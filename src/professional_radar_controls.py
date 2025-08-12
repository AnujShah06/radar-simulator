"""
Professional Radar Control System - Clean Layout Version
Fixed layout with proper widget positioning and clean professional appearance
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons, RadioButtons
from matplotlib.patches import Rectangle, Circle
from dataclasses import dataclass
from typing import Dict, List, Callable, Any
from enum import Enum
import time
from datetime import datetime

# Real radar system modes
class RadarMode(Enum):
    SEARCH = "Search"
    TRACK = "Track"
    TWS = "Track-While-Scan"
    STANDBY = "Standby"
    CALIBRATION = "Calibration"

class AlertLevel(Enum):
    ROUTINE = "ROUTINE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

@dataclass
class RadarConfiguration:
    """Professional radar system configuration"""
    max_range_km: float = 250.0
    sweep_rate_rpm: float = 15.0
    sensitivity_db: float = -110.0
    detection_threshold: float = 12.0
    radar_mode: RadarMode = RadarMode.SEARCH
    clutter_map_enable: bool = True
    weather_filter_enable: bool = True
    range_scale_km: float = 100.0
    trail_length_sec: float = 30.0

@dataclass
class SystemAlert:
    """Professional radar system alert"""
    timestamp: float
    level: AlertLevel
    message: str
    source: str
    acknowledged: bool = False

class ProfessionalRadarControls:
    """Clean professional radar control system"""
    
    def __init__(self, config: RadarConfiguration = None):
        self.config = config or RadarConfiguration()
        self.alerts: List[SystemAlert] = []
        self.callbacks: Dict[str, Callable] = {}
        
        # System status
        self.system_status = {
            'transmitter_power_kw': 850.0,
            'receiver_noise_db': -113.0,
            'antenna_elevation_deg': 2.5,
            'cooling_temp_c': 25.0,
            'waveguide_pressure_psi': 15.2,
            'bite_status': 'GO'
        }
        
        # Performance metrics
        self.performance_metrics = {
            'tracks_active': 0,
            'track_capacity': 256,
            'processor_load_percent': 45.0,
            'uptime_hours': 0.0,
            'availability_percent': 99.7
        }
        
        # Control states
        self.is_running = False
        self.emergency_stop_active = False
        
        self.setup_clean_gui()
        
    def setup_clean_gui(self):
        """Create clean, professional GUI layout with proper spacing"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(18, 12))  # Much larger window
        self.fig.suptitle('PROFESSIONAL RADAR CONTROL SYSTEM', 
                         fontsize=18, color='#00FF00', weight='bold')
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Much more spaced out grid layout
        gs = self.fig.add_gridspec(4, 4, 
                                  height_ratios=[1, 2, 2, 1], 
                                  width_ratios=[1, 1, 1, 1],
                                  hspace=0.4,  # Much more vertical space
                                  wspace=0.3)  # Much more horizontal space
        
        # Top status panel (full width)
        self.status_ax = self.fig.add_subplot(gs[0, :])
        
        # Middle control panels (well separated)
        self.radar_ax = self.fig.add_subplot(gs[1, 0])
        self.mode_ax = self.fig.add_subplot(gs[1, 1]) 
        self.display_ax = self.fig.add_subplot(gs[1, 2])
        self.monitor_ax = self.fig.add_subplot(gs[1, 3])
        
        # Second row of panels
        self.filter_ax = self.fig.add_subplot(gs[2, 0])
        self.alert_ax = self.fig.add_subplot(gs[2, 1])
        self.performance_ax = self.fig.add_subplot(gs[2, 2])
        self.scenario_ax = self.fig.add_subplot(gs[2, 3])
        
        # Bottom command panel (full width)
        self.command_ax = self.fig.add_subplot(gs[3, :])
        
        self.create_clean_controls()
        self.start_update_timer()
        
    def create_clean_controls(self):
        """Create all control panels with clean layout"""
        self.create_system_status()
        self.create_radar_controls()
        self.create_mode_controls()
        self.create_display_controls()
        self.create_system_monitor()
        self.create_filter_controls()
        self.create_alert_panel()
        self.create_performance_panel()
        self.create_scenario_panel()
        self.create_command_buttons()
        
    def create_system_status(self):
        """Clean system status display with proper spacing"""
        ax = self.status_ax
        ax.clear()
        ax.set_xlim(0, 20)  # Much wider
        ax.set_ylim(0, 3)   # Taller
        ax.set_title('SYSTEM STATUS & READINESS', color='#00FF00', weight='bold', fontsize=14)
        ax.axis('off')
        
        # Status indicators with proper spacing
        status_items = [
            ('TRANSMITTER', f"{self.system_status['transmitter_power_kw']:.0f}kW", 
             '#00FF00' if self.system_status['transmitter_power_kw'] > 800 else '#FFAA00'),
            ('RECEIVER', f"{self.system_status['receiver_noise_db']:.1f}dB", 
             '#00FF00' if self.system_status['receiver_noise_db'] < -110 else '#FFAA00'),
            ('ANTENNA', f"{self.system_status['antenna_elevation_deg']:.1f}°", '#00FF00'),
            ('COOLING', f"{self.system_status['cooling_temp_c']:.0f}°C", 
             '#00FF00' if self.system_status['cooling_temp_c'] < 40 else '#FFAA00'),
            ('BITE STATUS', self.system_status['bite_status'], 
             '#00FF00' if self.system_status['bite_status'] == 'GO' else '#FF4444')
        ]
        
        for i, (label, value, color) in enumerate(status_items):
            x_pos = i * 3.5 + 2  # Much better spacing
            
            # Larger status boxes
            rect = Rectangle((x_pos-1.2, 1.5), 2.4, 1.0, 
                           facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            ax.text(x_pos, 2.2, label, ha='center', va='center', 
                   color='white', fontsize=10, weight='bold')
            ax.text(x_pos, 1.8, value, ha='center', va='center', 
                   color=color, fontsize=11, weight='bold')
        
        # System readiness (right side)
        readiness = "SYSTEM READY" if self.is_running else "STANDBY MODE"
        readiness_color = '#00FF00' if self.is_running else '#FFAA00'
        
        ax.text(18, 2.0, readiness, ha='center', va='center', 
               color=readiness_color, fontsize=14, weight='bold',
               bbox=dict(boxstyle='round', facecolor='black', edgecolor=readiness_color, linewidth=2))
        
    def create_radar_controls(self):
        """Clean radar parameter controls with proper internal spacing"""
        ax = self.radar_ax
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('RADAR PARAMETERS', color='#00FF00', weight='bold', fontsize=12)
        ax.axis('off')
        
        # Display current values with better spacing
        params_text = f"""CURRENT SETTINGS:

Max Range: {self.config.max_range_km:.0f} km

Sweep Rate: {self.config.sweep_rate_rpm:.0f} RPM

Sensitivity: {self.config.sensitivity_db:.0f} dB

Detection: {self.config.detection_threshold:.1f} dB"""
        
        ax.text(5, 8.5, params_text, ha='center', va='top',
               color='#00AAFF', fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                        alpha=0.9, edgecolor='#00AAFF'))
        
        # Control buttons with better spacing
        control_buttons = [
            (2.5, 3.5, 'RANGE\n+', self.increase_range),
            (2.5, 2.5, 'RANGE\n-', self.decrease_range),
            (7.5, 3.5, 'SENS\n+', self.increase_sensitivity),
            (7.5, 2.5, 'SENS\n-', self.decrease_sensitivity)
        ]
        
        for x, y, label, callback in control_buttons:
            rect = Rectangle((x-0.8, y-0.4), 1.6, 0.8, 
                           facecolor='#333333', edgecolor='#00FF00', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x, y, label, ha='center', va='center', 
                   color='#00FF00', fontsize=9, weight='bold')
            
    def create_mode_controls(self):
        """Clean mode controls with better internal spacing"""
        ax = self.mode_ax
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('RADAR MODE', color='#00FF00', weight='bold', fontsize=12)
        ax.axis('off')
        
        # Current mode display with better spacing
        mode_text = f"""CURRENT MODE:

{self.config.radar_mode.value.upper()}


TRACK STATUS:

Active: {self.performance_metrics['tracks_active']:3d}

Capacity: {self.performance_metrics['track_capacity']:3d}

Load: {(self.performance_metrics['tracks_active']/self.performance_metrics['track_capacity']*100):5.1f}%"""
        
        ax.text(5, 9, mode_text, ha='center', va='top',
               color='#00FF00', fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#0f0f0f', 
                        alpha=0.9, edgecolor='#00FF00'))
        
        # Mode selection buttons with better spacing
        modes = ['SEARCH', 'TRACK', 'TWS']
        for i, mode in enumerate(modes):
            x, y = 1.5 + i * 2.5, 2.5  # Better horizontal spacing
            is_active = mode == self.config.radar_mode.value.upper()
            color = '#00FF00' if is_active else '#666666'
            
            rect = Rectangle((x-0.8, y-0.5), 1.6, 1.0, 
                           facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, mode, ha='center', va='center', 
                   color=color, fontsize=10, weight='bold')
        
    def create_display_controls(self):
        """Clean display controls"""
        ax = self.display_ax
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('DISPLAY CONTROL', color='#00FF00', weight='bold')
        ax.axis('off')
        
        # Display settings
        display_text = f"""DISPLAY SETTINGS:

Range Scale: {self.config.range_scale_km:.0f} km
Trail Length: {self.config.trail_length_sec:.0f} sec

OPTIONS:
Range Rings: ON
Bearing Lines: ON  
Velocity Vectors: ON
Track History: ON

BRIGHTNESS: 85%
CONTRAST: 92%"""
        
        ax.text(5, 7, display_text, ha='center', va='top',
               color='#FFAA00', fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                        alpha=0.9, edgecolor='#FFAA00'))
        
        # Display control buttons
        display_buttons = [
            (2.5, 2, 'SCALE\n+', self.increase_scale),
            (7.5, 2, 'SCALE\n-', self.decrease_scale)
        ]
        
        for x, y, label, callback in display_buttons:
            rect = Rectangle((x-0.8, y-0.4), 1.6, 0.8, 
                           facecolor='#333333', edgecolor='#FFAA00', linewidth=1)
            ax.add_patch(rect)
            ax.text(x, y, label, ha='center', va='center') 
    def create_filter_controls(self):
        """Filter controls panel"""
        ax = self.filter_ax
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('SIGNAL FILTERS', color='#00FF00', weight='bold')
        ax.axis('off')
        
        filter_text = f"""ACTIVE FILTERS:

Clutter Map: {'ON' if self.config.clutter_map_enable else 'OFF'}
Weather Filter: {'ON' if self.config.weather_filter_enable else 'OFF'}

PROCESSING:
MTI: ENABLED
CFAR: ENABLED
Doppler: ENABLED"""
        
        ax.text(5, 7, filter_text, ha='center', va='top',
               color='#00AAFF', fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                        alpha=0.9, edgecolor='#00AAFF'))
        
        # Filter toggle buttons
        filter_buttons = [
            (2.5, 2, 'CLUTTER\nTOGGLE', self.toggle_clutter),
            (7.5, 2, 'WEATHER\nTOGGLE', self.toggle_weather)
        ]
        
        for x, y, label, callback in filter_buttons:
            rect = Rectangle((x-1, y-0.5), 2, 1, 
                           facecolor='#333333', edgecolor='#00AAFF', linewidth=1)
            ax.add_patch(rect)
            ax.text(x, y, label, ha='center', va='center', 
                   color='#00AAFF', fontsize=8, weight='bold')
                   
    def create_alert_panel(self):
        """Alert panel"""
        ax = self.alert_ax
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('ALERTS', color='#00FF00', weight='bold')
        ax.axis('off')
        
        if not self.alerts:
            alert_text = "NO ACTIVE ALERTS\n\nSYSTEM NOMINAL\n\nALL SUBSYSTEMS\nOPERATIONAL"
            alert_color = '#00FF00'
        else:
            recent_alerts = self.alerts[-3:] if len(self.alerts) >= 3 else self.alerts
            alert_text = "RECENT ALERTS:\n\n"
            for alert in reversed(recent_alerts):
                time_str = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M")
                alert_text += f"{time_str} {alert.level.value[:4]}\n{alert.message[:20]}...\n\n"
            alert_color = '#FFAA00'
        
        ax.text(5, 8, alert_text, ha='center', va='top',
               color=alert_color, fontsize=10, weight='bold',
               bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                        alpha=0.9, edgecolor=alert_color))
                        
    def create_performance_panel(self):
        """Performance panel"""
        ax = self.performance_ax
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('PERFORMANCE', color='#00FF00', weight='bold')
        ax.axis('off')
        
        perf_text = f"""SYSTEM METRICS:

CPU Load: {self.performance_metrics['processor_load_percent']:5.1f}%
Uptime: {self.performance_metrics['uptime_hours']:6.1f}h
Availability: {self.performance_metrics['availability_percent']:5.1f}%

TRACKING:
Active: {self.performance_metrics['tracks_active']:3d}
Capacity: {self.performance_metrics['track_capacity']:3d}"""
        
        ax.text(5, 7, perf_text, ha='center', va='top',
               color='#00FF00', fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#0f0f0f', 
                        alpha=0.9, edgecolor='#00FF00'))
                        
    def create_scenario_panel(self):
        """Scenario panel"""
        ax = self.scenario_ax
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('SCENARIOS', color='#00FF00', weight='bold')
        ax.axis('off')
        
        scenario_text = f"""AVAILABLE SCENARIOS:

• Airport Traffic
• Naval Operations  
• Weather Tracking
• Border Patrol
• Search & Rescue

CURRENT:
{getattr(self, 'current_scenario', 'None Selected')}"""
        
        ax.text(5, 7, scenario_text, ha='center', va='top',
               color='#FFAA00', fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                        alpha=0.9, edgecolor='#FFAA00'))
        
        # Scenario button
        rect = Rectangle((3, 1.5), 4, 1, 
                       facecolor='#333333', edgecolor='#FFAA00', linewidth=1)
        ax.add_patch(rect)
        ax.text(5, 2, 'LOAD NEXT\nSCENARIO', ha='center', va='center', 
               color='#FFAA00', fontsize=9, weight='bold')
                   
    def create_system_monitor(self):
        """Clean system monitor"""
        ax = self.monitor_ax
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('SYSTEM MONITOR', color='#00FF00', weight='bold')
        ax.axis('off')
        
        # Alert display
        if not self.alerts:
            alert_text = "NO ACTIVE ALERTS\nSYSTEM NOMINAL"
            alert_color = '#00FF00'
        else:
            recent_alert = self.alerts[-1]
            alert_text = f"LATEST ALERT:\n{recent_alert.level.value}\n{recent_alert.message[:30]}..."
            alert_color = '#FF4444' if recent_alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY] else '#FFAA00'
        
        ax.text(5, 7, alert_text, ha='center', va='center',
               color=alert_color, fontsize=10, weight='bold',
               bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                        alpha=0.9, edgecolor=alert_color))
        
        # System health indicators
        health_items = [
            ('PWR', '#00FF00'),
            ('RCV', '#00FF00'), 
            ('ANT', '#00FF00'),
            ('CPU', '#FFAA00' if self.performance_metrics['processor_load_percent'] > 80 else '#00FF00')
        ]
        
        for i, (label, color) in enumerate(health_items):
            x = 1.5 + i * 2
            y = 3
            
            circle = Circle((x, y), 0.3, facecolor=color, alpha=0.8, edgecolor='white')
            ax.add_patch(circle)
            ax.text(x, y-1, label, ha='center', va='center', 
                   color='white', fontsize=8, weight='bold')
        
    def create_command_buttons(self):
        """Clean command buttons with proper spacing"""
        ax = self.command_ax
        ax.clear()
        ax.set_xlim(0, 24)  # Much wider
        ax.set_ylim(0, 3)   # Taller
        ax.set_title('COMMAND & CONTROL', color='#00FF00', weight='bold', fontsize=14)
        ax.axis('off')
        
        # Professional command buttons with better spacing
        buttons = [
            (3, 'SYSTEM\nSTART', '#006600', self.start_system),
            (6, 'SYSTEM\nSTOP', '#666600', self.stop_system),
            (9, 'SYSTEM\nRESET', '#000066', self.reset_system),
            (12, 'ACK\nALERTS', '#006666', self.acknowledge_alerts),
            (15, 'BIT\nTEST', '#666666', self.built_in_test),
            (18, 'LOAD\nSCENARIO', '#006600', self.load_scenario),
            (21, 'EMERGENCY\nSTOP', '#660000', self.emergency_stop)
        ]
        
        for x_pos, label, color, callback in buttons:
            # Larger buttons
            rect = Rectangle((x_pos-1.2, 1), 2.4, 1.2, 
                           facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_pos, 1.6, label, ha='center', va='center', 
                   color='white', fontsize=10, weight='bold')
    
    # Add new control functions
    def toggle_clutter(self):
        """Toggle clutter filter"""
        self.config.clutter_map_enable = not self.config.clutter_map_enable
        status = "ENABLED" if self.config.clutter_map_enable else "DISABLED"
        self.add_alert(AlertLevel.ROUTINE, f"Clutter filter {status}", "Filter Control")
        
    def toggle_weather(self):
        """Toggle weather filter"""
        self.config.weather_filter_enable = not self.config.weather_filter_enable
        status = "ENABLED" if self.config.weather_filter_enable else "DISABLED"
        self.add_alert(AlertLevel.ROUTINE, f"Weather filter {status}", "Filter Control")
    
    # Control callback functions
    def register_callback(self, event: str, callback: Callable):
        """Register callback for system events"""
        self.callbacks[event] = callback
    
    def increase_range(self):
        """Increase radar range"""
        self.config.max_range_km = min(500, self.config.max_range_km + 25)
        self.add_alert(AlertLevel.ROUTINE, f"Range increased to {self.config.max_range_km:.0f} km", "Range Control")
        if 'range_change' in self.callbacks:
            self.callbacks['range_change'](self.config.max_range_km)
            
    def decrease_range(self):
        """Decrease radar range"""
        self.config.max_range_km = max(50, self.config.max_range_km - 25)
        self.add_alert(AlertLevel.ROUTINE, f"Range decreased to {self.config.max_range_km:.0f} km", "Range Control")
        if 'range_change' in self.callbacks:
            self.callbacks['range_change'](self.config.max_range_km)
            
    def increase_sensitivity(self):
        """Increase sensitivity"""
        self.config.sensitivity_db = min(-90, self.config.sensitivity_db + 5)
        self.add_alert(AlertLevel.ROUTINE, f"Sensitivity: {self.config.sensitivity_db:.0f} dB", "Sensitivity Control")
        
    def decrease_sensitivity(self):
        """Decrease sensitivity"""
        self.config.sensitivity_db = max(-120, self.config.sensitivity_db - 5)
        self.add_alert(AlertLevel.ROUTINE, f"Sensitivity: {self.config.sensitivity_db:.0f} dB", "Sensitivity Control")
        
    def increase_scale(self):
        """Increase display scale"""
        self.config.range_scale_km = min(250, self.config.range_scale_km + 25)
        if 'range_scale_change' in self.callbacks:
            self.callbacks['range_scale_change'](self.config.range_scale_km)
            
    def decrease_scale(self):
        """Decrease display scale"""
        self.config.range_scale_km = max(25, self.config.range_scale_km - 25)
        if 'range_scale_change' in self.callbacks:
            self.callbacks['range_scale_change'](self.config.range_scale_km)
    
    def start_system(self):
        """Start radar system"""
        if self.emergency_stop_active:
            self.add_alert(AlertLevel.WARNING, "Cannot start - Emergency stop active", "System Control")
            return
            
        self.is_running = True
        self.emergency_stop_active = False
        self.system_status.update({
            'transmitter_power_kw': 850.0,
            'receiver_noise_db': -113.0,
            'cooling_temp_c': 25.0
        })
        
        self.add_alert(AlertLevel.ROUTINE, "Radar system OPERATIONAL", "System Control")
        if 'system_start' in self.callbacks:
            self.callbacks['system_start']()
            
    def stop_system(self):
        """Stop radar system"""
        self.is_running = False
        self.system_status.update({
            'transmitter_power_kw': 0.0
        })
        self.add_alert(AlertLevel.CAUTION, "Radar system SHUTDOWN", "System Control")
        if 'system_stop' in self.callbacks:
            self.callbacks['system_stop']()
            
    def reset_system(self):
        """Reset system"""
        self.stop_system()
        self.config = RadarConfiguration()
        self.alerts.clear()
        self.add_alert(AlertLevel.WARNING, "System RESET completed", "System Control")
        if 'system_reset' in self.callbacks:
            self.callbacks['system_reset']()
            
    def emergency_stop(self):
        """Emergency stop"""
        self.emergency_stop_active = True
        self.is_running = False
        self.system_status.update({
            'transmitter_power_kw': 0.0,
            'bite_status': 'E-STOP'
        })
        self.add_alert(AlertLevel.EMERGENCY, "EMERGENCY STOP ACTIVATED", "Emergency System")
        if 'emergency_stop' in self.callbacks:
            self.callbacks['emergency_stop']()
            
    def acknowledge_alerts(self):
        """Acknowledge alerts"""
        for alert in self.alerts:
            alert.acknowledged = True
        self.add_alert(AlertLevel.ROUTINE, "All alerts acknowledged", "Alert System")
        
    def built_in_test(self):
        """Run BIT test"""
        self.add_alert(AlertLevel.ROUTINE, "Running BIT diagnostics...", "BIT System")
        import random
        if random.random() > 0.8:
            self.add_alert(AlertLevel.WARNING, "BIT: Transmitter marginal", "BIT System")
        else:
            self.add_alert(AlertLevel.ROUTINE, "BIT: All systems PASS", "BIT System")
            
    def load_scenario(self):
        """Load scenario"""
        scenarios = ['Airport Traffic', 'Naval Operations', 'Weather Tracking']
        scenario = scenarios[getattr(self, 'scenario_index', 0) % len(scenarios)]
        self.scenario_index = getattr(self, 'scenario_index', 0) + 1
        self.current_scenario = scenario  # Store current scenario
        
        self.add_alert(AlertLevel.ROUTINE, f"Loading: {scenario}", "Scenario Manager")
        if 'scenario_change' in self.callbacks:
            self.callbacks['scenario_change'](scenario.lower().replace(' ', '_'))
    
    def toggle_clutter(self):
        """Toggle clutter filter"""
        self.config.clutter_map_enable = not self.config.clutter_map_enable
        status = "ENABLED" if self.config.clutter_map_enable else "DISABLED"
        self.add_alert(AlertLevel.ROUTINE, f"Clutter filter {status}", "Filter Control")
        
    def toggle_weather(self):
        """Toggle weather filter"""
        self.config.weather_filter_enable = not self.config.weather_filter_enable
        status = "ENABLED" if self.config.weather_filter_enable else "DISABLED"
        self.add_alert(AlertLevel.ROUTINE, f"Weather filter {status}", "Filter Control")
    
    def add_alert(self, level: AlertLevel, message: str, source: str):
        """Add system alert"""
        alert = SystemAlert(
            timestamp=time.time(),
            level=level,
            message=message,
            source=source
        )
        self.alerts.append(alert)
        
        # Keep only last 50 alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
            
    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics"""
        self.performance_metrics.update(metrics)
        
    def start_update_timer(self):
        """Start update timer"""
        self.update_timer = self.fig.canvas.new_timer(interval=2000)  # 2 seconds
        self.update_timer.add_callback(self.update_display)
        self.update_timer.start()
        
    def update_display(self):
        """Update display"""
        # Simulate parameter changes
        import random
        if self.is_running:
            self.system_status['cooling_temp_c'] += random.uniform(-0.5, 0.5)
            self.system_status['cooling_temp_c'] = max(20, min(50, self.system_status['cooling_temp_c']))
            
            self.performance_metrics['processor_load_percent'] += random.uniform(-5, 5)
            self.performance_metrics['processor_load_percent'] = max(0, min(100, 
                self.performance_metrics['processor_load_percent']))
        
        # Refresh all panels
        self.create_system_status()
        self.create_radar_controls()
        self.create_mode_controls()
        self.create_display_controls()
        self.create_system_monitor()
        self.create_filter_controls()
        self.create_alert_panel()
        self.create_performance_panel()
        self.create_scenario_panel()
        
        self.fig.canvas.draw_idle()
    
    def get_configuration(self) -> RadarConfiguration:
        """Get current configuration"""
        return self.config
        
    def run(self):
        """Run the control system"""
        # Much better spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, 
                           hspace=0.4, wspace=0.3)
        plt.show()

def demo_clean_controls():
    """Demo the clean control system"""
    print("🎯 CLEAN PROFESSIONAL RADAR CONTROLS")
    print("=" * 40)
    
    config = RadarConfiguration(
        max_range_km=250.0,
        sweep_rate_rpm=15.0,
        radar_mode=RadarMode.SEARCH
    )
    
    control_system = ProfessionalRadarControls(config)
    
    # Register callbacks
    def on_range_change(range_val):
        print(f"📡 Range changed to: {range_val:.0f} km")
        
    def on_system_start():
        print("🚀 RADAR SYSTEM STARTED")
        control_system.update_performance_metrics({'tracks_active': 8})
        
    control_system.register_callback('range_change', on_range_change)
    control_system.register_callback('system_start', on_system_start)
    
    # Add initial alerts
    control_system.add_alert(AlertLevel.ROUTINE, "System initialized", "Startup")
    control_system.add_alert(AlertLevel.ROUTINE, "All subsystems nominal", "Health Check")
    
    print("✅ Clean control system ready")
    print("   Click buttons to control radar parameters")
    print("   All changes update the radar system in real-time")
    
    control_system.run()

if __name__ == "__main__":
    demo_clean_controls()