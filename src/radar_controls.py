"""
Advanced Radar Control System - Matplotlib Version
Professional operator interface for radar system management and configuration.
Compatible with existing matplotlib-based radar display system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons, RadioButtons
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import Dict, List, Callable, Any
from enum import Enum
import threading
import time
import json
from datetime import datetime

class RadarMode(Enum):
    SEARCH = "Search"
    TRACK = "Track"
    STANDBY = "Standby"
    CALIBRATION = "Calibration"

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass
class RadarConfiguration:
    """Radar system configuration parameters"""
    max_range: float = 100.0  # km
    scan_rate: float = 6.0    # RPM
    sensitivity: float = 0.7  # 0.0 - 1.0
    clutter_rejection: bool = True
    weather_filter: bool = True
    track_confirmation: int = 3  # detections needed
    mode: RadarMode = RadarMode.SEARCH
    auto_track: bool = True
    alert_threshold: float = 0.8  # threat level

@dataclass
class SystemAlert:
    """System alert/notification"""
    timestamp: float
    level: AlertLevel
    message: str
    source: str

class RadarControlPanel:
    """Professional radar control interface using matplotlib"""
    
    def __init__(self, config: RadarConfiguration = None):
        self.config = config or RadarConfiguration()
        self.alerts: List[SystemAlert] = []
        self.callbacks: Dict[str, Callable] = {}
        self.system_status = {
            'power': True,
            'transmitter': True,
            'receiver': True,
            'processor': True,
            'display': True,
            'last_update': time.time()
        }
        
        # Performance metrics
        self.performance_metrics = {
            'targets_tracked': 0,
            'detection_rate': 95.5,
            'false_alarm_rate': 0.02,
            'cpu_usage': 45.2,
            'memory_usage': 67.8,
            'uptime': 0,
            'start_time': time.time()
        }
        
        # Control states
        self.is_running = False
        self.emergency_stop_active = False
        
        self.setup_gui()
        
    def setup_gui(self):
        """Initialize the control panel GUI"""
        # Create figure with professional dark theme
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('RADAR CONTROL SYSTEM - PROFESSIONAL INTERFACE', 
                         fontsize=16, color='#00ff00', weight='bold')
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Create grid layout for panels
        gs = self.fig.add_gridspec(4, 4, height_ratios=[1, 2, 2, 1], width_ratios=[1, 1, 1, 1])
        
        # System status panel (top)
        self.status_ax = self.fig.add_subplot(gs[0, :])
        
        # Control panels
        self.radar_controls_ax = self.fig.add_subplot(gs[1, :2])
        self.scenario_controls_ax = self.fig.add_subplot(gs[2, :2])
        
        # Performance and alerts
        self.performance_ax = self.fig.add_subplot(gs[1, 2:])
        self.alerts_ax = self.fig.add_subplot(gs[2, 2:])
        
        # Button panel (bottom)
        self.button_ax = self.fig.add_subplot(gs[3, :])
        
        self.create_all_panels()
        self.start_update_timer()
        
    def create_all_panels(self):
        """Create all control panels"""
        self.create_system_status()
        self.create_radar_controls()
        self.create_scenario_controls()
        self.create_performance_monitor()
        self.create_alert_system()
        self.create_system_buttons()
        
    def create_system_status(self):
        """System status display"""
        self.status_ax.clear()
        self.status_ax.set_xlim(0, 10)
        self.status_ax.set_ylim(0, 2)
        self.status_ax.set_title('SYSTEM STATUS', color='#00ff00', weight='bold')
        self.status_ax.axis('off')
        
        # Status indicators
        status_items = ['Power', 'Transmitter', 'Receiver', 'Processor', 'Display']
        colors = ['#00ff00' if self.system_status.get(item.lower(), True) else '#ff4444' 
                 for item in status_items]
        
        for i, (item, color) in enumerate(zip(status_items, colors)):
            x_pos = i * 2 + 1
            # Status indicator circle
            circle = Circle((x_pos, 1.5), 0.15, color=color, alpha=0.8)
            self.status_ax.add_patch(circle)
            # Label
            self.status_ax.text(x_pos, 1.0, item, ha='center', va='center', 
                              color='white', fontsize=10, weight='bold')
            # Status text
            status_text = "ONLINE" if color == '#00ff00' else "OFFLINE"
            self.status_ax.text(x_pos, 0.6, status_text, ha='center', va='center', 
                              color=color, fontsize=8, weight='bold')
        
        # Current mode display
        mode_text = f"MODE: {self.config.mode.value.upper()}"
        self.status_ax.text(9, 1.5, mode_text, ha='center', va='center', 
                          color='#ffaa00', fontsize=12, weight='bold',
                          bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
        
    def create_radar_controls(self):
        """Radar parameter controls"""
        self.radar_controls_ax.clear()
        self.radar_controls_ax.set_xlim(0, 10)
        self.radar_controls_ax.set_ylim(0, 10)
        self.radar_controls_ax.set_title('RADAR PARAMETERS', color='#00ff00', weight='bold')
        self.radar_controls_ax.axis('off')
        
        # Create sliders for radar parameters
        slider_height = 0.4
        slider_width = 8
        
        # Range slider
        range_ax = plt.axes([0.12, 0.65, 0.35, 0.03])
        self.range_slider = Slider(range_ax, 'Range (km)', 10, 200, 
                                  valinit=self.config.max_range, 
                                  valfmt='%.0f km', color='#00ff00')
        self.range_slider.on_changed(self.on_range_change)
        
        # Sensitivity slider
        sens_ax = plt.axes([0.12, 0.60, 0.35, 0.03])
        self.sens_slider = Slider(sens_ax, 'Sensitivity', 0.1, 1.0, 
                                 valinit=self.config.sensitivity, 
                                 valfmt='%.2f', color='#00ff00')
        self.sens_slider.on_changed(self.on_sensitivity_change)
        
        # Scan rate slider
        scan_ax = plt.axes([0.12, 0.55, 0.35, 0.03])
        self.scan_slider = Slider(scan_ax, 'Scan Rate (RPM)', 1, 12, 
                                 valinit=self.config.scan_rate, 
                                 valfmt='%.1f RPM', color='#00ff00')
        self.scan_slider.on_changed(self.on_scan_rate_change)
        
        # Mode selection
        mode_ax = plt.axes([0.12, 0.48, 0.15, 0.15])
        mode_labels = [mode.value for mode in RadarMode]
        self.mode_radio = RadioButtons(mode_ax, mode_labels)
        self.mode_radio.set_active([mode.value for mode in RadarMode].index(self.config.mode.value))
        self.mode_radio.on_clicked(self.on_mode_change)
        
        # Filter checkboxes
        filter_ax = plt.axes([0.30, 0.48, 0.15, 0.15])
        filter_labels = ['Clutter Reject', 'Weather Filter', 'Auto Track']
        self.filter_checks = CheckButtons(filter_ax, filter_labels, 
                                        [self.config.clutter_rejection,
                                         self.config.weather_filter,
                                         self.config.auto_track])
        self.filter_checks.on_clicked(self.on_filter_change)
        
    def create_scenario_controls(self):
        """Scenario selection and management"""
        self.scenario_controls_ax.clear()
        self.scenario_controls_ax.set_xlim(0, 10)
        self.scenario_controls_ax.set_ylim(0, 10)
        self.scenario_controls_ax.set_title('SCENARIO MANAGEMENT', color='#00ff00', weight='bold')
        self.scenario_controls_ax.axis('off')
        
        # Scenario selection
        scenario_ax = plt.axes([0.12, 0.32, 0.15, 0.12])
        scenario_labels = ['Airport Traffic', 'Naval Ops', 'Weather Mon', 'Custom']
        self.scenario_radio = RadioButtons(scenario_ax, scenario_labels)
        self.scenario_radio.on_clicked(self.on_scenario_change)
        
        # Status text area
        status_text = f"""CURRENT SCENARIO: Airport Traffic
TARGETS ACTIVE: {self.performance_metrics['targets_tracked']}
DETECTION RATE: {self.performance_metrics['detection_rate']:.1f}%
SYSTEM UPTIME: {self.get_uptime_string()}"""
        
        self.scenario_controls_ax.text(5.5, 7, status_text, ha='left', va='top',
                                     color='#ffffff', fontsize=10, family='monospace',
                                     bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                                             alpha=0.8, edgecolor='#00ff00'))
        
    def create_performance_monitor(self):
        """Performance monitoring display"""
        self.performance_ax.clear()
        self.performance_ax.set_xlim(0, 10)
        self.performance_ax.set_ylim(0, 10)
        self.performance_ax.set_title('PERFORMANCE MONITOR', color='#00ff00', weight='bold')
        self.performance_ax.axis('off')
        
        # Performance metrics display
        metrics_text = f"""SYSTEM PERFORMANCE

CPU Usage:      {self.performance_metrics['cpu_usage']:6.1f}%
Memory Usage:   {self.performance_metrics['memory_usage']:6.1f}%
Detection Rate: {self.performance_metrics['detection_rate']:6.1f}%
False Alarms:   {self.performance_metrics['false_alarm_rate']:6.3f}%
Targets:        {self.performance_metrics['targets_tracked']:6d}
Uptime:         {self.get_uptime_string()}"""
        
        self.performance_ax.text(1, 8, metrics_text, ha='left', va='top',
                               color='#00ff00', fontsize=11, family='monospace',
                               bbox=dict(boxstyle='round', facecolor='#0f0f0f', 
                                       alpha=0.9, edgecolor='#00ff00'))
        
        # Performance bars
        metrics = [
            ('CPU', self.performance_metrics['cpu_usage'], 100),
            ('MEM', self.performance_metrics['memory_usage'], 100),
            ('DET', self.performance_metrics['detection_rate'], 100)
        ]
        
        for i, (label, value, max_val) in enumerate(metrics):
            y_pos = 4 - i * 0.8
            # Background bar
            bg_rect = Rectangle((1, y_pos), 8, 0.4, facecolor='#333333', alpha=0.5)
            self.performance_ax.add_patch(bg_rect)
            # Value bar
            bar_width = (value / max_val) * 8
            color = '#00ff00' if value < 80 else '#ffaa00' if value < 95 else '#ff4444'
            val_rect = Rectangle((1, y_pos), bar_width, 0.4, facecolor=color, alpha=0.8)
            self.performance_ax.add_patch(val_rect)
            # Label and value
            self.performance_ax.text(0.5, y_pos + 0.2, label, ha='right', va='center',
                                   color='white', fontsize=10, weight='bold')
            self.performance_ax.text(9.5, y_pos + 0.2, f'{value:.1f}%', ha='left', va='center',
                                   color='white', fontsize=10, weight='bold')
        
    def create_alert_system(self):
        """Alert and message system"""
        self.alerts_ax.clear()
        self.alerts_ax.set_xlim(0, 10)
        self.alerts_ax.set_ylim(0, 10)
        self.alerts_ax.set_title('SYSTEM ALERTS', color='#00ff00', weight='bold')
        self.alerts_ax.axis('off')
        
        # Display recent alerts
        recent_alerts = self.alerts[-8:] if len(self.alerts) > 8 else self.alerts
        
        if not recent_alerts:
            self.alerts_ax.text(5, 5, 'NO ACTIVE ALERTS\nSYSTEM NOMINAL', 
                              ha='center', va='center', color='#00ff00', 
                              fontsize=12, weight='bold')
        else:
            alert_text = ""
            for i, alert in enumerate(reversed(recent_alerts)):
                time_str = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
                color_code = {'INFO': '■', 'WARNING': '▲', 'CRITICAL': '●'}[alert.level.value]
                alert_line = f"{color_code} {time_str} {alert.level.value}: {alert.message}\n"
                alert_text += alert_line
                
            self.alerts_ax.text(0.5, 9, alert_text, ha='left', va='top',
                              color='#ffffff', fontsize=9, family='monospace',
                              bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                                      alpha=0.9, edgecolor='#ff4444'))
        
    def create_system_buttons(self):
        """Create system control buttons"""
        self.button_ax.clear()
        self.button_ax.set_xlim(0, 10)
        self.button_ax.set_ylim(0, 2)
        self.button_ax.set_title('SYSTEM CONTROL', color='#00ff00', weight='bold')
        self.button_ax.axis('off')
        
        # Control buttons
        button_specs = [
            (1, 'START', '#00aa00', self.start_system),
            (2.5, 'STOP', '#aa6600', self.stop_system),
            (4, 'RESET', '#0066aa', self.reset_system),
            (6, 'CLEAR ALERTS', '#666666', self.clear_alerts),
            (8, 'EMERGENCY', '#aa0000', self.emergency_stop)
        ]
        
        self.control_buttons = []
        for x_pos, label, color, callback in button_specs:
            btn_ax = plt.axes([x_pos/10 - 0.06, 0.02, 0.12, 0.06])
            button = Button(btn_ax, label, color=color, hovercolor=color)
            button.on_clicked(lambda event, cb=callback: cb())
            button.label.set_color('white')
            button.label.set_weight('bold')
            self.control_buttons.append(button)
        
    def register_callback(self, event: str, callback: Callable):
        """Register callback for system events"""
        self.callbacks[event] = callback
        
    def on_mode_change(self, label):
        """Handle radar mode change"""
        new_mode = RadarMode(label)
        self.config.mode = new_mode
        self.add_alert(AlertLevel.INFO, f"Mode: {new_mode.value}", "Mode Control")
        if 'mode_change' in self.callbacks:
            self.callbacks['mode_change'](new_mode)
            
    def on_range_change(self, value):
        """Handle range change"""
        self.config.max_range = float(value)
        if 'range_change' in self.callbacks:
            self.callbacks['range_change'](self.config.max_range)
            
    def on_sensitivity_change(self, value):
        """Handle sensitivity change"""
        self.config.sensitivity = float(value)
        if 'sensitivity_change' in self.callbacks:
            self.callbacks['sensitivity_change'](self.config.sensitivity)
            
    def on_scan_rate_change(self, value):
        """Handle scan rate change"""
        self.config.scan_rate = float(value)
        if 'scan_rate_change' in self.callbacks:
            self.callbacks['scan_rate_change'](self.config.scan_rate)
            
    def on_filter_change(self, label):
        """Handle filter setting changes"""
        filter_map = {
            'Clutter Reject': 'clutter_rejection',
            'Weather Filter': 'weather_filter', 
            'Auto Track': 'auto_track'
        }
        
        if label in filter_map:
            attr = filter_map[label]
            current_val = getattr(self.config, attr)
            setattr(self.config, attr, not current_val)
            status = "ON" if not current_val else "OFF"
            self.add_alert(AlertLevel.INFO, f"{label}: {status}", "Filter Control")
            
        if 'filter_change' in self.callbacks:
            self.callbacks['filter_change'](self.config)
            
    def on_scenario_change(self, label):
        """Handle scenario change"""
        scenario_map = {
            'Airport Traffic': 'airport',
            'Naval Ops': 'naval',
            'Weather Mon': 'weather',
            'Custom': 'custom'
        }
        scenario = scenario_map.get(label, 'airport')
        self.add_alert(AlertLevel.INFO, f"Scenario: {label}", "Scenario Manager")
        if 'scenario_change' in self.callbacks:
            self.callbacks['scenario_change'](scenario)
            
    def start_system(self):
        """Start radar system"""
        self.is_running = True
        self.emergency_stop_active = False
        self.update_system_status({
            'transmitter': True,
            'receiver': True,
            'processor': True,
            'display': True
        })
        self.add_alert(AlertLevel.INFO, "System STARTED", "System Control")
        if 'system_start' in self.callbacks:
            self.callbacks['system_start']()
            
    def stop_system(self):
        """Stop radar system"""
        self.is_running = False
        self.add_alert(AlertLevel.WARNING, "System STOPPED", "System Control")
        if 'system_stop' in self.callbacks:
            self.callbacks['system_stop']()
            
    def reset_system(self):
        """Reset radar system"""
        self.is_running = False
        self.emergency_stop_active = False
        self.performance_metrics['targets_tracked'] = 0
        self.performance_metrics['start_time'] = time.time()
        self.add_alert(AlertLevel.WARNING, "System RESET", "System Control")
        if 'system_reset' in self.callbacks:
            self.callbacks['system_reset']()
            
    def emergency_stop(self):
        """Emergency system stop"""
        self.emergency_stop_active = True
        self.is_running = False
        self.update_system_status({
            'transmitter': False,
            'receiver': False,
            'processor': False
        })
        self.add_alert(AlertLevel.CRITICAL, "EMERGENCY STOP", "Emergency System")
        if 'emergency_stop' in self.callbacks:
            self.callbacks['emergency_stop']()
            
    def clear_alerts(self):
        """Clear alert display"""
        self.alerts.clear()
        self.add_alert(AlertLevel.INFO, "Alerts cleared", "Control Panel")
        
    def add_alert(self, level: AlertLevel, message: str, source: str):
        """Add system alert"""
        timestamp = time.time()
        alert = SystemAlert(timestamp, level, message, source)
        self.alerts.append(alert)
        
        # Keep only last 50 alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
            
    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics display"""
        self.performance_metrics.update(metrics)
        
    def update_system_status(self, status: Dict[str, bool]):
        """Update system status indicators"""
        self.system_status.update(status)
        self.system_status['last_update'] = time.time()
        
    def get_uptime_string(self):
        """Get formatted uptime string"""
        uptime = time.time() - self.performance_metrics['start_time']
        hours = int(uptime // 3600)
        mins = int((uptime % 3600) // 60)
        secs = int(uptime % 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
        
    def start_update_timer(self):
        """Start periodic updates"""
        self.update_timer = self.fig.canvas.new_timer(interval=1000)  # 1 second
        self.update_timer.add_callback(self.update_display)
        self.update_timer.start()
        
    def update_display(self):
        """Update all display panels"""
        # Simulate some metric changes
        import random
        self.performance_metrics['cpu_usage'] = max(10, min(95, 
            self.performance_metrics['cpu_usage'] + random.uniform(-3, 3)))
        self.performance_metrics['memory_usage'] = max(30, min(90,
            self.performance_metrics['memory_usage'] + random.uniform(-2, 2)))
        
        # Refresh panels
        self.create_system_status()
        self.create_scenario_controls()
        self.create_performance_monitor()
        self.create_alert_system()
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def get_configuration(self) -> RadarConfiguration:
        """Get current radar configuration"""
        return self.config
        
    def run(self):
        """Run the control panel"""
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.15)
        plt.show()

# Demo function for testing
def demo_radar_controls():
    """Demonstration of radar control system"""
    config = RadarConfiguration(
        max_range=150.0,
        scan_rate=8.0,
        sensitivity=0.8,
        mode=RadarMode.SEARCH
    )
    
    control_panel = RadarControlPanel(config)
    
    # Register some demo callbacks
    def on_mode_change(mode):
        print(f"🎯 Mode changed to: {mode.value}")
        
    def on_range_change(range_val):
        print(f"📡 Range changed to: {range_val} km")
        
    def on_system_start():
        print("🚀 System started!")
        control_panel.add_alert(AlertLevel.INFO, "All systems operational", "System Check")
        control_panel.update_performance_metrics({'targets_tracked': 5})
        
    def on_emergency_stop():
        print("🛑 EMERGENCY STOP!")
        control_panel.update_system_status({
            'transmitter': False,
            'receiver': False,
            'processor': False
        })
        
    control_panel.register_callback('mode_change', on_mode_change)
    control_panel.register_callback('range_change', on_range_change)
    control_panel.register_callback('system_start', on_system_start)
    control_panel.register_callback('emergency_stop', on_emergency_stop)
    
    # Add some demo alerts
    control_panel.add_alert(AlertLevel.INFO, "Control system ready", "Initialization")
    control_panel.add_alert(AlertLevel.WARNING, "High CPU usage detected", "Performance Monitor")
    control_panel.add_alert(AlertLevel.INFO, "Scenario loaded: Airport Traffic", "Scenario Manager")
    
    print("🎯 Radar Control System Demo - Matplotlib Version")
    
    control_panel.run()

if __name__ == "__main__":
    demo_radar_controls()