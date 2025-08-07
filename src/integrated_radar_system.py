"""
Integrated radar system combining data generation and display
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.radar_data_generator import RadarDataGenerator, EnvironmentType
from src.radar_display import RadarDisplay

class IntegratedRadarSystem:
    def __init__(self, max_range_km=200):
        self.data_generator = RadarDataGenerator(max_range_km)
        self.display = RadarDisplay(max_range_km)
        self.current_sweep_angle = 0
        self.sweep_speed = 6  # degrees per frame
        self.detected_targets = []
        self.target_history = []  # For trails
        
    def load_scenario(self, scenario_name):
        """Load a pre-defined scenario"""
        self.data_generator.create_scenario(scenario_name)
        print(f"Loaded scenario: {scenario_name}")
        print(f"Generated {len(self.data_generator.targets)} targets")
        
    def update_system(self, frame):
        """Update the entire radar system (called by animation)"""
        #Update target positions
        self.data_generator.update_targets(time_step_seconds=1.0)
        
        #Update sweep angle
        self.current_sweep_angle = (self.current_sweep_angle + self.sweep_speed) % 360
        
        #Detect targets in current sweep
        new_detections = self.data_generator.simulate_radar_detection(
            self.current_sweep_angle, sweep_width_deg=15
        )
        
        self.detected_targets.extend(new_detections)
        
        #Keep only recent detections (last 30 seconds)
        current_time = self.data_generator.time_elapsed
        self.detected_targets = [
            d for d in self.detected_targets 
            if current_time - d['detection_time'] <= 30
        ]
        
        self.update_display()
        
        return []
    
    def update_display(self):
        """Update the radar display with current detections"""
        #clear previous plots
        self.display.ax.clear()
        self.display.setup_appearance()
        
        #plot detected targets with age-based fading
        current_time = self.data_generator.time_elapsed
        
        for detection in self.detected_targets:
            age_seconds = current_time - detection['detection_time']
            alpha = max(0.1, 1.0 - age_seconds / 30.0)  #fade over 30 seconds
            
            #convert to polar coordinates for display
            range_km = detection['range']
            bearing_rad = np.radians(detection['bearing'])
            
            if detection.get('is_false_alarm'):
                color = 'yellow'
                marker = 'x'
            elif detection['target'] and detection['target'].target_type.value == 'aircraft':
                color = 'red'
                marker = 'o'
            elif detection['target'] and detection['target'].target_type.value == 'ship':
                color = 'blue'
                marker = 's'
            elif detection['target'] and detection['target'].target_type.value == 'weather':
                color = 'green'
                marker = '*'
            else:
                color = 'white'
                marker = 'o'
            
            self.display.ax.plot(bearing_rad, range_km, marker, 
                               color=color, markersize=6, alpha=alpha)
        
        #draw current sweep line
        sweep_rad = np.radians(self.current_sweep_angle)
        self.display.ax.plot([sweep_rad, sweep_rad], [0, self.display.max_range], 
                           'yellow', linewidth=2, alpha=0.8)
        
        num_real_targets = len([d for d in self.detected_targets if not d.get('is_false_alarm')])
        num_false_alarms = len([d for d in self.detected_targets if d.get('is_false_alarm')])
        
        status_text = f"Targets: {num_real_targets} | False Alarms: {num_false_alarms}"
        status_text += f" | Environment: {self.data_generator.environment.value.title()}"
        
        self.display.ax.text(np.radians(0), self.display.max_range * 1.15, status_text,
                           ha='center', va='center', color='lime', fontsize=10)
    
    def run_simulation(self, scenario_name="busy_airport"):
        """Run the integrated radar simulation"""
        self.load_scenario(scenario_name)
        
        self.animation = animation.FuncAnimation(
            self.display.fig, self.update_system, 
            interval=100, blit=False, cache_frame_data=False
        )
        
        plt.show()

def demo_integrated_system():
    """Demonstrate the integrated radar system"""
    print("Integrated Radar System Demo")
    print("===========================")
    print()
    print("Available scenarios:")
    print("1. busy_airport - Commercial aviation traffic")
    print("2. naval_operations - Carrier group with escorts")  
    print("3. storm_tracking - Weather avoidance scenario")
    print()
    
    choice = input("Choose scenario (1-3) or press Enter for busy_airport: ").strip()
    
    scenarios = {
        "1": "busy_airport",
        "2": "naval_operations", 
        "3": "storm_tracking",
        "": "busy_airport"
    }
    
    scenario = scenarios.get(choice, "busy_airport")
    
    print(f"\nStarting {scenario} scenario...")
    print("Close the window to stop the simulation.")
    
    radar_system = IntegratedRadarSystem(max_range_km=250)
    radar_system.run_simulation(scenario)

if __name__ == "__main__":
    demo_integrated_system()