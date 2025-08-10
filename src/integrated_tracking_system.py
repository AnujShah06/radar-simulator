"""
Complete integrated radar tracking system
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List
import time

# Import our components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.radar_data_generator import RadarDataGenerator, EnvironmentType
from src.signal_processing import SignalProcessor
from src.target_detection import TargetDetector
from src.multi_target_tracker import MultiTargetTracker, Track

class IntegratedTrackingSystem:
    """Complete radar system with tracking"""
    
    def __init__(self, max_range_km=200):
        self.max_range_km = max_range_km
        
        # Core components
        self.data_generator = RadarDataGenerator(max_range_km)
        self.signal_processor = SignalProcessor()
        self.target_detector = TargetDetector()
        self.tracker = MultiTargetTracker()
        
        # System state
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.sweep_rate = 6.0  # degrees per update
        
        # Performance metrics
        self.processing_times = []
        self.detection_counts = []
        self.track_counts = []
        
    def load_scenario(self, scenario_name: str):
        """Load a radar scenario"""
        self.data_generator.create_scenario(scenario_name)
        print(f"Loaded scenario: {scenario_name}")
        print(f"Generated {len(self.data_generator.targets)} targets")
    
    def update_system(self, time_step=1.0) -> Dict:
        """Update the complete radar system"""
        start_time = time.time()
        
        # Update simulation time
        self.current_time += time_step
        
        # Update radar data (target movement)
        self.data_generator.update_targets(time_step)
        
        # Update sweep angle
        self.sweep_angle = (self.sweep_angle + self.sweep_rate) % 360
        
        # Simulate radar detection 
        raw_detections = self.data_generator.simulate_radar_detection(
            self.sweep_angle, sweep_width_deg=60  # Much wider beam
        )
        
        # Process detections through signal processing and detection pipeline
        detected_targets = self.target_detector.process_raw_detections(raw_detections)
        
        # Update tracker
        active_tracks = self.tracker.update(detected_targets, self.current_time)
        
        # Record performance metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.detection_counts.append(len(detected_targets))
        self.track_counts.append(len([t for t in active_tracks.values() if t.confirmed]))
        
        # Keep only recent metrics
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
            self.detection_counts = self.detection_counts[-100:]
            self.track_counts = self.track_counts[-100:]
        
        return {
            'raw_detections': raw_detections,
            'processed_detections': detected_targets,
            'active_tracks': active_tracks,
            'confirmed_tracks': [t for t in active_tracks.values() if t.confirmed],
            'sweep_angle': self.sweep_angle,
            'processing_time': processing_time
        }
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        tracker_stats = self.tracker.get_tracking_statistics()
        
        status = {
            'current_time': self.current_time,
            'sweep_angle': self.sweep_angle,
            'targets_in_scenario': len(self.data_generator.targets),
            'processing_stats': {
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
                'max_processing_time': np.max(self.processing_times) if self.processing_times else 0,
                'avg_detections': np.mean(self.detection_counts) if self.detection_counts else 0,
                'avg_tracks': np.mean(self.track_counts) if self.track_counts else 0
            },
            'tracking_stats': tracker_stats
        }
        
        return status
    
    def run_batch_simulation(self, duration_seconds: int = 60, time_step: float = 1.0):
        """Run simulation for specified duration"""
        print(f"Running batch simulation for {duration_seconds} seconds...")
        
        results = {
            'times': [],
            'detections': [],
            'tracks': [],
            'processing_times': []
        }
        
        steps = int(duration_seconds / time_step)
        for step in range(steps):
            result = self.update_system(time_step)
            
            results['times'].append(self.current_time)
            results['detections'].append(len(result['processed_detections']))
            results['tracks'].append(len(result['confirmed_tracks']))
            results['processing_times'].append(result['processing_time'])
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{steps}: {len(result['confirmed_tracks'])} tracks, "
                      f"{result['processing_time']*1000:.1f}ms processing")
        
        # Print final statistics
        final_status = self.get_system_status()
        print(f"\nSimulation Complete!")
        print(f"  Average processing time: {final_status['processing_stats']['avg_processing_time']*1000:.1f}ms")
        print(f"  Average detections per update: {final_status['processing_stats']['avg_detections']:.1f}")
        print(f"  Average confirmed tracks: {final_status['processing_stats']['avg_tracks']:.1f}")
        print(f"  Total tracks created: {final_status['tracking_stats']['total_created']}")
        print(f"  Final confirmed tracks: {final_status['tracking_stats']['confirmed_tracks']}")
        
        return results, final_status

def test_integrated_system():
    """Test the complete integrated tracking system"""
    print("Testing Integrated Tracking System")
    print("=" * 50)
    
    # Create system
    system = IntegratedTrackingSystem(max_range_km=250)
    
    # Load scenario
    system.load_scenario("busy_airport")
    
    # Run simulation
    results, final_status = system.run_batch_simulation(duration_seconds=30, time_step=1.0)
    
    # Plot results
    plot_system_performance(results, final_status)
    
    print("\n✅ Integrated system test complete!")
    
    return system, results

def plot_system_performance(results, status):
    """Plot system performance metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Integrated Radar Tracking System Performance', fontsize=16, weight='bold')
    
    times = results['times']
    
    # Plot 1: Detections and tracks over time
    ax1.plot(times, results['detections'], 'b-o', linewidth=2, markersize=4, label='Detections')
    ax1.plot(times, results['tracks'], 'r-s', linewidth=2, markersize=4, label='Confirmed Tracks')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title('Detections and Tracks Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Processing time
    ax2.plot(times, np.array(results['processing_times']) * 1000, 'g-', linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Processing Time (ms)')
    ax2.set_title('Real-time Performance')
    ax2.grid(True, alpha=0.3)
    
    # Add performance threshold line
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='100ms threshold')
    ax2.legend()
    
    # Plot 3: Track statistics
    tracking_stats = status['tracking_stats']
    categories = ['Confirmed', 'Tentative', 'Terminated']
    values = [
        tracking_stats['confirmed_tracks'],
        tracking_stats['tentative_tracks'], 
        tracking_stats['terminated_tracks']
    ]
    colors = ['green', 'orange', 'red']
    
    ax3.bar(categories, values, color=colors, alpha=0.7)
    ax3.set_ylabel('Number of Tracks')
    ax3.set_title('Track Status Distribution')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(values):
        ax3.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: System metrics
    metrics = {
        'Avg Detections': status['processing_stats']['avg_detections'],
        'Avg Tracks': status['processing_stats']['avg_tracks'], 
        'Avg Process Time (ms)': status['processing_stats']['avg_processing_time'] * 1000,
        'Track Quality': tracking_stats.get('average_quality', 0) * 100
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax4.barh(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    ax4.set_xlabel('Value')
    ax4.set_title('System Performance Metrics')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(metric_values):
        ax4.text(v + max(metric_values) * 0.01, i, f'{v:.1f}', 
                va='center', ha='left', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_integrated_system()