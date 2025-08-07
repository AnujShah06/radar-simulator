"""
Integrated radar system combining data generation and display
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.radar_data_generator import RadarDataGenerator, EnvironmentType
from src.radar_display import RadarDisplay
from src.integrated_radar_system import IntegratedRadarSystem
import time

def quick_static_demo():
    """Show a static view of different scenarios"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    scenarios = ["busy_airport", "naval_operations", "storm_tracking"]
    
    fig = plt.figure(figsize=(15, 10))
    
    for i, scenario_name in enumerate(scenarios, 1):
        #radar system
        radar_system = IntegratedRadarSystem(max_range_km=200)
        radar_system.load_scenario(scenario_name)
        
        #simulate several sweeps to get detections
        for sweep_angle in range(0, 360, 30):
            radar_system.current_sweep_angle = sweep_angle
            detections = radar_system.data_generator.simulate_radar_detection(sweep_angle)
            radar_system.detected_targets.extend(detections)
        
        #subplot
        ax = fig.add_subplot(2, 2, i, projection='polar')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 200)
        ax.set_title(f'{scenario_name.replace("_", " ").title()} - Static View')
        
        #plot all detections
        colors = {'aircraft': 'red', 'ship': 'blue', 'weather': 'green'}
        
        for detection in radar_system.detected_targets:
            if detection.get('is_false_alarm'):
                color, marker = 'yellow', 'x'
            elif detection['target']:
                target_type = detection['target'].target_type.value
                color = colors.get(target_type, 'white')
                marker = '*' if target_type == 'weather' else 'o'
            else:
                color, marker = 'white', 'o'
            
            theta_rad = np.radians(detection['bearing'])
            ax.plot(theta_rad, detection['range'], marker, 
                   color=color, markersize=6, alpha=0.8)
        
        #range rings
        for r in range(50, 201, 50):
            theta_full = np.linspace(0, 2*np.pi, 100)
            ax.plot(theta_full, np.full_like(theta_full, r), 'g-', alpha=0.3, linewidth=0.5)
    
    #title and legend
    fig.suptitle('Day 3: Radar Data Generation - All Scenarios', fontsize=16)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Aircraft'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Ship'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=10, label='Weather'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='yellow', markersize=8, label='False Alarm')
    ]
    
    #legend to fourth subplot area
    ax_legend = fig.add_subplot(2, 2, 4)
    ax_legend.axis('off')
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=12)
    ax_legend.text(0.5, 0.7, 'Legend', ha='center', va='center', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n Static demo complete!")

def performance_test():
    """Test system performance with many targets"""
    print("\n--- Performance Test ---")
    
    generator = RadarDataGenerator(max_range_km=300)
    
    # Add lots of targets
    print("Adding 100 targets...")
    for i in range(100):
        if i < 60:  # 60 aircraft
            x = np.random.uniform(-250, 250)
            y = np.random.uniform(-250, 250)
            heading = np.random.uniform(0, 360)
            speed = np.random.uniform(200, 900)
            generator.add_aircraft(x, y, heading, speed)
        elif i < 80:  # 20 ships
            x = np.random.uniform(-200, 200)
            y = np.random.uniform(-200, 200)
            heading = np.random.uniform(0, 360)
            speed = np.random.uniform(10, 40)
            generator.add_ship(x, y, heading, speed)
        else:  # 20 weather returns
            generator.add_weather_returns(
                np.random.uniform(-150, 150),
                np.random.uniform(-150, 150),
                np.random.uniform(20, 50)
            )
    
    print(f"Total targets: {len(generator.targets)}")
    
    print("Testing target updates...")
    start_time = time.time()
    for _ in range(60):  # 60 updates (1 minute of simulation)
        generator.update_targets(time_step_seconds=1.0)
    update_time = time.time() - start_time
    
    print(f"Update performance: {update_time:.3f}s for 60 updates")
    print(f"That's {update_time/60:.3f}s per update (should be < 0.1s for real-time)")
    
    print("Testing detection performance...")
    start_time = time.time()
    total_detections = 0
    for angle in range(0, 360, 10):  # Test every 10 degrees
        detections = generator.simulate_radar_detection(angle)
        total_detections += len(detections)
    detection_time = time.time() - start_time
    
    print(f"Detection performance: {detection_time:.3f}s for full 360° scan")
    print(f"Total detections in full scan: {total_detections}")
    print(f"Remaining targets after range filtering: {len(generator.targets)}")
    
    if update_time/60 < 0.1 and detection_time < 1.0:
        print("Performance test PASSED - System ready for real-time operation!")
    else:
        print("Performance test indicates optimization may be needed")

def main():
    """Main demo for Day 3"""
    
    demos = [
        ("1", "Static scenario overview", quick_static_demo),
        ("2", "Performance test with 100 targets", performance_test),
        ("3", "Full animated demo", lambda: IntegratedRadarSystem().run_simulation())
    ]
    
    print("Available demos:")
    for code, name, _ in demos:
        print(f"{code}. {name}")
    print()
    
    choice = input("Choose demo (1-3): ").strip()
    
    for code, name, demo_func in demos:
        if choice == code:
            print(f"\n--- {name.title()} ---")
            demo_func()
            break
    else:
        print("Invalid choice, running static demo...")
        quick_static_demo()


if __name__ == "__main__":
    main()