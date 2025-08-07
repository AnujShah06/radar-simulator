"""
Test different radar scenarios and data generation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.radar_data_generator import RadarDataGenerator, EnvironmentType
import matplotlib.pyplot as plt
import numpy as np

def test_target_movement():
    """Test how targets move over time"""
    generator = RadarDataGenerator(max_range_km=200)
    
    # Create a single aircraft
    aircraft = generator.add_aircraft(0, 50, heading=90, speed_kmh=800)  # Flying East
    
    # Track its position over time
    times = []
    positions_x = []
    positions_y = []
    ranges = []
    bearings = []
    
    for minute in range(10):  # 10 minutes
        times.append(minute)
        positions_x.append(aircraft.position_x)
        positions_y.append(aircraft.position_y)
        ranges.append(aircraft.range_km)
        bearings.append(aircraft.bearing_deg)
        
        generator.update_targets(time_step_seconds=60)  # 1 minute steps
    
    # Plot the results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position plot
    ax1.plot(positions_x, positions_y, 'b-o')
    ax1.set_xlabel('X Position (km)')
    ax1.set_ylabel('Y Position (km)')
    ax1.set_title('Aircraft Flight Path')
    ax1.grid(True)
    ax1.axis('equal')
    
    # Range over time
    ax2.plot(times, ranges, 'r-o')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Range (km)')
    ax2.set_title('Range from Radar')
    ax2.grid(True)
    
    # Bearing over time
    ax3.plot(times, bearings, 'g-o')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Bearing (degrees)')
    ax3.set_title('Bearing from Radar')
    ax3.grid(True)
    
    # Speed calculation
    distances = [np.sqrt((positions_x[i+1] - positions_x[i])**2 + 
                        (positions_y[i+1] - positions_y[i])**2) 
                for i in range(len(positions_x)-1)]
    speeds = [d * 60 for d in distances]  # km/h (distance per minute * 60)
    
    ax4.plot(times[1:], speeds, 'm-o')
    ax4.axhline(y=800, color='k', linestyle='--', label='Expected Speed')
    ax4.set_xlabel('Time (minutes)')
    ax4.set_ylabel('Calculated Speed (km/h)')
    ax4.set_title('Speed Verification')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Aircraft started at ({positions_x[0]:.1f}, {positions_y[0]:.1f})")
    print(f"Aircraft ended at ({positions_x[-1]:.1f}, {positions_y[-1]:.1f})")
    print(f"Distance traveled: {sum(distances):.1f} km in {len(times)-1} minutes")
    print(f"Average speed: {np.mean(speeds):.1f} km/h (expected: 800 km/h)")

def test_detection_probabilities():
    """Test how detection probability changes with different factors"""
    generator = RadarDataGenerator(max_range_km=200)
    
    # Test aircraft at different ranges
    ranges = np.arange(20, 200, 20)
    detection_probs = []
    
    for r in ranges:
        aircraft = generator.add_aircraft(r, 0, heading=0, speed_kmh=500)
        prob = generator.calculate_detection_probability(aircraft)
        detection_probs.append(prob)
        generator.targets.pop()  # Remove the test aircraft
    
    # Test different weather conditions
    weather_conditions = list(EnvironmentType)
    weather_probs = []
    
    test_aircraft = generator.add_aircraft(100, 0, heading=0, speed_kmh=500)
    for weather in weather_conditions:
        generator.set_environment(weather)
        prob = generator.calculate_detection_probability(test_aircraft)
        weather_probs.append(prob)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Range vs detection probability
    ax1.plot(ranges, detection_probs, 'b-o')
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('Detection Probability')
    ax1.set_title('Detection Probability vs Range')
    ax1.grid(True)
    ax1.set_ylim(0, 1)
    
    # Weather vs detection probability
    weather_names = [w.value.replace('_', ' ').title() for w in weather_conditions]
    ax2.bar(weather_names, weather_probs, color=['blue', 'lightblue', 'darkblue', 'cyan', 'gray'])
    ax2.set_ylabel('Detection Probability')
    ax2.set_title('Detection Probability vs Weather')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("Detection probability analysis complete!")

def compare_scenarios():
    """Compare different pre-built scenarios"""
    scenarios = ["busy_airport", "naval_operations", "storm_tracking"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))
    
    for i, scenario_name in enumerate(scenarios):
        generator = RadarDataGenerator(max_range_km=200)
        generator.create_scenario(scenario_name)
        
        ax = axes[i]
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 200)
        ax.set_title(scenario_name.replace('_', ' ').title())
        
        # Plot all targets
        colors = {'aircraft': 'red', 'ship': 'blue', 'weather': 'green', 'helicopter': 'orange'}
        
        for target in generator.targets:
            theta_rad = np.radians(target.bearing_deg)
            color = colors.get(target.target_type.value, 'white')
            marker = 'o' if target.target_type.value != 'weather' else '*'
            ax.plot(theta_rad, target.range_km, marker, color=color, markersize=6)
        
        # Add legend for first subplot
        if i == 0:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                    markersize=8, label='Aircraft'),
                             Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                                   markersize=8, label='Ship'),
                            Line2D([0], [0], marker='*', color='w', markerfacecolor='green', 
                                   markersize=10, label='Weather')]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0))
       
        # Print scenario statistics
        target_counts = {}
        for target in generator.targets:
            t_type = target.target_type.value
            target_counts[t_type] = target_counts.get(t_type, 0) + 1
        
        print(f"\n{scenario_name.replace('_', ' ').title()} Scenario:")
        print(f"  Environment: {generator.environment.value}")
        for t_type, count in target_counts.items():
            print(f"  {t_type.title()}: {count}")
    
    plt.tight_layout()
    plt.show()

def main():
    """Run all Day 3 tests"""
    
    tests = [
        ("1", "Target Movement Analysis", test_target_movement),
        ("2", "Detection Probability Analysis", test_detection_probabilities),
        ("3", "Scenario Comparison", compare_scenarios)
    ]
    
    print("Available tests:")
    for code, name, _ in tests:
        print(f"{code}. {name}")
    print()
    
    choice = input("Choose test (1-3) or 'all' for all tests: ").strip().lower()
    
    if choice == 'all':
        for _, name, test_func in tests:
            print(f"\n--- Running {name} ---")
            test_func()
    else:
        for code, name, test_func in tests:
            if choice == code:
                print(f"\n--- Running {name} ---")
                test_func()
                break
        else:
            print("Invalid choice. Running all tests...")
            for _, name, test_func in tests:
                print(f"\n--- Running {name} ---")
                test_func()

if __name__ == "__main__":
    main()