"""
Showcases the full radar tracking pipeline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.integrated_tracking_system import IntegratedTrackingSystem
import time

def quick_demo():
    """Quick demonstration of the tracking system"""
    print("🎯 DAY 5 TRACKING SYSTEM DEMO")
    print("=" * 50)
    print()
    print("This demo showcases your complete radar tracking system:")
    print("  🔬 Kalman filter-based target tracking")
    print("  🎯 Multi-target management and association")
    print("  📊 Real-time performance monitoring")
    print("  🧠 Advanced track lifecycle management")
    print()
    
    # Create and configure system
    system = IntegratedTrackingSystem(max_range_km=200)
    system.load_scenario("busy_airport")
    
    print("Running 15-second simulation with busy airport scenario...")
    print("-" * 50)
    
    # Store results for visualization
    all_results = []
    
    for second in range(15):
        result = system.update_system(1.0)
        
        confirmed_tracks = result['confirmed_tracks']
        processing_time = result['processing_time'] * 1000
        
        print(f"t={second+1:2d}s: {len(result['processed_detections']):2d} detections → "
              f"{len(confirmed_tracks):2d} tracks (processing: {processing_time:4.1f}ms)")
        
        # Store for plotting
        all_results.append({
            'time': second + 1,
            'detections': len(result['processed_detections']),
            'tracks': len(confirmed_tracks),
            'processing_time': processing_time,
            'confirmed_tracks': confirmed_tracks
        })
    
    # Final system status
    final_status = system.get_system_status()
    print(f"\n📊 FINAL STATISTICS:")
    print(f"   Total tracks created: {final_status['tracking_stats']['total_created']}")
    print(f"   Active confirmed tracks: {final_status['tracking_stats']['confirmed_tracks']}")
    print(f"   Average processing time: {final_status['processing_stats']['avg_processing_time']*1000:.1f}ms")
    print(f"   Average track quality: {final_status['tracking_stats'].get('average_quality', 0):.2f}")
    
    # Show track details
    active_tracks = system.tracker.get_confirmed_tracks()
    if active_tracks:
        print(f"\n🎯 ACTIVE TRACKS:")
        for track in active_tracks[:5]:  # Show first 5 tracks
            print(f"   {track.id}: ({track.state.x:6.1f}, {track.state.y:6.1f}) km")
            print(f"      Speed: {track.state.speed_kmh:5.1f} km/h, Heading: {track.state.heading_deg:5.1f}°")
            print(f"      Classification: {track.classification} (confidence: {track.classification_confidence:.2f})")
            print(f"      Quality: {track.quality_score:.2f}, Hits: {track.hits}, Age: {track.age:.1f}s")
            print()
    
    # Create visualization
    plot_demo_results(all_results)
    
    print("✅ Demo complete! Your tracking system is working excellently.")
    
    return system, all_results

def plot_demo_results(results):
    """Plot demo results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Day 5: Advanced Radar Tracking System Demo', fontsize=16, weight='bold')
    
    times = [r['time'] for r in results]
    detections = [r['detections'] for r in results]
    tracks = [r['tracks'] for r in results]
    processing_times = [r['processing_time'] for r in results]
    
    # Detection and track counts over time
    ax1.plot(times, detections, 'o-', color='blue', linewidth=2, markersize=6, label='Detections')
    ax1.plot(times, tracks, 's-', color='red', linewidth=2, markersize=6, label='Confirmed Tracks')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title('Detection and Tracking Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Processing time performance
    ax2.plot(times, processing_times, 'g-^', linewidth=2, markersize=6)
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Real-time threshold (100ms)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Processing Time (ms)')
    ax2.set_title('Real-time Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Track positions (final snapshot)
    final_tracks = results[-1]['confirmed_tracks'] if results else []
    if final_tracks:
        for i, track in enumerate(final_tracks):
            color = plt.cm.tab10(i % 10)
            ax3.plot(track.state.x, track.state.y, 'o', color=color, markersize=10, 
                    label=f'{track.id} ({track.classification})')
            
            # Add velocity vector
            vx_scaled = track.state.vx * 5  # Scale for visibility
            vy_scaled = track.state.vy * 5
            ax3.arrow(track.state.x, track.state.y, vx_scaled, vy_scaled,
                     head_width=2, head_length=3, fc=color, ec=color, alpha=0.7)
    
    ax3.set_xlabel('X Position (km)')
    ax3.set_ylabel('Y Position (km)')
    ax3.set_title('Final Track Positions & Velocities')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # System performance metrics
    avg_detections = np.mean(detections)
    avg_tracks = np.mean(tracks)
    avg_processing = np.mean(processing_times)
    detection_efficiency = avg_tracks / avg_detections if avg_detections > 0 else 0
    
    metrics = {
        'Avg Detections': avg_detections,
        'Avg Tracks': avg_tracks,
        'Avg Processing (ms)': avg_processing,
        'Track Efficiency': detection_efficiency * 100
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = ['skyblue', 'lightgreen', 'lightyellow', 'lightcoral']
    
    bars = ax4.barh(metric_names, metric_values, color=colors)
    ax4.set_xlabel('Value')
    ax4.set_title('System Performance Summary')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax4.text(bar.get_width() + max(metric_values) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}', va='center', ha='left', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main demo menu"""
    print("Advanced Tracking System!")
    print()
    print("Available demonstrations:")
    print("1. Quick tracking demo (15 seconds)")
    print("2. Performance analysis (comprehensive)")
    print("3. Single target accuracy test")
    print()
    
    choice = input("Choose demo (1-3) or press Enter for quick demo: ").strip() or "1"
    
    if choice == "1":
        quick_demo()
    elif choice == "2":
        from notebooks.day5_performance_analysis import main as perf_main
        perf_main()
    elif choice == "3":
        from notebooks.day5_performance_analysis import test_tracking_accuracy
        test_tracking_accuracy()
    else:
        print("Invalid choice, running quick demo...")
        quick_demo()

if __name__ == "__main__":
    main()