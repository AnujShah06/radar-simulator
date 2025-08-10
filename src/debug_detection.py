"""Debug the detection pipeline"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.radar_data_generator import RadarDataGenerator
from src.target_detection import TargetDetector

def debug_detection_chain():
    print("Debugging Detection Chain")
    print("=" * 30)
    
    # Create components
    generator = RadarDataGenerator(max_range_km=200)
    detector = TargetDetector()
    
    # Make detection very lenient
    detector.min_detections_for_confirmation = 1
    detector.association_distance_threshold = 50.0
    
    # Create simple scenario
    generator.create_scenario("busy_airport")
    print(f"Created {len(generator.targets)} targets")
    
    # Simulate one sweep that should hit everything
    raw_detections = generator.simulate_radar_detection(0, sweep_width_deg=360)
    print(f"Raw detections: {len(raw_detections)}")
    
    if raw_detections:
        # Process through pipeline
        detected_targets = detector.process_raw_detections(raw_detections)
        print(f"Processed detections: {len(detected_targets)}")
        
        for target in detected_targets:
            print(f"  {target.id}: {target.classification} at ({target.range_km:.1f}, {target.bearing_deg:.1f})")
    else:
        print("❌ No raw detections - radar detection failed")

if __name__ == "__main__":
    debug_detection_chain()