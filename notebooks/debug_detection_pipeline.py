"""
Debug the complete detection pipeline step by step
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.radar_data_generator import RadarDataGenerator
from src.target_detection import TargetDetector

def debug_complete_pipeline():
    print("🔍 DEBUGGING COMPLETE DETECTION PIPELINE")
    print("=" * 50)
    
    # Create components
    generator = RadarDataGenerator(max_range_km=200)
    detector = TargetDetector()
    
    # Create scenario
    generator.create_scenario("busy_airport")
    print(f"✅ Created scenario with {len(generator.targets)} targets")
    
    # Get raw detections
    raw_detections = generator.simulate_radar_detection(0, sweep_width_deg=360)
    print(f"✅ Raw detections from radar: {len(raw_detections)}")
    
    if not raw_detections:
        print("❌ No raw detections - radar sweep issue")
        return
    
    print(f"\n🔍 STEP-BY-STEP PIPELINE DEBUG:")
    
    # Step 1: Signal processing
    print(f"\nStep 1: Signal Processing")
    radar_returns = detector.signal_processor.process_radar_sweep(raw_detections)
    print(f"  Input: {len(raw_detections)} raw detections")
    print(f"  Output: {len(radar_returns)} radar returns")
    
    if not radar_returns:
        print("❌ FAILED at signal processing")
        return
    
    # Step 2: Filtering
    print(f"\nStep 2: Signal Filtering")
    filtered_returns = detector.signal_processor.filter_detections(radar_returns)
    print(f"  Input: {len(radar_returns)} radar returns")
    print(f"  Output: {len(filtered_returns)} filtered returns")
    
    if not filtered_returns:
        print("❌ FAILED at signal filtering - all signals filtered out")
        print("🔧 This is likely the problem!")
        
        # Debug filtering step by step
        signals = [r.signal_strength for r in radar_returns]
        noise_levels = [r.noise_level for r in radar_returns]
        
        print(f"\n  Signal strengths: {[f'{s:.3f}' for s in signals[:5]]}...")
        print(f"  Noise levels: {[f'{n:.3f}' for n in noise_levels[:5]]}...")
        
        # Test threshold detection
        valid_detections = detector.signal_processor.threshold_detection(signals, noise_levels=noise_levels)
        print(f"  Threshold results: {valid_detections[:5]}...")
        
        return
    
    # Step 3: Clustering
    print(f"\nStep 3: Target Clustering")
    clustered_returns = detector.cluster_nearby_returns(filtered_returns)
    print(f"  Input: {len(filtered_returns)} filtered returns")
    print(f"  Output: {len(clustered_returns)} clusters")
    
    for i, cluster in enumerate(clustered_returns[:3]):
        print(f"    Cluster {i+1}: {len(cluster)} returns")
    
    # Step 4: Confirmation
    print(f"\nStep 4: Target Confirmation")
    print(f"  Confirmation requirement: {detector.min_detections_for_confirmation} detections")
    
    confirmed_targets = detector.confirm_targets(clustered_returns)
    print(f"  Input: {len(clustered_returns)} clusters")
    print(f"  Output: {len(confirmed_targets)} confirmed targets")
    
    if not confirmed_targets:
        print("❌ FAILED at target confirmation")
        print(f"  Issue: Clusters have fewer than {detector.min_detections_for_confirmation} detections")
        print("🔧 Need to lower confirmation requirement!")
        return
    
    # Step 5: Classification
    print(f"\nStep 5: Target Classification")
    classified_targets = detector.classify_targets(confirmed_targets)
    print(f"  Input: {len(confirmed_targets)} confirmed targets")
    print(f"  Output: {len(classified_targets)} classified targets")
    
    for target in classified_targets:
        print(f"    {target.id}: {target.classification} (confidence: {target.confidence:.2f})")
    
    print(f"\n🎉 PIPELINE COMPLETE: {len(classified_targets)} final detections")

if __name__ == "__main__":
    debug_complete_pipeline()