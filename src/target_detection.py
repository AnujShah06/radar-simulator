"""
Target detection and classification algorithms
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from signal_processing import RadarReturn, SignalProcessor
import random

@dataclass
class DetectedTarget:
    """A confirmed target detection"""
    id: str
    range_km: float
    bearing_deg: float
    signal_strength: float
    snr_db: float
    doppler_shift: float
    classification: str  #classes: aircraft/ship/weather/unknown
    confidence: float
    timestamp: float
    raw_returns: List[RadarReturn]

class TargetDetector:
    def __init__(self):
        self.signal_processor = SignalProcessor()
        self.detection_history = []
        self.confirmed_targets = []
        self.target_id_counter = 1
        
        #detection parameters
        self.min_detections_for_confirmation = 3
        self.max_time_between_detections = 10.0  #seconds
        self.association_distance_threshold = 5.0  #km
        
        #classification thresholds
        self.classification_rules = {
            'aircraft': {
                'doppler_range': (100, 15000),     #much wider range for aircraft
                'rcs_range': (0.1, 200),          #m²
                'signal_variability': 0.2
            },
            'ship': {
                'doppler_range': (0, 300),        #ships can be fast
                'rcs_range': (50, 2000),          #m²
                'signal_variability': 0.1
            },
            'weather': {
                'doppler_range': (0, 200),        #weather can move
                'rcs_range': (0.01, 50),          #m²
                'signal_variability': 0.4
            }
        }
    
    def process_raw_detections(self, raw_detections: List[Dict]) -> List[DetectedTarget]:
        """Process raw detections through the complete detection pipeline"""
        
        radar_returns = self.signal_processor.process_radar_sweep(raw_detections)
        
        filtered_returns = self.signal_processor.filter_detections(radar_returns)
        
        clustered_returns = self.cluster_nearby_returns(filtered_returns)
        
        confirmed_targets = self.confirm_targets(clustered_returns)
        
        classified_targets = self.classify_targets(confirmed_targets)
        
        return classified_targets
    
    def cluster_nearby_returns(self, radar_returns: List[RadarReturn]) -> List[List[RadarReturn]]:
        """Group nearby radar returns that likely come from the same target"""
        if not radar_returns:
            return []
        
        clusters = []
        used_returns = set()
        
        for i, return1 in enumerate(radar_returns):
            if i in used_returns:
                continue
            
            cluster = [return1]
            used_returns.add(i)
            
            for j, return2 in enumerate(radar_returns):
                if j in used_returns or i == j:
                    continue
                
                distance = self.calculate_detection_distance(return1, return2)
                
                if distance < self.association_distance_threshold:
                    cluster.append(return2)
                    used_returns.add(j)
            
            clusters.append(cluster)
        
        return clusters
    def calculate_detection_distance(self, return1: RadarReturn, return2: RadarReturn) -> float:
        """Calculate distance between two radar returns"""
        #convert to cartesian coordinates
        x1 = return1.range_km * np.sin(np.radians(return1.bearing_deg))
        y1 = return1.range_km * np.cos(np.radians(return1.bearing_deg))
        
        x2 = return2.range_km * np.sin(np.radians(return2.bearing_deg))
        y2 = return2.range_km * np.cos(np.radians(return2.bearing_deg))
        
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    
    def confirm_targets(self, clustered_returns: List[List[RadarReturn]]) -> List[DetectedTarget]:
        """Confirm targets based on multiple consistent detections"""
        confirmed_targets = []
        
        for cluster in clustered_returns:
            if len(cluster) < self.min_detections_for_confirmation:
                continue
            
            #calculate average position and signal characteristics
            avg_range = np.mean([r.range_km for r in cluster])
            avg_bearing = np.mean([r.bearing_deg for r in cluster])
            avg_signal = np.mean([r.signal_strength for r in cluster])
            avg_doppler = np.mean([r.doppler_shift for r in cluster])
            
            #calculate signal-to-noise ratio
            avg_noise = np.mean([r.noise_level for r in cluster])
            snr_db = self.signal_processor.calculate_snr(avg_signal, avg_noise)
            
            #create confirmed target
            target = DetectedTarget(
                id=f"TGT_{self.target_id_counter:03d}",
                range_km=avg_range,
                bearing_deg=avg_bearing,
                signal_strength=avg_signal,
                snr_db=snr_db,
                doppler_shift=avg_doppler,
                classification="unknown",  #classified next
                confidence=0.0,  #calculated during classification
                timestamp=max([r.timestamp for r in cluster]),
                raw_returns=cluster
            )
            
            confirmed_targets.append(target)
            self.target_id_counter += 1
        
        return confirmed_targets
    
    def classify_targets(self, targets: List[DetectedTarget]) -> List[DetectedTarget]:
        """Classify targets based on their characteristics"""
        for target in targets:
            #calculate signal variability
            signal_values = [r.signal_strength for r in target.raw_returns]
            signal_std = np.std(signal_values)
            signal_variability = signal_std / np.mean(signal_values) if signal_values else 0
            
            #estimate RCS from received signal strength and range
            estimated_rcs = self.estimate_rcs_from_signal(target.signal_strength, target.range_km)
            
            print(f"\n  Classifying {target.id}:")
            print(f"    Doppler: {abs(target.doppler_shift):.1f} m/s")
            print(f"    Estimated RCS: {estimated_rcs:.1f} m²")
            print(f"    Signal variability: {signal_variability:.3f}")
            
            #score each classification
            classification_scores = {}
            
            for class_name, rules in self.classification_rules.items():
                score = 0
                
                #doppler score - check absolute value
                doppler_min, doppler_max = rules['doppler_range']
                abs_doppler = abs(target.doppler_shift)
                if doppler_min <= abs_doppler <= doppler_max:
                    score += 0.4
                    print(f"      {class_name} doppler match: +0.4")
                
                #rcs score
                rcs_min, rcs_max = rules['rcs_range']
                if rcs_min <= estimated_rcs <= rcs_max:
                    score += 0.4
                    print(f"      {class_name} RCS match: +0.4")
                
                #signal variability score - more lenient
                expected_variability = rules['signal_variability']
                variability_diff = abs(signal_variability - expected_variability)
                if variability_diff < 0.2:  #more lenient than 0.1
                    score += 0.2
                    print(f"      {class_name} variability match: +0.2")
                
                classification_scores[class_name] = score
                print(f"      {class_name} total score: {score:.1f}")
            
            #choose best classification
            best_class = max(classification_scores, key=classification_scores.get)
            best_score = classification_scores[best_class]
            
            print(f"    Best: {best_class} with score {best_score:.1f}")
            
            target.classification = best_class
            target.confidence = best_score
            
            #lower the confidence threshold
            if target.confidence < 0.4:  #reduced from 0.5
                target.classification = "unknown"
                print(f"    → Confidence too low, marked as unknown")
        
        return targets
    
    def estimate_rcs_from_signal(self, signal_strength: float, range_km: float) -> float:
        """Estimate radar cross section from received signal strength and range"""
        #inverse of radar range equation (simplified)
        #approximation - real radar systems use calibration
        
        range_m = range_km * 1000
        range_factor = (range_m / 1000) ** 4  #fourth power law
        estimated_rcs = signal_strength * range_factor * 10  #scale factor
        
        return max(0.01, estimated_rcs)  #minimum rcs
    
    def get_detection_statistics(self, targets: List[DetectedTarget]) -> Dict:
        """Calculate detection statistics"""
        if not targets:
            return {"total": 0}
        
        stats = {
            "total": len(targets),
            "by_classification": {},
            "avg_snr": np.mean([t.snr_db for t in targets]),
            "avg_confidence": np.mean([t.confidence for t in targets]),
            "range_distribution": {
                "min": min([t.range_km for t in targets]),
                "max": max([t.range_km for t in targets]),
                "avg": np.mean([t.range_km for t in targets])
            }
        }
        
        for target in targets:
            class_name = target.classification
            stats["by_classification"][class_name] = stats["by_classification"].get(class_name, 0) + 1
        
        return stats

#test the detection system
def test_target_detection():
    """Test the target detection system"""
    print("Testing Target Detection System")
    print("=" * 40)
    
    detector = TargetDetector()
    
    #lower the confirmation threshold for testing
    detector.min_detections_for_confirmation = 2  #reduce from 3 to 2
    
    # Create mock raw detections with multiple returns per target
    raw_detections = [
        #aircraft target - 3 detections
        {
            'range': 75.0,
            'bearing': 45.0,
            'target': type('MockTarget', (), {
                'radar_cross_section': 20.0,
                'speed': 800,
                'heading': 90,
                'target_type': type('TargetType', (), {'value': 'aircraft'})()
            })(),
            'detection_time': 1.0
        },
        {
            'range': 76.0,  #same aircraft, slightly different measurement
            'bearing': 46.0,
            'target': type('MockTarget', (), {
                'radar_cross_section': 20.0,
                'speed': 800,
                'heading': 90,
                'target_type': type('TargetType', (), {'value': 'aircraft'})()
            })(),
            'detection_time': 1.1
        },
        {
            'range': 74.5,  #third detection of same aircraft
            'bearing': 44.5,
            'target': type('MockTarget', (), {
                'radar_cross_section': 20.0,
                'speed': 800,
                'heading': 90,
                'target_type': type('TargetType', (), {'value': 'aircraft'})()
            })(),
            'detection_time': 1.2
        },
        
        #ship target - 3 detections
        {
            'range': 150.0,
            'bearing': 120.0,
            'target': type('MockTarget', (), {
                'radar_cross_section': 500.0,
                'speed': 25,
                'heading': 180,
                'target_type': type('TargetType', (), {'value': 'ship'})()
            })(),
            'detection_time': 1.0
        },
        {
            'range': 151.0,  #same ship
            'bearing': 121.0,
            'target': type('MockTarget', (), {
                'radar_cross_section': 500.0,
                'speed': 25,
                'heading': 180,
                'target_type': type('TargetType', (), {'value': 'ship'})()
            })(),
            'detection_time': 1.1
        },
        {
            'range': 149.5,  #third detection of same ship
            'bearing': 119.5,
            'target': type('MockTarget', (), {
                'radar_cross_section': 500.0,
                'speed': 25,
                'heading': 180,
                'target_type': type('TargetType', (), {'value': 'ship'})()
            })(),
            'detection_time': 1.2
        },
        
        #weather returns - 2 detections
        {
            'range': 90.0,
            'bearing': 200.0,
            'target': type('MockTarget', (), {
                'radar_cross_section': 2.0,
                'speed': 15,
                'heading': 270,
                'target_type': type('TargetType', (), {'value': 'weather'})()
            })(),
            'detection_time': 1.0
        },
        {
            'range': 91.0,  #same weather cell
            'bearing': 201.0,
            'target': type('MockTarget', (), {
                'radar_cross_section': 2.0,
                'speed': 15,
                'heading': 270,
                'target_type': type('TargetType', (), {'value': 'weather'})()
            })(),
            'detection_time': 1.1
        },
        
        #single false alarm (won't be confirmed)
        {
            'range': 180.0,
            'bearing': 300.0,
            'target': None,
            'detection_time': 1.0,
            'is_false_alarm': True
        }
    ]
    
    print(f"\nProcessing {len(raw_detections)} raw detections...")
    print("  Expected: 3 confirmed targets (aircraft, ship, weather)")
    
    # Process through detection pipeline
    detected_targets = detector.process_raw_detections(raw_detections)
    
    print(f"\nDetection Results:")
    print(f"  Raw detections: {len(raw_detections)}")
    print(f"  Confirmed targets: {len(detected_targets)}")
    
    # Display target details
    if detected_targets:
        for target in detected_targets:
            print(f"\n  {target.id}: {target.classification.upper()}")
            print(f"    Position: {target.range_km:.1f} km, {target.bearing_deg:.1f}°")
            print(f"    Signal: {target.signal_strength:.3f}, SNR: {target.snr_db:.1f} dB")
            print(f"    Doppler: {target.doppler_shift:.1f} m/s")
            print(f"    Confidence: {target.confidence:.2f}")
            print(f"    Raw returns: {len(target.raw_returns)}")
    else:
        print("  ❌ No targets confirmed - checking detection pipeline...")
        
        #debug the pipeline
        radar_returns = detector.signal_processor.process_radar_sweep(raw_detections)
        print(f"    Step 1 - Radar returns: {len(radar_returns)}")
        
        filtered_returns = detector.signal_processor.filter_detections(radar_returns)
        print(f"    Step 2 - Filtered returns: {len(filtered_returns)}")
        
        clustered_returns = detector.cluster_nearby_returns(filtered_returns)
        print(f"    Step 3 - Clusters: {len(clustered_returns)}")
        
        for i, cluster in enumerate(clustered_returns):
            print(f"      Cluster {i+1}: {len(cluster)} returns")
    
    # Statistics
    stats = detector.get_detection_statistics(detected_targets)
    print(f"\n  Detection Statistics:")
    print(f"    Total targets: {stats['total']}")
    if stats['total'] > 0:
        print(f"    Average SNR: {stats.get('avg_snr', 0):.1f} dB")
        print(f"    Average confidence: {stats.get('avg_confidence', 0):.2f}")
        
        if 'by_classification' in stats:
            print("    By classification:")
            for class_name, count in stats['by_classification'].items():
                print(f"      {class_name}: {count}")
    
    print("\n✅ Target detection test complete!")

# Add a simpler test function
def test_signal_processing_only():
    """Test just the signal processing without clustering/confirmation"""
    print("\nTesting Signal Processing Only")
    print("=" * 30)
    
    detector = TargetDetector()
    
    # Simple test detection
    simple_detection = {
        'range': 100.0,
        'bearing': 45.0,
        'target': type('MockTarget', (), {
            'radar_cross_section': 50.0,
            'speed': 600,
            'heading': 90,
            'target_type': type('TargetType', (), {'value': 'aircraft'})()
        })(),
        'detection_time': 1.0
    }
    
    # Process just one detection
    radar_returns = detector.signal_processor.process_radar_sweep([simple_detection])
    
    print(f"Input: 1 raw detection")
    print(f"Output: {len(radar_returns)} radar returns")
    
    if radar_returns:
        ret = radar_returns[0]
        print(f"  Range: {ret.range_km:.1f} km")
        print(f"  Bearing: {ret.bearing_deg:.1f}°")
        print(f"  Signal: {ret.signal_strength:.3f}")
        print(f"  Noise: {ret.noise_level:.3f}")
        print(f"  Doppler: {ret.doppler_shift:.1f} m/s")
    
    print("✅ Signal processing working!")

def debug_detection_pipeline():
    """Debug the detection pipeline step by step"""
    print("\nDetailed Pipeline Debug")
    print("=" * 30)
    
    detector = TargetDetector()
    detector.min_detections_for_confirmation = 2
    
    #simple test with one strong target
    test_detection = {
        'range': 100.0,
        'bearing': 45.0,
        'target': type('MockTarget', (), {
            'radar_cross_section': 100.0,  #large RCS
            'speed': 600,
            'heading': 90,
            'target_type': type('TargetType', (), {'value': 'aircraft'})()
        })(),
        'detection_time': 1.0
    }
    
    #duplicate it to have 2 detections
    raw_detections = [test_detection, test_detection.copy()]
    
    print(f"Step 0: Input - {len(raw_detections)} raw detections")
    
    #step 1: process radar sweep
    radar_returns = detector.signal_processor.process_radar_sweep(raw_detections)
    print(f"Step 1: Radar processing - {len(radar_returns)} returns")
    for i, ret in enumerate(radar_returns):
        print(f"  Return {i+1}: Signal={ret.signal_strength:.3f}, Noise={ret.noise_level:.3f}")
    
    #step 2: filter detections
    print(f"\nStep 2: Filtering...")
    signals = [r.signal_strength for r in radar_returns]
    print(f"  Original signals: {[f'{s:.3f}' for s in signals]}")
    
    #test each filter step
    filtered_signals = detector.signal_processor.moving_average_filter(signals)
    print(f"  After moving avg: {[f'{s:.3f}' for s in filtered_signals]}")
    
    smoothed_signals = detector.signal_processor.exponential_filter(filtered_signals)
    print(f"  After smoothing: {[f'{s:.3f}' for s in smoothed_signals]}")
    
    #test threshold detection
    threshold = detector.signal_processor.detection_threshold
    print(f"  Detection threshold: {threshold}")
    
    valid_detections = detector.signal_processor.threshold_detection(smoothed_signals)
    print(f"  Threshold results: {valid_detections}")
    
    #test SNR calculation
    for i, (radar_return, is_valid, filtered_signal) in enumerate(
        zip(radar_returns, valid_detections, smoothed_signals)):
        snr = detector.signal_processor.calculate_snr(filtered_signal, radar_return.noise_level)
        print(f"  Return {i+1}: SNR = {snr:.1f} dB, Valid = {is_valid}")
    
    #full filter
    filtered_returns = detector.signal_processor.filter_detections(radar_returns)
    print(f"\nStep 2 Result: {len(filtered_returns)} filtered returns")
    
    if len(filtered_returns) == 0:
        print("❌ All returns filtered out!")
        print("Try running with more lenient thresholds...")
    else:
        print("✅ Some returns survived filtering")
        
        #continue with clustering
        clustered_returns = detector.cluster_nearby_returns(filtered_returns)
        print(f"Step 3: {len(clustered_returns)} clusters")
        
        confirmed_targets = detector.confirm_targets(clustered_returns)
        print(f"Step 4: {len(confirmed_targets)} confirmed targets")

if __name__ == "__main__":
    test_signal_processing_only()
    debug_detection_pipeline()
    test_target_detection()