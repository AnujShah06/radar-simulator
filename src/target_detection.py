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
    classification: str  # 'aircraft', 'ship', 'weather', 'unknown'
    confidence: float
    timestamp: float
    raw_returns: List[RadarReturn]

class TargetDetector:
    def __init__(self):
        self.signal_processor = SignalProcessor()
        self.detection_history = []
        self.confirmed_targets = []
        self.target_id_counter = 1
        
        # Detection parameters
        self.min_detections_for_confirmation = 3
        self.max_time_between_detections = 10.0  # seconds
        self.association_distance_threshold = 5.0  # km
        
        # Classification thresholds
        self.classification_rules = {
            'aircraft': {
                'doppler_range': (50, 500),    # m/s
                'rcs_range': (0.1, 200),       # m²
                'signal_variability': 0.2
            },
            'ship': {
                'doppler_range': (0, 50),      # m/s
                'rcs_range': (100, 2000),      # m²
                'signal_variability': 0.1
            },
            'weather': {
                'doppler_range': (0, 30),      # m/s
                'rcs_range': (0.01, 10),       # m²
                'signal_variability': 0.4
            }
        }
    
    def process_raw_detections(self, raw_detections: List[Dict]) -> List[DetectedTarget]:
        """Process raw detections through the complete detection pipeline"""
        
        # Step 1: Signal processing
        radar_returns = self.signal_processor.process_radar_sweep(raw_detections)
        
        # Step 2: Noise filtering
        filtered_returns = self.signal_processor.filter_detections(radar_returns)
        
        # Step 3: Cluster nearby returns (same target might generate multiple returns)
        clustered_returns = self.cluster_nearby_returns(filtered_returns)
        
        # Step 4: Confirm targets based on multiple detections
        confirmed_targets = self.confirm_targets(clustered_returns)
        
        # Step 5: Classify targets
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
            
            # Find nearby returns
            for j, return2 in enumerate(radar_returns):
                if j in used_returns or i == j:
                    continue
                
                # Calculate distance between returns
                distance = self.calculate_detection_distance(return1, return2)
                
                if distance < self.association_distance_threshold:
                    cluster.append(return2)
                    used_returns.add(j)
            
            clusters.append(cluster)
        
        return clusters
