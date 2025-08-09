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
                'doppler_range': (50, 500),    #m/s
                'rcs_range': (0.1, 200),       #m²
                'signal_variability': 0.2
            },
            'ship': {
                'doppler_range': (0, 50),      #m/s
                'rcs_range': (100, 2000),      #m²
                'signal_variability': 0.1
            },
            'weather': {
                'doppler_range': (0, 30),      #m/s
                'rcs_range': (0.01, 10),       #m²
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
