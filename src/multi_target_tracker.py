"""
Multi-target tracking system using Kalman filters
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time
from kalman_filter import KalmanFilter, TrackState
from target_detection import DetectedTarget

@dataclass
class Track:
    """A tracked target with history and metadata"""
    id: str
    kalman_filter: KalmanFilter
    state: TrackState
    detections: List[DetectedTarget] = field(default_factory=list)
    
    # Track quality metrics
    hits: int = 0           # Number of associated detections
    misses: int = 0         # Number of missed detections
    age: float = 0.0        # Time since track started
    last_update: float = 0.0 # Time of last update
    
    # Track status
    confirmed: bool = False
    terminated: bool = False
    
    # Classification
    classification: str = "unknown"
    classification_confidence: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Percentage of successful associations"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def quality_score(self) -> float:
        """Overall track quality (0-1)"""
        hit_score = min(1.0, self.hit_rate)
        age_score = min(1.0, self.age / 30.0)  # Mature after 30 seconds
        conf_score = self.classification_confidence
        
        return (hit_score * 0.5 + age_score * 0.3 + conf_score * 0.2)

class MultiTargetTracker:
    """
    Multi-target tracking system managing multiple Kalman filters
    """
    
    def __init__(self):
        self.tracks: Dict[str, Track] = {}
        self.track_id_counter = 1
        
        # Association parameters
        self.max_association_distance = 10.0  # km
        self.max_missed_detections = 5
        self.min_hits_for_confirmation = 3
        self.max_track_age_without_update = 30.0  # seconds
        
        # Tracking statistics
        self.total_tracks_created = 0
        self.total_tracks_terminated = 0
        self.current_time = 0.0
        
    def update(self, detections: List[DetectedTarget], timestamp: float) -> Dict[str, Track]:
        """
        Update all tracks with new detections
        
        Args:
            detections: List of detected targets
            timestamp: Current time
            
        Returns:
            Dictionary of active tracks
        """
        self.current_time = timestamp
        dt = timestamp - max([t.last_update for t in self.tracks.values()], default=timestamp-1.0)
        dt = max(0.1, min(dt, 10.0))  # Clamp time step to reasonable range
        
        print(f"\n--- Multi-Target Update at t={timestamp:.1f}s ---")
        print(f"Input: {len(detections)} detections, {len(self.tracks)} active tracks")
        
        # Step 1: Predict all existing tracks
        self.predict_tracks(dt)
        
        # Step 2: Associate detections with tracks
        associations, unassociated_detections = self.associate_detections(detections)
        
        # Step 3: Update associated tracks
        self.update_associated_tracks(associations, timestamp)
        
        # Step 4: Handle missed detections
        self.handle_missed_detections()
        
        # Step 5: Initialize new tracks from unassociated detections
        self.initialize_new_tracks(unassociated_detections, timestamp)
        
        # Step 6: Manage track lifecycle
        self.manage_track_lifecycle()
        
        # Step 7: Update track classifications
        self.update_track_classifications()
        
        print(f"Output: {len([t for t in self.tracks.values() if t.confirmed])} confirmed tracks")
        
        return {tid: track for tid, track in self.tracks.items() if not track.terminated}
    
    def predict_tracks(self, dt: float):
        """Predict all track states forward in time"""
        for track in self.tracks.values():
            if not track.terminated:
                predicted_state = track.kalman_filter.predict(dt)
                predicted_state.timestamp = self.current_time
                track.state = predicted_state
    
    def associate_detections(self, detections: List[DetectedTarget]) -> Tuple[Dict[str, DetectedTarget], List[DetectedTarget]]:
        """
        Associate detections with existing tracks using nearest neighbor
        
        Returns:
            (associations, unassociated_detections)
        """
        associations = {}
        unassociated_detections = []
        used_detections = set()
        
        # Calculate distance matrix
        active_tracks = [(tid, track) for tid, track in self.tracks.items() 
                        if not track.terminated]
        
        if not active_tracks or not detections:
            return {}, detections
        
        # Simple nearest neighbor association
        for track_id, track in active_tracks:
            best_detection = None
            best_distance = float('inf')
            best_idx = -1
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                # Calculate distance between track prediction and detection
                distance = self.calculate_association_distance(track, detection)
                
                if distance < self.max_association_distance and distance < best_distance:
                    best_detection = detection
                    best_distance = distance
                    best_idx = i
            
            if best_detection is not None:
                associations[track_id] = best_detection
                used_detections.add(best_idx)
                print(f"  Associated {track_id} with detection at ({best_detection.range_km:.1f}, {best_detection.bearing_deg:.1f}) dist={best_distance:.1f}km")
        
        # Collect unassociated detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                unassociated_detections.append(detection)
        
        print(f"  Associations: {len(associations)}, Unassociated: {len(unassociated_detections)}")
        return associations, unassociated_detections
    
    def calculate_association_distance(self, track: Track, detection: DetectedTarget) -> float:
        """Calculate distance between track prediction and detection"""
        # Convert detection to Cartesian coordinates
        det_x = detection.range_km * np.sin(np.radians(detection.bearing_deg))
        det_y = detection.range_km * np.cos(np.radians(detection.bearing_deg))
        
        # Calculate Euclidean distance
        dx = track.state.x - det_x
        dy = track.state.y - det_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        return distance
    
    