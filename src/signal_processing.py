"""
Radar signal processing algorithms
"""
import numpy as np
from scipy import signal
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
from enum import Enum

@dataclass
class RadarReturn:
    """Single radar return/detection"""
    range_km: float
    bearing_deg: float
    signal_strength: float
    noise_level: float
    timestamp: float
    doppler_shift: float = 0.0
    is_valid: bool = True

class FilterType(Enum):
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    KALMAN = "kalman"
    THRESHOLD = "threshold"

class SignalProcessor:
    def __init__(self):
        self.detection_threshold = 0.5
        self.noise_floor = 0.1
        self.false_alarm_rate = 0.02
        self.filter_window_size = 5
        
        # Signal processing history
        self.signal_history = []
        self.filtered_history = []
        
    def add_noise_to_signal(self, clean_signal: float, noise_level: float = 0.1) -> float:
        """Add realistic radar noise to a clean signal"""
        # Thermal noise (always present)
        thermal_noise = np.random.normal(0, noise_level)
        
        # Clutter noise (ground/sea returns)
        clutter_noise = np.random.exponential(noise_level * 0.5) if random.random() < 0.3 else 0
        
        # Electronic interference
        interference = np.random.uniform(-noise_level, noise_level) if random.random() < 0.1 else 0
        
        noisy_signal = clean_signal + thermal_noise + clutter_noise + interference
        return max(0, noisy_signal)  # Signal strength can't be negative
    
    def moving_average_filter(self, signals: List[float], window_size: int = None) -> List[float]:
        """Apply moving average filter to reduce noise"""
        if window_size is None:
            window_size = self.filter_window_size
            
        if len(signals) < window_size:
            return signals.copy()
        
        filtered = []
        for i in range(len(signals)):
            if i < window_size - 1:
                # Not enough data for full window, use available data
                window_data = signals[:i+1]
            else:
                # Full window available
                window_data = signals[i-window_size+1:i+1]
            
            filtered.append(np.mean(window_data))
        
        return filtered
    
    def exponential_filter(self, signals: List[float], alpha: float = 0.3) -> List[float]:
        """Apply exponential smoothing filter"""
        if not signals:
            return []
        
        filtered = [signals[0]]  # First value unchanged
        
        for i in range(1, len(signals)):
            # Exponential smoothing: new_value = alpha * current + (1-alpha) * previous
            smoothed = alpha * signals[i] + (1 - alpha) * filtered[-1]
            filtered.append(smoothed)
        
        return filtered
    
    def threshold_detection(self, signals: List[float], threshold: float = None) -> List[bool]:
        """Detect targets based on signal threshold"""
        if threshold is None:
            threshold = self.detection_threshold
        
        # Adaptive threshold based on noise floor
        noise_estimate = np.percentile(signals, 25) if signals else 0  # 25th percentile as noise estimate
        adaptive_threshold = max(threshold, noise_estimate * 3)  # At least 3x noise level
        
        detections = [signal_val > adaptive_threshold for signal_val in signals]
        return detections
    
    def calculate_snr(self, signal_strength: float, noise_level: float) -> float:
        """Calculate Signal-to-Noise Ratio in dB"""
        if noise_level <= 0:
            return float('inf')
        
        snr_linear = signal_strength / noise_level
        snr_db = 10 * np.log10(max(snr_linear, 1e-10))  # Avoid log(0)
        return snr_db