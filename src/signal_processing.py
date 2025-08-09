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
        
        #signal processing history
        self.signal_history = []
        self.filtered_history = []
        
    def add_noise_to_signal(self, clean_signal: float, noise_level: float = 0.1) -> float:
        """Add realistic radar noise to a clean signal"""
        #thermal noise (always present)
        thermal_noise = np.random.normal(0, noise_level)
        
        #clutter noise (ground/sea returns)
        clutter_noise = np.random.exponential(noise_level * 0.5) if random.random() < 0.3 else 0
        
        #electronic interference
        interference = np.random.uniform(-noise_level, noise_level) if random.random() < 0.1 else 0
        
        noisy_signal = clean_signal + thermal_noise + clutter_noise + interference
        return max(0, noisy_signal)  #signal strength can't be negative
    
    def moving_average_filter(self, signals: List[float], window_size: int = None) -> List[float]:
        """Apply moving average filter to reduce noise"""
        if window_size is None:
            window_size = self.filter_window_size
            
        if len(signals) < window_size:
            return signals.copy()
        
        filtered = []
        for i in range(len(signals)):
            if i < window_size - 1:
                window_data = signals[:i+1]
            else:
                window_data = signals[i-window_size+1:i+1]
            
            filtered.append(np.mean(window_data))
        
        return filtered
    
    def exponential_filter(self, signals: List[float], alpha: float = 0.3) -> List[float]:
        """Apply exponential smoothing filter"""
        if not signals:
            return []
        
        filtered = [signals[0]]
        
        for i in range(1, len(signals)):
            #exponential smoothing: new_value = alpha * current + (1-alpha) * previous
            smoothed = alpha * signals[i] + (1 - alpha) * filtered[-1]
            filtered.append(smoothed)
        
        return filtered
    
    def threshold_detection(self, signals: List[float], threshold: float = None) -> List[bool]:
        """Detect targets based on signal threshold"""
        if threshold is None:
            threshold = self.detection_threshold
        
        #adaptive threshold based on noise floor
        noise_estimate = np.percentile(signals, 25) if signals else 0  #percentile25 as noise estimate
        adaptive_threshold = max(threshold, noise_estimate * 3)  #at least 3x noise level
        
        detections = [signal_val > adaptive_threshold for signal_val in signals]
        return detections
    
    def calculate_snr(self, signal_strength: float, noise_level: float) -> float:
        """Calculate Signal-to-Noise Ratio in dB"""
        if noise_level <= 0:
            return float('inf')
        
        snr_linear = signal_strength / noise_level
        snr_db = 10 * np.log10(max(snr_linear, 1e-10))  #avoid log(0)
        return snr_db

    def radar_range_equation(self, target_rcs: float, range_km: float, 
                           radar_power: float = 1000000, #onemw
                           antenna_gain: float = 40,      #gain40db
                           frequency_ghz: float = 10) -> float:
        """Calculate received signal strength using radar range equation"""
        
        range_m = range_km * 1000
        antenna_gain_linear = 10 ** (antenna_gain / 10)
        
        #radar range equation: Pr = (Pt * G^2 * λ^2 * σ) / ((4π)^3 * R^4)
        #where: Pt=transmit power, G=antenna gain, λ=wavelength, σ=RCS, R=range
        
        wavelength = 3e8 / (frequency_ghz * 1e9)  # c / f
        
        numerator = radar_power * (antenna_gain_linear ** 2) * (wavelength ** 2) * target_rcs
        denominator = ((4 * np.pi) ** 3) * (range_m ** 4)
        
        received_power = numerator / denominator
        
        #normalize to 0-1 range for easier processing
        normalized_strength = min(1.0, received_power * 1e12)  #scalefactor
        
        return normalized_strength
    
    def process_radar_sweep(self, raw_detections: List[Dict]) -> List[RadarReturn]:
        """Process raw radar detections through signal processing pipeline"""
        processed_returns = []
        
        for detection in raw_detections:
            range_km = detection.get('range', 0)
            bearing = detection.get('bearing', 0)
            target = detection.get('target')
            timestamp = detection.get('detection_time', 0)
            
            if target:
                signal_strength = self.radar_range_equation(
                    target_rcs=target.radar_cross_section,
                    range_km=range_km
                )
            else:
                signal_strength = random.uniform(0.1, 0.4)
            
            noise_level = self.noise_floor + random.uniform(0, 0.05)
            noisy_signal = self.add_noise_to_signal(signal_strength, noise_level)
            
            doppler_shift = 0
            if target and hasattr(target, 'speed'):
                #simplified doppler: f_shift = 2 * v * cos(angle) / λ
                #for target moving toward/away from radar
                radial_velocity = target.speed * np.cos(np.radians(target.heading - bearing))
                wavelength = 0.03  #ten ghz radar
                doppler_shift = 2 * radial_velocity * 1000/3600 / wavelength  # Convert km/h to m/s
            
            radar_return = RadarReturn(
                range_km=range_km,
                bearing_deg=bearing,
                signal_strength=noisy_signal,
                noise_level=noise_level,
                timestamp=timestamp,
                doppler_shift=doppler_shift,
                is_valid=True
            )
            
            processed_returns.append(radar_return)
        
        return processed_returns
    
    def filter_detections(self, radar_returns: List[RadarReturn]) -> List[RadarReturn]:
        """Apply filtering to remove noise and false alarms"""
        if not radar_returns:
            return []
        
        signals = [r.signal_strength for r in radar_returns]
        
        filtered_signals = self.moving_average_filter(signals)
        
        smoothed_signals = self.exponential_filter(filtered_signals)
        
        valid_detections = self.threshold_detection(smoothed_signals)
        
        filtered_returns = []
        for i, (radar_return, is_valid, filtered_signal) in enumerate(
            zip(radar_returns, valid_detections, smoothed_signals)):
            
            if is_valid:
                snr = self.calculate_snr(filtered_signal, radar_return.noise_level)
                
                if snr > 6:  #min snr 6 db
                    filtered_return = RadarReturn(
                        range_km=radar_return.range_km,
                        bearing_deg=radar_return.bearing_deg,
                        signal_strength=filtered_signal,
                        noise_level=radar_return.noise_level,
                        timestamp=radar_return.timestamp,
                        doppler_shift=radar_return.doppler_shift,
                        is_valid=True
                    )
                    filtered_returns.append(filtered_return)
        
        return filtered_returns

#test functions
def test_signal_processing():
    """Test the signal processing functionality"""
    print("Testing Signal Processing System")
    print("=" * 40)
    
    processor = SignalProcessor()
    
    # Test 1: Noise addition
    print("\n1. Testing noise addition:")
    clean_signal = 0.8
    for i in range(5):
        noisy = processor.add_noise_to_signal(clean_signal, noise_level=0.1)
        print(f"   Clean: {clean_signal:.3f} → Noisy: {noisy:.3f}")
    
    # Test 2: Moving average filter
    print("\n2. Testing moving average filter:")
    noisy_signals = [0.1, 0.8, 0.2, 0.9, 0.15, 0.85, 0.25, 0.95]
    filtered = processor.moving_average_filter(noisy_signals, window_size=3)
    print(f"   Original:  {[f'{x:.2f}' for x in noisy_signals]}")
    print(f"   Filtered:  {[f'{x:.2f}' for x in filtered]}")
    
    # Test 3: Radar range equation
    print("\n3. Testing radar range equation:")
    test_ranges = [50, 100, 150, 200]
    test_rcs = [20, 100, 500]  # aircraft, large aircraft, ship
    
    for rcs in test_rcs:
        print(f"\n   Target RCS: {rcs} m²")
        for range_km in test_ranges:
            strength = processor.radar_range_equation(rcs, range_km)
            print(f"     {range_km:3d} km: {strength:.4f}")
    
    # Test 4: SNR calculation
    print("\n4. Testing SNR calculation:")
    test_signals = [0.8, 0.4, 0.2, 0.1]
    noise = 0.05
    for sig in test_signals:
        snr = processor.calculate_snr(sig, noise)
        print(f"   Signal: {sig:.2f}, Noise: {noise:.2f} → SNR: {snr:.1f} dB")
    
    print("\n✅ Signal processing tests complete!")

if __name__ == "__main__":
    test_signal_processing()