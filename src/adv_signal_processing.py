"""
Day 8 Task 1: Enhanced Signal Processing
Advanced filtering techniques, clutter rejection, interference mitigation
Professional-grade radar signal processing capabilities
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from typing import List, Dict, Tuple, Optional
import random
import matplotlib.pyplot as plt

# Import shared types from signal_types module
from signal_types import RadarReturn, FilterType, ClutterType, InterferenceType

class AdvancedSignalProcessor:
    """Professional-grade radar signal processor with advanced capabilities"""
    
    def __init__(self):
        # Basic parameters
        self.detection_threshold = 0.12
        self.noise_floor = 0.04
        self.false_alarm_rate = 0.001
        
        # Advanced processing parameters
        self.mti_filter_order = 3
        self.cfar_window_size = 16
        self.cfar_guard_cells = 4
        self.doppler_resolution = 32  # FFT bins for doppler processing
        self.adaptive_threshold_window = 50
        
        # Clutter and interference parameters
        self.clutter_suppression_db = 30
        self.interference_threshold = 0.8
        self.sidelobe_suppression_db = 25
        
        # Filter histories for adaptive processing
        self.signal_history = []
        self.clutter_map = {}  # Range-bearing clutter map
        self.interference_statistics = {}
        self.adaptive_weights = np.ones(10) * 0.1  # Adaptive filter weights
        
        # Performance metrics
        self.processing_stats = {
            'signals_processed': 0,
            'clutter_detections': 0,
            'interference_events': 0,
            'false_alarms_suppressed': 0,
            'valid_detections': 0
        }
        
        print("🔬 Advanced Signal Processor initialized")
        print(f"   • MTI filter order: {self.mti_filter_order}")
        print(f"   • CFAR window: {self.cfar_window_size} cells")
        print(f"   • Doppler bins: {self.doppler_resolution}")
        print(f"   • Clutter suppression: {self.clutter_suppression_db} dB")
    
    def generate_realistic_clutter(self, range_km: float, bearing_deg: float, 
                                  environment_type: str = "mixed") -> Tuple[float, ClutterType]:
        """Generate realistic radar clutter based on environment"""
        
        # Distance-dependent clutter (stronger at short ranges)
        range_factor = np.exp(-range_km / 50.0)
        
        # Bearing-dependent effects (terrain shadowing)
        bearing_factor = 1.0 + 0.3 * np.sin(np.radians(bearing_deg * 3))
        
        if environment_type == "coastal":
            # Sea clutter dominant
            if range_km < 30:
                clutter_power = 0.4 * range_factor * (1 + 0.6 * np.random.random())
                clutter_type = ClutterType.SEA
            else:
                clutter_power = 0.1 * range_factor
                clutter_type = ClutterType.GROUND
                
        elif environment_type == "urban":
            # Strong ground clutter with multipath
            clutter_power = 0.6 * range_factor * bearing_factor
            clutter_type = ClutterType.URBAN
            
        elif environment_type == "mountainous":
            # Variable ground clutter
            clutter_power = 0.3 * range_factor * (0.5 + bearing_factor)
            clutter_type = ClutterType.GROUND
            
        elif environment_type == "weather":
            # Weather clutter
            clutter_power = 0.5 * np.random.exponential(0.3)
            clutter_type = ClutterType.WEATHER
            
        else:  # mixed environment
            # Random mix of clutter types
            rand_factor = np.random.random()
            if rand_factor < 0.4:
                clutter_power = 0.2 * range_factor
                clutter_type = ClutterType.GROUND
            elif rand_factor < 0.7:
                clutter_power = 0.15 * range_factor
                clutter_type = ClutterType.SEA
            else:
                clutter_power = 0.1
                clutter_type = ClutterType.WEATHER
        
        # Add temporal variation
        clutter_power *= (1 + 0.2 * np.sin(self.processing_stats['signals_processed'] * 0.1))
        
        return max(0, clutter_power), clutter_type
    
    def generate_electronic_interference(self, timestamp: float) -> Tuple[float, InterferenceType]:
        """Generate electronic warfare interference"""
        
        interference_power = 0.0
        interference_type = InterferenceType.ATMOSPHERIC
        
        # Periodic jamming (simulating electronic warfare)
        if np.sin(timestamp * 2.0) > 0.8:  # 20% duty cycle jamming
            interference_power = 0.7 + 0.3 * np.random.random()
            interference_type = InterferenceType.JAMMING
            self.processing_stats['interference_events'] += 1
            
        # Multipath interference (range-dependent)
        elif np.random.random() < 0.05:  # 5% probability
            interference_power = 0.3 + 0.2 * np.random.random()
            interference_type = InterferenceType.MULTIPATH
            
        # Sidelobe interference
        elif np.random.random() < 0.02:  # 2% probability
            interference_power = 0.2 + 0.3 * np.random.random()
            interference_type = InterferenceType.SIDELOBE
            
        # Atmospheric noise
        else:
            interference_power = 0.02 * np.random.random()
            interference_type = InterferenceType.ATMOSPHERIC
            
        return interference_power, interference_type
    
    def mti_filter(self, signal_samples: np.ndarray, filter_order: int = None) -> np.ndarray:
        """Moving Target Indication filter to suppress stationary clutter"""
        if filter_order is None:
            filter_order = self.mti_filter_order
        
        if len(signal_samples) < filter_order + 1:
            return signal_samples
        
        # Design MTI filter (differencing filter)
        mti_coeffs = np.zeros(filter_order + 1)
        mti_coeffs[0] = 1
        mti_coeffs[-1] = -1
        
        # Apply filter
        filtered_signal = signal.lfilter(mti_coeffs, [1], signal_samples)
        
        return filtered_signal
    
    def cfar_detection(self, signal_samples: np.ndarray, guard_cells: int = None, 
                      window_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Constant False Alarm Rate detection"""
        if guard_cells is None:
            guard_cells = self.cfar_guard_cells
        if window_size is None:
            window_size = self.cfar_window_size
        
        detections = np.zeros_like(signal_samples, dtype=bool)
        thresholds = np.zeros_like(signal_samples)
        
        half_window = window_size // 2
        
        for i in range(len(signal_samples)):
            # Define reference window (excluding guard cells)
            left_start = max(0, i - half_window - guard_cells)
            left_end = max(0, i - guard_cells)
            right_start = min(len(signal_samples), i + guard_cells + 1)
            right_end = min(len(signal_samples), i + half_window + guard_cells + 1)
            
            # Calculate noise power from reference cells
            reference_cells = np.concatenate([
                signal_samples[left_start:left_end],
                signal_samples[right_start:right_end]
            ])
            
            if len(reference_cells) > 0:
                noise_power = np.mean(reference_cells)
                # CFAR threshold with probability of false alarm
                cfar_threshold = noise_power * (-np.log(self.false_alarm_rate))
                thresholds[i] = cfar_threshold
                
                # Detection decision
                detections[i] = signal_samples[i] > cfar_threshold
            
        return detections, thresholds
    
    def doppler_filter_bank(self, signal_samples: np.ndarray) -> Dict[str, np.ndarray]:
        """Doppler filter bank for velocity discrimination"""
        
        # Ensure we have enough samples for FFT
        fft_size = max(self.doppler_resolution, len(signal_samples))
        padded_signal = np.pad(signal_samples, (0, fft_size - len(signal_samples)), 'constant')
        
        # Apply window function
        windowed_signal = padded_signal * np.hanning(len(padded_signal))
        
        # Compute FFT for doppler analysis
        doppler_spectrum = fft(windowed_signal)
        doppler_magnitude = np.abs(doppler_spectrum)
        doppler_frequencies = fftfreq(len(doppler_spectrum))
        
        # Separate velocity bins
        velocity_bins = {
            'stationary': doppler_magnitude[np.abs(doppler_frequencies) < 0.05],
            'slow_moving': doppler_magnitude[(np.abs(doppler_frequencies) >= 0.05) & 
                                           (np.abs(doppler_frequencies) < 0.2)],
            'fast_moving': doppler_magnitude[np.abs(doppler_frequencies) >= 0.2]
        }
        
        return {
            'spectrum': doppler_spectrum,
            'magnitude': doppler_magnitude,
            'frequencies': doppler_frequencies,
            'velocity_bins': velocity_bins
        }
    
    def adaptive_interference_cancellation(self, signal_samples: np.ndarray, 
                                         reference_samples: np.ndarray) -> np.ndarray:
        """Adaptive interference cancellation using LMS algorithm"""
        
        if len(reference_samples) != len(signal_samples):
            # Pad or trim reference to match signal length
            if len(reference_samples) > len(signal_samples):
                reference_samples = reference_samples[:len(signal_samples)]
            else:
                reference_samples = np.pad(reference_samples, 
                                         (0, len(signal_samples) - len(reference_samples)), 
                                         'constant')
        
        # Adaptive filter parameters
        mu = 0.01  # Step size
        filter_order = min(10, len(signal_samples) // 2)
        
        # Initialize adaptive filter
        w = np.zeros(filter_order)  # Filter weights
        output = np.zeros(len(signal_samples))
        
        # LMS adaptive filtering
        for n in range(filter_order, len(signal_samples)):
            # Reference input vector
            x = reference_samples[n-filter_order:n]
            
            # Filter output
            y = np.dot(w, x)
            
            # Error signal
            error = signal_samples[n] - y
            output[n] = error
            
            # Weight update (LMS algorithm)
            w += mu * error * x
        
        return output
    
    def wiener_filter(self, signal_samples: np.ndarray, noise_estimate: float) -> np.ndarray:
        """Wiener filtering for optimal signal restoration"""
        
        # Estimate signal power spectrum
        signal_fft = fft(signal_samples)
        signal_power = np.abs(signal_fft) ** 2
        
        # Noise power spectrum (assumed white)
        noise_power = noise_estimate ** 2
        
        # Wiener filter frequency response
        H = signal_power / (signal_power + noise_power)
        
        # Apply filter in frequency domain
        filtered_fft = signal_fft * H
        filtered_signal = np.real(ifft(filtered_fft))
        
        return filtered_signal
    
    def process_advanced_detection(self, raw_detections: List[Dict]) -> List[RadarReturn]:
        """Advanced signal processing pipeline with all techniques"""
        
        if not raw_detections:
            return []
        
        print(f"🔬 Processing {len(raw_detections)} raw detections with advanced techniques")
        
        processed_returns = []
        
        for detection in raw_detections:
            self.processing_stats['signals_processed'] += 1
            
            range_km = detection.get('range', 0)
            bearing = detection.get('bearing', 0)
            target = detection.get('target')
            timestamp = detection.get('detection_time', 0)
            
            # Calculate base signal strength
            if target:
                signal_strength = self.calculate_target_signal(target, range_km)
            else:
                signal_strength = random.uniform(0.1, 0.3)  # False alarm
            
            # Generate realistic clutter
            clutter_power, clutter_type = self.generate_realistic_clutter(
                range_km, bearing, "mixed")
            
            # Generate interference
            interference_power, interference_type = self.generate_electronic_interference(timestamp)
            
            # Combine signal components
            total_signal = signal_strength + clutter_power + interference_power
            noise_level = self.noise_floor + random.uniform(0, 0.02)
            
            # Add thermal noise
            noisy_signal = total_signal + np.random.normal(0, noise_level)
            
            # Create signal samples for advanced processing (simulate pulse compression)
            num_samples = 32
            signal_samples = noisy_signal * (1 + 0.1 * np.random.randn(num_samples))
            
            # Apply advanced processing techniques
            
            # 1. MTI filtering for clutter suppression
            if clutter_power > 0.1:
                mti_filtered = self.mti_filter(signal_samples)
                clutter_suppression = min(clutter_power * 0.8, clutter_power)  # 80% suppression
                self.processing_stats['clutter_detections'] += 1
            else:
                mti_filtered = signal_samples
                clutter_suppression = 0
            
            # 2. Adaptive interference cancellation
            if interference_power > 0.3:
                reference_noise = np.random.randn(num_samples) * interference_power
                interference_canceled = self.adaptive_interference_cancellation(
                    mti_filtered, reference_noise)
                interference_suppression = min(interference_power * 0.7, interference_power)
            else:
                interference_canceled = mti_filtered
                interference_suppression = 0
            
            # 3. Wiener filtering for noise reduction
            wiener_filtered = self.wiener_filter(interference_canceled, noise_level)
            
            # 4. CFAR detection
            cfar_detections, cfar_thresholds = self.cfar_detection(np.abs(wiener_filtered))
            is_cfar_detection = np.any(cfar_detections)
            
            # 5. Doppler analysis
            doppler_analysis = self.doppler_filter_bank(wiener_filtered)
            
            # Calculate final processed signal strength
            final_signal = np.mean(np.abs(wiener_filtered))
            final_clutter = max(0, clutter_power - clutter_suppression)
            final_interference = max(0, interference_power - interference_suppression)
            final_noise = noise_level * 0.5  # Wiener filter reduces noise
            
            # Calculate detection confidence
            snr = final_signal / (final_noise + 0.001)
            confidence = min(1.0, snr / 5.0) if is_cfar_detection else 0.1
            
            # Determine doppler shift from analysis
            doppler_shift = self.estimate_doppler_shift(doppler_analysis)
            
            # Create enhanced radar return
            radar_return = RadarReturn(
                range_km=range_km,
                bearing_deg=bearing,
                signal_strength=final_signal,
                noise_level=final_noise,
                timestamp=timestamp,
                doppler_shift=doppler_shift,
                clutter_level=final_clutter,
                interference_level=final_interference,
                is_valid=is_cfar_detection and confidence > 0.3,
                confidence=confidence
            )
            
            if radar_return.is_valid:
                self.processing_stats['valid_detections'] += 1
                processed_returns.append(radar_return)
            else:
                self.processing_stats['false_alarms_suppressed'] += 1
        
        print(f"✅ Advanced processing complete:")
        print(f"   • Valid detections: {len(processed_returns)}")
        print(f"   • Clutter events: {self.processing_stats['clutter_detections']}")
        print(f"   • Interference events: {self.processing_stats['interference_events']}")
        print(f"   • False alarms suppressed: {self.processing_stats['false_alarms_suppressed']}")
        
        return processed_returns
    
    def calculate_target_signal(self, target, range_km: float) -> float:
        """Calculate target signal strength using radar range equation"""
        if hasattr(target, 'radar_cross_section'):
            rcs = target.radar_cross_section
        else:
            rcs = 10.0  # Default RCS
        
        # Radar range equation (simplified)
        # P_r = P_t * G^2 * λ^2 * σ / ((4π)^3 * R^4)
        range_m = range_km * 1000
        if range_m > 0:
            power_factor = rcs / (range_m ** 4)
            signal_strength = min(1.0, max(0.01, power_factor * 1e12))  # Scale factor
        else:
            signal_strength = 1.0
        
        return signal_strength
    
    def estimate_doppler_shift(self, doppler_analysis: Dict) -> float:
        """Estimate doppler shift from frequency analysis"""
        magnitude = doppler_analysis['magnitude']
        frequencies = doppler_analysis['frequencies']
        
        # Find peak frequency
        peak_idx = np.argmax(magnitude)
        peak_frequency = frequencies[peak_idx]
        
        # Convert to doppler shift (simplified)
        # Doppler shift = 2 * v * f_c / c, where v is radial velocity
        # For simulation, we'll scale the frequency appropriately
        doppler_shift = peak_frequency * 1000  # Scale to Hz
        
        return doppler_shift
    
    def get_processing_statistics(self) -> Dict:
        """Get comprehensive processing statistics"""
        total_processed = self.processing_stats['signals_processed']
        
        if total_processed > 0:
            detection_rate = self.processing_stats['valid_detections'] / total_processed
            clutter_rate = self.processing_stats['clutter_detections'] / total_processed
            interference_rate = self.processing_stats['interference_events'] / total_processed
            false_alarm_rate = self.processing_stats['false_alarms_suppressed'] / total_processed
        else:
            detection_rate = clutter_rate = interference_rate = false_alarm_rate = 0
        
        return {
            'total_signals_processed': total_processed,
            'valid_detections': self.processing_stats['valid_detections'],
            'detection_rate': detection_rate,
            'clutter_events': self.processing_stats['clutter_detections'],
            'clutter_rate': clutter_rate,
            'interference_events': self.processing_stats['interference_events'],
            'interference_rate': interference_rate,
            'false_alarms_suppressed': self.processing_stats['false_alarms_suppressed'],
            'false_alarm_suppression_rate': false_alarm_rate,
            'filter_performance': {
                'mti_order': self.mti_filter_order,
                'cfar_window': self.cfar_window_size,
                'clutter_suppression_db': self.clutter_suppression_db,
                'noise_floor': self.noise_floor
            }
        }

def test_advanced_signal_processing():
    """Test the advanced signal processing capabilities"""
    print("🧪 TESTING ADVANCED SIGNAL PROCESSING")
    print("=" * 60)
    
    # Create processor
    processor = AdvancedSignalProcessor()
    
    # Create test data
    test_detections = []
    for i in range(20):
        detection = {
            'range': 50 + i * 10,
            'bearing': i * 18,  # 18 degree spacing
            'target': type('Target', (), {
                'radar_cross_section': 10 + i * 2
            })(),
            'detection_time': i * 0.1
        }
        test_detections.append(detection)
    
    # Add some false alarms
    for i in range(5):
        false_alarm = {
            'range': 30 + i * 15,
            'bearing': 45 + i * 30,
            'target': None,  # No actual target
            'detection_time': i * 0.2
        }
        test_detections.append(false_alarm)
    
    print(f"Created {len(test_detections)} test detections")
    
    # Process through advanced pipeline
    processed_returns = processor.process_advanced_detection(test_detections)
    
    print(f"\n📊 PROCESSING RESULTS:")
    stats = processor.get_processing_statistics()
    
    print(f"Total processed: {stats['total_signals_processed']}")
    print(f"Valid detections: {stats['valid_detections']}")
    print(f"Detection rate: {stats['detection_rate']:.2%}")
    print(f"Clutter events: {stats['clutter_events']} ({stats['clutter_rate']:.2%})")
    print(f"Interference events: {stats['interference_events']} ({stats['interference_rate']:.2%})")
    print(f"False alarms suppressed: {stats['false_alarms_suppressed']} ({stats['false_alarm_suppression_rate']:.2%})")
    
    print(f"\n🔧 FILTER CONFIGURATION:")
    filter_config = stats['filter_performance']
    print(f"MTI filter order: {filter_config['mti_order']}")
    print(f"CFAR window size: {filter_config['cfar_window']} cells")
    print(f"Clutter suppression: {filter_config['clutter_suppression_db']} dB")
    print(f"Noise floor: {filter_config['noise_floor']:.3f}")
    
    print("\n✅ Advanced signal processing test complete!")
    
    return processor, processed_returns

if __name__ == "__main__":
    processor, results = test_advanced_signal_processing()
    
    if results:
        print(f"\n📋 SAMPLE PROCESSED RETURNS:")
        for i, ret in enumerate(results[:5]):  # Show first 5
            print(f"  {i+1}. Range: {ret.range_km:5.1f}km, "
                  f"Bearing: {ret.bearing_deg:5.1f}°, "
                  f"Signal: {ret.signal_strength:.3f}, "
                  f"Confidence: {ret.confidence:.2f}")
            if ret.clutter_level > 0:
                print(f"      Clutter: {ret.clutter_level:.3f}")
            if ret.interference_level > 0:
                print(f"      Interference: {ret.interference_level:.3f}")