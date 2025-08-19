"""
Performance Optimization System - Day 7 Task 3
===============================================
Advanced performance optimization for 60+ FPS radar operation.
Implements adaptive quality, efficient memory management, and real-time optimization.

Features:
• Adaptive frame rate management with quality scaling
• Efficient memory usage and garbage collection
• Real-time performance monitoring and adjustment
• Multi-threading for heavy computations
• Optimized rendering pipeline
• Smart caching and data structures
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Wedge, FancyBboxPatch
import time
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
import gc
import psutil
import weakref
from collections import deque, defaultdict

# Import radar components
try:
    from src.radar_data_generator import RadarDataGenerator
    from src.signal_processing import SignalProcessor
    from src.target_detection import TargetDetector
    from src.multi_target_tracker import MultiTargetTracker
    print("✅ All radar components imported successfully")
except ImportError as e:
    print(f"⚠️  Warning: Some components not found: {e}")

class PerformanceLevel(Enum):
    """System performance levels"""
    MAXIMUM = "Maximum Quality"
    HIGH = "High Quality"
    BALANCED = "Balanced"
    PERFORMANCE = "Performance Mode"
    MINIMUM = "Minimum Quality"

class OptimizationMode(Enum):
    """Optimization strategies"""
    REAL_TIME = "Real-Time Priority"
    QUALITY = "Quality Priority"
    BALANCED = "Balanced"
    POWER_SAVE = "Power Save"

@dataclass
class PerformanceSettings:
    """Performance optimization settings"""
    target_fps: float = 60.0
    max_targets: int = 100
    trail_points: int = 50
    sweep_history: int = 30
    update_interval_ms: int = 16  # ~60 FPS
    
    # Rendering optimizations
    adaptive_quality: bool = True
    level_of_detail: bool = True
    culling_enabled: bool = True
    batch_rendering: bool = True
    
    # Memory optimizations
    garbage_collect_interval: int = 100  # frames
    cache_size_mb: float = 50.0
    object_pooling: bool = True
    
    # Threading settings
    async_processing: bool = True
    thread_pool_size: int = 2
    processing_queue_size: int = 10

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    frame_rate: float = 0.0
    frame_time_ms: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage: float = 0.0
    
    # Rendering metrics
    triangles_rendered: int = 0
    objects_culled: int = 0
    cache_hit_rate: float = 0.0
    
    # Processing metrics
    detection_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    rendering_time_ms: float = 0.0
    
    # Quality metrics
    quality_level: float = 1.0
    lod_level: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            'frame_rate': self.frame_rate,
            'frame_time_ms': self.frame_time_ms,
            'cpu_usage': self.cpu_usage,
            'memory_usage_mb': self.memory_usage_mb,
            'detection_time_ms': self.detection_time_ms,
            'tracking_time_ms': self.tracking_time_ms,
            'rendering_time_ms': self.rendering_time_ms,
            'quality_level': self.quality_level
        }

class PerformanceProfiler:
    """Advanced performance profiling and monitoring"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.frame_times = deque(maxlen=history_size)
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.quality_history = deque(maxlen=history_size)
        
        self.frame_start_time = 0.0
        self.current_metrics = PerformanceMetrics()
        
        # Performance counters
        self.total_frames = 0
        self.dropped_frames = 0
        self.optimization_count = 0
        
        # System monitoring
        self.process = psutil.Process()
        
    def start_frame(self):
        """Mark the start of a frame"""
        self.frame_start_time = time.perf_counter()
        
    def end_frame(self) -> PerformanceMetrics:
        """Mark the end of a frame and calculate metrics"""
        frame_end_time = time.perf_counter()
        frame_time = frame_end_time - self.frame_start_time
        
        # Update metrics
        self.current_metrics.frame_time_ms = frame_time * 1000
        self.current_metrics.frame_rate = 1.0 / max(frame_time, 0.001)
        
        # System metrics
        self.current_metrics.cpu_usage = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        self.current_metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
        
        # Update history
        self.frame_times.append(frame_time)
        self.cpu_history.append(self.current_metrics.cpu_usage)
        self.memory_history.append(self.current_metrics.memory_usage_mb)
        self.quality_history.append(self.current_metrics.quality_level)
        
        self.total_frames += 1
        
        return self.current_metrics
    
    def get_average_fps(self, window_size: int = 30) -> float:
        """Get average FPS over recent frames"""
        if len(self.frame_times) < window_size:
            return 0.0
        recent_times = list(self.frame_times)[-window_size:]
        avg_time = sum(recent_times) / len(recent_times)
        return 1.0 / max(avg_time, 0.001)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get comprehensive performance statistics"""
        if not self.frame_times:
            return {}
            
        frame_times_list = list(self.frame_times)
        
        return {
            'avg_fps': self.get_average_fps(),
            'min_fps': 1.0 / max(frame_times_list),
            'max_fps': 1.0 / min(frame_times_list),
            'avg_frame_time_ms': np.mean(frame_times_list) * 1000,
            'frame_time_99p_ms': np.percentile(frame_times_list, 99) * 1000,
            'avg_cpu_usage': np.mean(list(self.cpu_history)) if self.cpu_history else 0,
            'avg_memory_mb': np.mean(list(self.memory_history)) if self.memory_history else 0,
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'frame_drop_rate': self.dropped_frames / max(self.total_frames, 1) * 100
        }

class AdaptiveQualityManager:
    """Manages adaptive quality based on performance"""
    
    def __init__(self, settings: PerformanceSettings):
        self.settings = settings
        self.current_quality = 1.0
        self.target_fps = settings.target_fps
        
        # Quality levels
        self.quality_levels = {
            0.25: PerformanceLevel.MINIMUM,
            0.5: PerformanceLevel.PERFORMANCE,
            0.75: PerformanceLevel.BALANCED,
            0.9: PerformanceLevel.HIGH,
            1.0: PerformanceLevel.MAXIMUM
        }
        
        # Adjustment parameters
        self.adjustment_rate = 0.1
        self.stability_threshold = 5  # frames
        self.quality_history = deque(maxlen=self.stability_threshold)
        
    def update_quality(self, current_fps: float, frame_time_ms: float) -> float:
        """Update quality level based on performance"""
        fps_ratio = current_fps / self.target_fps
        
        # Determine if we need to adjust quality
        if fps_ratio < 0.8:  # Below 80% of target FPS
            # Decrease quality for better performance
            target_quality = max(0.25, self.current_quality - self.adjustment_rate)
        elif fps_ratio > 1.1 and frame_time_ms < 10:  # Above target with headroom
            # Increase quality if we have performance headroom
            target_quality = min(1.0, self.current_quality + self.adjustment_rate * 0.5)
        else:
            # Maintain current quality
            target_quality = self.current_quality
        
        # Apply smoothing
        self.quality_history.append(target_quality)
        if len(self.quality_history) >= self.stability_threshold:
            # Only change if consistently needed
            avg_target = np.mean(list(self.quality_history))
            if abs(avg_target - self.current_quality) > 0.05:
                self.current_quality = avg_target
        
        return self.current_quality
    
    def get_quality_settings(self) -> Dict[str, Any]:
        """Get settings based on current quality level"""
        quality = self.current_quality
        
        return {
            'trail_points': int(self.settings.trail_points * quality),
            'sweep_history': int(self.settings.sweep_history * quality),
            'detail_level': quality,
            'particle_count': int(100 * quality),
            'shadow_quality': quality > 0.7,
            'anti_aliasing': quality > 0.8,
            'texture_resolution': quality,
            'effects_enabled': quality > 0.5
        }

class ObjectPool:
    """Object pooling for memory optimization"""
    
    def __init__(self, object_type, initial_size: int = 10):
        self.object_type = object_type
        self.available = []
        self.in_use = set()
        
        # Pre-allocate objects
        for _ in range(initial_size):
            obj = object_type()
            self.available.append(obj)
    
    def acquire(self):
        """Get an object from the pool"""
        if self.available:
            obj = self.available.pop()
        else:
            obj = self.object_type()
        
        self.in_use.add(id(obj))
        return obj
    
    def release(self, obj):
        """Return an object to the pool"""
        obj_id = id(obj)
        if obj_id in self.in_use:
            self.in_use.remove(obj_id)
            # Reset object state if needed
            if hasattr(obj, 'reset'):
                obj.reset()
            self.available.append(obj)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        return {
            'available': len(self.available),
            'in_use': len(self.in_use),
            'total': len(self.available) + len(self.in_use)
        }

class OptimizedRadarSystem:
    """
    High-Performance Optimized Radar System
    
    This system implements advanced optimization techniques for smooth
    60+ FPS operation with adaptive quality and efficient resource usage.
    """
    
    def __init__(self):
        print("⚡ Initializing High-Performance Optimized Radar System...")
        
        # Performance management
        self.settings = PerformanceSettings()
        self.profiler = PerformanceProfiler()
        self.quality_manager = AdaptiveQualityManager(self.settings)
        
        # Core components with optimization
        self.data_generator = RadarDataGenerator(max_range_km=200)
        self.signal_processor = SignalProcessor()
        self.target_detector = TargetDetector()
        self.tracker = MultiTargetTracker()
        
        # Apply initial optimizations
        self.apply_optimizations()
        
        # System state
        self.is_running = False
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.frame_count = 0
        
        # Optimized data structures
        self.sweep_history = deque(maxlen=self.settings.sweep_history)
        self.target_trails = defaultdict(lambda: deque(maxlen=self.settings.trail_points))
        self.render_cache = {}
        
        # Threading for async processing
        self.processing_queue = queue.Queue(maxsize=self.settings.processing_queue_size)
        self.result_queue = queue.Queue()
        self.worker_threads = []
        
        if self.settings.async_processing:
            self.start_worker_threads()
        
        # Memory management
        self.gc_counter = 0
        self.last_gc_time = time.time()
        
        # Performance monitoring
        self.performance_history = deque(maxlen=300)  # 5 seconds at 60 FPS
        self.optimization_log = []
        
        # Display components
        self.fig = None
        self.axes = {}
        self.animation = None
        
        self.setup_optimized_display()
        self.load_performance_scenario()
        
    def apply_optimizations(self):
        """Apply performance optimizations to radar components"""
        # Optimize tracker for performance
        self.tracker.max_association_distance = 15.0
        self.tracker.min_hits_for_confirmation = 1
        self.tracker.max_missed_detections = 10
        
        # Optimize signal processor
        self.signal_processor.detection_threshold = 0.08
        self.signal_processor.false_alarm_rate = 0.03
        
        # Optimize target detector
        self.target_detector.min_detections_for_confirmation = 1
        self.target_detector.association_distance_threshold = 12.0
        
        print("⚡ Applied performance optimizations:")
        print("   • Reduced association distances for faster processing")
        print("   • Optimized confirmation thresholds")
        print("   • Enabled adaptive quality management")
        
    def start_worker_threads(self):
        """Start worker threads for async processing"""
        for i in range(self.settings.thread_pool_size):
            thread = threading.Thread(target=self.worker_function, daemon=True)
            thread.start()
            self.worker_threads.append(thread)
        print(f"⚡ Started {self.settings.thread_pool_size} worker threads")
    
    def worker_function(self):
        """Worker thread function for async processing"""
        while True:
            try:
                task = self.processing_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                # Process task
                task_type, data = task
                if task_type == 'detection':
                    result = self.process_detections_async(data)
                    self.result_queue.put(('detection_result', result))
                
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️  Worker thread error: {e}")
    
    def process_detections_async(self, detection_data):
        """Process detections asynchronously"""
        detections, current_time = detection_data
        
        # Process through pipeline
        targets = self.target_detector.process_raw_detections(detections)
        
        if targets:
            # Update tracker
            active_tracks = self.tracker.update(targets, current_time)
            confirmed_tracks = self.tracker.get_confirmed_tracks()
            return confirmed_tracks
        
        return []
    
    def setup_optimized_display(self):
        """Setup optimized display with performance monitoring"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor('black')
        
        # Optimized layout
        gs = self.fig.add_gridspec(3, 5, height_ratios=[3, 1, 1], width_ratios=[3, 1, 1, 1, 1])
        
        # Main radar display
        self.axes['radar'] = self.fig.add_subplot(gs[0, :3], projection='polar')
        self.setup_radar_scope()
        
        # Performance monitoring panels
        self.axes['performance'] = self.fig.add_subplot(gs[0, 3])
        self.axes['optimization'] = self.fig.add_subplot(gs[0, 4])
        
        # Real-time metrics
        self.axes['fps_graph'] = self.fig.add_subplot(gs[1, :2])
        self.axes['cpu_memory'] = self.fig.add_subplot(gs[1, 2])
        self.axes['quality'] = self.fig.add_subplot(gs[1, 3])
        self.axes['threading'] = self.fig.add_subplot(gs[1, 4])
        
        # System controls
        self.axes['controls'] = self.fig.add_subplot(gs[2, :2])
        self.axes['settings'] = self.fig.add_subplot(gs[2, 2])
        self.axes['profiler'] = self.fig.add_subplot(gs[2, 3])
        self.axes['status'] = self.fig.add_subplot(gs[2, 4])
        
        # Style all panels
        for name, ax in self.axes.items():
            if name != 'radar':
                ax.set_facecolor('#001122')
                for spine in ax.spines.values():
                    spine.set_color('#00ff00')
                    spine.set_linewidth(1)
                ax.tick_params(colors='#00ff00', labelsize=8)
        
        # Title
        self.fig.suptitle('HIGH-PERFORMANCE OPTIMIZED RADAR SYSTEM - 60+ FPS TARGET', 
                         fontsize=18, color='#00ff00', weight='bold', y=0.95)
    
    def setup_radar_scope(self):
        """Configure optimized radar PPI scope"""
        ax = self.axes['radar']
        ax.set_facecolor('black')
        ax.set_ylim(0, 200)
        ax.set_title('OPTIMIZED RADAR PPI SCOPE\nAdaptive Quality & Performance', 
                    color='#00ff00', pad=20, fontsize=14, weight='bold')
        
        # Range rings
        for r in [50, 100, 150, 200]:
            circle = Circle((0, 0), r, fill=False, color='#00ff00', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
            ax.text(np.pi/4, r-5, f'{r}km', color='#00ff00', fontsize=10, ha='center')
        
        # Bearing lines
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            ax.plot([rad, rad], [0, 200], color='#00ff00', alpha=0.2, linewidth=0.5)
            ax.text(rad, 210, f'{angle}°', color='#00ff00', fontsize=9, ha='center')
        
        # Configure polar display
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.grid(True, color='#00ff00', alpha=0.2)
        ax.set_rticks([])
        ax.set_thetagrids([])
    
    def load_performance_scenario(self):
        """Load scenario optimized for performance testing"""
        print("📡 Loading performance test scenario...")
        
        # Load enough targets to stress the system
        aircraft_data = [
            (-120, 150, 90, 450),   (-80, 180, 45, 380),   (60, 120, 225, 520),
            (-150, -100, 135, 420), (100, -80, 315, 360),  (0, 180, 180, 600),
            (-90, 60, 270, 280),    (140, 40, 225, 480),   (-60, -140, 45, 340),
            (80, 160, 135, 380),    (-180, 20, 90, 520),   (160, -120, 270, 400)
        ]
        
        for x, y, heading, speed in aircraft_data:
            self.data_generator.add_aircraft(x, y, heading, speed)
            
        # Add ships
        ship_data = [
            (-160, -140, 45, 25), (140, -160, 270, 18), (-80, -180, 90, 35),
            (180, -120, 225, 22), (-120, -100, 135, 28)
        ]
        
        for x, y, heading, speed in ship_data:
            self.data_generator.add_ship(x, y, heading, speed)
            
        # Add weather for processing load
        self.data_generator.add_weather_returns(-100, 100, 40)
        self.data_generator.add_weather_returns(120, 140, 30)
        
        total_targets = len(self.data_generator.targets)
        print(f"✅ Performance test scenario loaded: {total_targets} targets")
        print("   • Stress test configuration for performance validation")
    
    def animate_optimized(self, frame):
        """Optimized animation loop with performance monitoring"""
        self.profiler.start_frame()
        
        if not self.is_running:
            self.update_static_performance_displays()
            metrics = self.profiler.end_frame()
            return []
        
        # Update system time
        self.current_time += 1.0 / self.settings.target_fps
        self.frame_count += 1
        
        # Adaptive sweep rate based on performance
        current_quality = self.quality_manager.current_quality
        sweep_rate = 30.0 * current_quality  # Adjust sweep rate with quality
        self.sweep_angle = (self.sweep_angle + sweep_rate * 0.1) % 360
        
        # Update target positions
        detection_start = time.perf_counter()
        self.data_generator.update_targets(1.0 / self.settings.target_fps)
        
        # Process detections
        self.process_optimized_detection()
        detection_time = (time.perf_counter() - detection_start) * 1000
        
        # Update displays
        render_start = time.perf_counter()
        self.update_optimized_radar_display()
        self.update_performance_panels()
        render_time = (time.perf_counter() - render_start) * 1000
        
        # Memory management
        self.manage_memory()
        
        # End frame and get metrics
        metrics = self.profiler.end_frame()
        metrics.detection_time_ms = detection_time
        metrics.rendering_time_ms = render_time
        
        # Update adaptive quality
        new_quality = self.quality_manager.update_quality(
            metrics.frame_rate, metrics.frame_time_ms
        )
        metrics.quality_level = new_quality
        
        # Store performance history
        self.performance_history.append(metrics.to_dict())
        
        # Log significant performance changes
        if abs(new_quality - current_quality) > 0.1:
            self.optimization_log.append({
                'time': self.current_time,
                'fps': metrics.frame_rate,
                'quality': new_quality,
                'reason': 'adaptive_quality_adjustment'
            })
            print(f"⚡ Quality adjusted to {new_quality:.2f} (FPS: {metrics.frame_rate:.1f})")
        
        return []
    
    