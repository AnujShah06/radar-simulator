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

