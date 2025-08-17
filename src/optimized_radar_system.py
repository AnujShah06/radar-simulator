"""
High-Performance Radar System - Day 7 Task 3
============================================
Optimized radar system with 60+ FPS performance and adaptive quality control.
Features professional-grade performance optimization techniques.

Performance Features:
• 60+ FPS target with adaptive quality
• Memory management and efficient rendering
• Real-time performance monitoring
• Adaptive detail reduction under load
• Multi-threaded processing pipeline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Wedge, FancyBboxPatch
from matplotlib.collections import LineCollection, PatchCollection
import time
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
import gc
import psutil
import weakref

# Import radar components
try:
    from src.radar_data_generator import RadarDataGenerator
    from src.signal_processing import SignalProcessor
    from src.target_detection import TargetDetector
    from src.multi_target_tracker import MultiTargetTracker
    print("✅ All radar components imported successfully")
except ImportError as e:
    print(f"⚠️  Warning: Some components not found: {e}")

class QualityLevel(Enum):
    """Adaptive quality levels for performance optimization"""
    ULTRA = "Ultra"      # 60+ FPS, full detail
    HIGH = "High"        # 45+ FPS, high detail
    MEDIUM = "Medium"    # 30+ FPS, medium detail
    LOW = "Low"          # 15+ FPS, basic detail
    MINIMAL = "Minimal"  # Any FPS, minimal detail

@dataclass
class PerformanceMetrics:
    """Real-time performance monitoring"""
    frame_times: deque = field(default_factory=lambda: deque(maxlen=120))  # 2 seconds at 60fps
    processing_times: deque = field(default_factory=lambda: deque(maxlen=60))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=30))
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Current metrics
    current_fps: float = 0.0
    target_fps: float = 60.0
    avg_frame_time: float = 0.0
    avg_processing_time: float = 0.0
    current_memory_mb: float = 0.0
    current_cpu_percent: float = 0.0
    
    # Performance targets
    quality_level: QualityLevel = QualityLevel.ULTRA
    frames_dropped: int = 0
    quality_adjustments: int = 0
    
    def update_frame_time(self, frame_time: float):
        """Update frame timing metrics"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 10:
            self.avg_frame_time = np.mean(list(self.frame_times)[-10:])
            self.current_fps = 1.0 / max(self.avg_frame_time, 0.001)
    
    def update_processing_time(self, proc_time: float):
        """Update processing time metrics"""
        self.processing_times.append(proc_time)
        if len(self.processing_times) > 10:
            self.avg_processing_time = np.mean(list(self.processing_times)[-10:])
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            process = psutil.Process()
            self.current_memory_mb = process.memory_info().rss / 1024 / 1024
            self.current_cpu_percent = process.cpu_percent()
            
            self.memory_usage.append(self.current_memory_mb)
            self.cpu_usage.append(self.current_cpu_percent)
        except:
            # Fallback if psutil not available
            self.current_memory_mb = 0.0
            self.current_cpu_percent = 0.0

@dataclass
class QualitySettings:
    """Quality settings for adaptive performance"""
    max_trail_length: int = 100
    track_detail_level: int = 3  # 0=minimal, 3=full
    sweep_trail_length: int = 40
    update_frequency: float = 1.0  # Multiplier for update rate
    range_ring_detail: int = 8
    bearing_line_detail: int = 12
    target_symbol_size: float = 1.0
    text_detail_level: int = 2  # 0=none, 2=full
    velocity_vectors: bool = True
    glow_effects: bool = True

class AdaptiveQualityManager:
    """Manages adaptive quality based on performance"""
    
    def __init__(self, target_fps: float = 60.0):
        self.target_fps = target_fps
        self.quality_settings = {
            QualityLevel.ULTRA: QualitySettings(
                max_trail_length=100, track_detail_level=3, sweep_trail_length=40,
                update_frequency=1.0, range_ring_detail=8, bearing_line_detail=12,
                target_symbol_size=1.0, text_detail_level=2, velocity_vectors=True, glow_effects=True
            ),
            QualityLevel.HIGH: QualitySettings(
                max_trail_length=80, track_detail_level=3, sweep_trail_length=30,
                update_frequency=0.8, range_ring_detail=6, bearing_line_detail=8,
                target_symbol_size=0.9, text_detail_level=2, velocity_vectors=True, glow_effects=True
            ),
            QualityLevel.MEDIUM: QualitySettings(
                max_trail_length=60, track_detail_level=2, sweep_trail_length=20,
                update_frequency=0.6, range_ring_detail=4, bearing_line_detail=6,
                target_symbol_size=0.8, text_detail_level=1, velocity_vectors=True, glow_effects=False
            ),
            QualityLevel.LOW: QualitySettings(
                max_trail_length=40, track_detail_level=1, sweep_trail_length=15,
                update_frequency=0.4, range_ring_detail=4, bearing_line_detail=4,
                target_symbol_size=0.7, text_detail_level=1, velocity_vectors=False, glow_effects=False
            ),
            QualityLevel.MINIMAL: QualitySettings(
                max_trail_length=20, track_detail_level=0, sweep_trail_length=10,
                update_frequency=0.2, range_ring_detail=2, bearing_line_detail=2,
                target_symbol_size=0.6, text_detail_level=0, velocity_vectors=False, glow_effects=False
            )
        }
        
        self.current_quality = QualityLevel.ULTRA
        self.quality_change_cooldown = 0
        self.performance_history = deque(maxlen=30)  # 30 frames of history
    
    def update_quality(self, current_fps: float, target_fps: float) -> QualityLevel:
        """Update quality level based on performance"""
        self.performance_history.append(current_fps)
        
        # Only adjust quality if cooldown has expired
        if self.quality_change_cooldown > 0:
            self.quality_change_cooldown -= 1
            return self.current_quality
        
        # Calculate average FPS over recent history
        if len(self.performance_history) >= 10:
            avg_fps = np.mean(list(self.performance_history)[-10:])
        else:
            avg_fps = current_fps
        
        # Determine target quality level
        if avg_fps >= target_fps * 0.95:  # Within 5% of target
            target_quality = QualityLevel.ULTRA
        elif avg_fps >= target_fps * 0.75:  # Within 25% of target
            target_quality = QualityLevel.HIGH
        elif avg_fps >= target_fps * 0.5:   # Within 50% of target
            target_quality = QualityLevel.MEDIUM
        elif avg_fps >= target_fps * 0.25:  # Within 75% of target
            target_quality = QualityLevel.LOW
        else:
            target_quality = QualityLevel.MINIMAL
        
        # Apply quality change if needed
        if target_quality != self.current_quality:
            old_quality = self.current_quality
            self.current_quality = target_quality
            self.quality_change_cooldown = 60  # Wait 60 frames before next change
            
            print(f"⚡ Quality adjusted: {old_quality.value} → {target_quality.value} (FPS: {avg_fps:.1f})")
            
        return self.current_quality
    
    def get_current_settings(self) -> QualitySettings:
        """Get current quality settings"""
        return self.quality_settings[self.current_quality]

class OptimizedRenderManager:
    """Optimized rendering with selective updates"""
    
    def __init__(self):
        self.cached_elements = {}
        self.dirty_flags = set()
        self.static_collections = {}
        self.dynamic_collections = {}
        
        # Rendering optimization flags
        self.enable_caching = True
        self.enable_culling = True
        self.enable_lod = True  # Level of detail
        
    def mark_dirty(self, element_type: str):
        """Mark element type as needing redraw"""
        self.dirty_flags.add(element_type)
    
    def is_dirty(self, element_type: str) -> bool:
        """Check if element needs redraw"""
        return element_type in self.dirty_flags
    
    def clear_dirty(self, element_type: str):
        """Clear dirty flag for element"""
        self.dirty_flags.discard(element_type)
    
    def create_static_collection(self, name: str, patches: List, colors: List):
        """Create optimized static patch collection"""
        if self.enable_caching:
            collection = PatchCollection(patches, facecolors=colors, alpha=0.3)
            self.static_collections[name] = collection
            return collection
        return None
    
    def create_dynamic_collection(self, name: str, lines: List, colors: List):
        """Create optimized dynamic line collection"""
        if self.enable_caching:
            collection = LineCollection(lines, colors=colors, alpha=0.8)
            self.dynamic_collections[name] = collection
            return collection
        return None

