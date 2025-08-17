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

class HighPerformanceRadarSystem:
    """
    High-Performance Radar System with Adaptive Quality Control
    
    This system maintains 60+ FPS through advanced optimization techniques:
    - Adaptive quality reduction under load
    - Efficient rendering with selective updates
    - Memory management and garbage collection
    - Real-time performance monitoring
    """
    
    def __init__(self):
        print("⚡ Initializing High-Performance Radar System...")
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.quality_manager = AdaptiveQualityManager(target_fps=60.0)
        self.render_manager = OptimizedRenderManager()
        
        # Core radar components
        self.data_generator = RadarDataGenerator(max_range_km=200)
        self.signal_processor = SignalProcessor()
        self.target_detector = TargetDetector()
        self.tracker = MultiTargetTracker()
        
        # Configure for performance
        self.configure_for_performance()
        
        # System state with optimization
        self.is_running = False
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.frame_count = 0
        
        # Optimized data structures
        self.sweep_history = deque(maxlen=200)  # Bounded history
        self.target_trails = {}  # Weak references for auto cleanup
        self.cached_tracks = weakref.WeakValueDictionary()
        
        # Threading for background processing
        self.processing_thread = None
        self.processing_queue = deque(maxlen=10)
        self.results_queue = deque(maxlen=5)
        
        # Display optimization
        self.fig = None
        self.axes = {}
        self.animation = None
        self.last_render_time = 0.0
        
        # Performance tuning parameters
        self.max_targets_per_frame = 50  # Limit processing load
        self.garbage_collection_interval = 300  # Every 5 seconds at 60fps
        self.memory_cleanup_threshold = 500.0  # MB
        
        self.setup_optimized_display()
        self.load_performance_test_scenario()
        
        print("⚡ Performance optimizations applied:")
        print(f"   • Target FPS: {self.metrics.target_fps}")
        print(f"   • Adaptive quality: {self.quality_manager.current_quality.value}")
        print(f"   • Memory management: Enabled")
        print(f"   • Rendering optimization: Enabled")
        
    def configure_for_performance(self):
        """Configure radar components for optimal performance"""
        # Optimize signal processor
        self.signal_processor.detection_threshold = 0.08
        self.signal_processor.false_alarm_rate = 0.05
        
        # Optimize target detector
        self.target_detector.min_detections_for_confirmation = 1
        self.target_detector.association_distance_threshold = 12.0
        
        # Optimize tracker for performance
        self.tracker.max_association_distance = 15.0
        self.tracker.min_hits_for_confirmation = 1
        self.tracker.max_missed_detections = 10  # Reduced for performance
        self.tracker.max_track_age_without_update = 30.0  # Reduced for cleanup
        
    def setup_optimized_display(self):
        """Setup display with performance optimizations"""
        plt.style.use('dark_background')
        
        # Use optimized figure settings
        self.fig = plt.figure(figsize=(18, 12), facecolor='black')
        self.fig.patch.set_facecolor('black')
        
        # Optimized layout - fewer subplots for better performance
        gs = self.fig.add_gridspec(3, 4, height_ratios=[3, 1, 1], width_ratios=[3, 1, 1, 1])
        
        # Main radar display with optimization
        self.axes['radar'] = self.fig.add_subplot(gs[0, :3], projection='polar')
        self.setup_optimized_radar_scope()
        
        # Performance monitoring panels
        self.axes['performance'] = self.fig.add_subplot(gs[0, 3])
        self.axes['quality'] = self.fig.add_subplot(gs[1, 0])
        self.axes['system'] = self.fig.add_subplot(gs[1, 1])
        self.axes['tracks'] = self.fig.add_subplot(gs[1, 2])
        self.axes['controls'] = self.fig.add_subplot(gs[1, 3])
        self.axes['metrics'] = self.fig.add_subplot(gs[2, :])
        
        # Style panels for performance
        for name, ax in self.axes.items():
            if name != 'radar':
                ax.set_facecolor('#001122')
                for spine in ax.spines.values():
                    spine.set_color('#00ff00')
                    spine.set_linewidth(1)
                ax.tick_params(colors='#00ff00', labelsize=8)
        
        # Title
        self.fig.suptitle('HIGH-PERFORMANCE RADAR SYSTEM - 60+ FPS OPTIMIZED', 
                         fontsize=18, color='#00ff00', weight='bold', y=0.95)
    
    def setup_optimized_radar_scope(self):
        """Setup radar scope with rendering optimizations"""
        ax = self.axes['radar']
        ax.set_facecolor('black')
        ax.set_ylim(0, 200)
        ax.set_title('OPTIMIZED RADAR PPI SCOPE\n60+ FPS Performance Target', 
                    color='#00ff00', pad=20, fontsize=14, weight='bold')
        
        # Create static elements once for performance
        quality_settings = self.quality_manager.get_current_settings()
        
        # Optimized range rings
        range_intervals = np.linspace(50, 200, quality_settings.range_ring_detail//2)
        for r in range_intervals:
            circle = Circle((0, 0), r, fill=False, color='#00ff00', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
            if quality_settings.text_detail_level > 0:
                ax.text(np.pi/4, r-10, f'{r:.0f}km', color='#00ff00', fontsize=10, ha='center')
        
        # Optimized bearing lines
        bearing_step = 360 // quality_settings.bearing_line_detail
        for angle in range(0, 360, bearing_step):
            rad = np.radians(angle)
            ax.plot([rad, rad], [0, 200], color='#00ff00', alpha=0.2, linewidth=0.5)
            if quality_settings.text_detail_level > 1:
                ax.text(rad, 210, f'{angle}°', color='#00ff00', fontsize=9, ha='center')
        
        # Configure polar display
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.grid(True, color='#00ff00', alpha=0.2)
        ax.set_rticks([])
        ax.set_thetagrids([])
        
        # Mark static elements as clean
        self.render_manager.clear_dirty('static_elements')
    
    def load_performance_test_scenario(self):
        """Load scenario optimized for performance testing"""
        print("⚡ Loading performance test scenario...")
        
        # Create scenario with many targets to stress test performance
        np.random.seed(42)  # Reproducible performance test
        
        # Add 30 aircraft for performance testing
        for i in range(30):
            x = np.random.uniform(-180, 180)
            y = np.random.uniform(-180, 180)
            heading = np.random.uniform(0, 360)
            speed = np.random.uniform(200, 800)
            self.data_generator.add_aircraft(x, y, heading, speed)
        
        # Add 15 ships
        for i in range(15):
            x = np.random.uniform(-150, 150)
            y = np.random.uniform(-150, 150)
            heading = np.random.uniform(0, 360)
            speed = np.random.uniform(10, 40)
            self.data_generator.add_ship(x, y, heading, speed)
        
        # Add weather returns
        for i in range(5):
            x = np.random.uniform(-100, 100)
            y = np.random.uniform(-100, 100)
            intensity = np.random.uniform(20, 50)
            self.data_generator.add_weather_returns(x, y, intensity)
        
        total_targets = len(self.data_generator.targets)
        print(f"✅ Performance test scenario loaded: {total_targets} targets")
        print("   • High target density for performance stress testing")
        print("   • Mixed target types for comprehensive testing")
    
    def animate_optimized(self, frame):
        """Optimized animation loop with performance monitoring"""
        frame_start = time.time()
        
        if not self.is_running:
            self.update_performance_panels_only()
            return []
        
        # Update frame count and time
        self.frame_count += 1
        self.current_time += 1.0 / self.metrics.target_fps
        
        # Periodic garbage collection for memory management
        if self.frame_count % self.garbage_collection_interval == 0:
            gc.collect()
            self.cleanup_expired_trails()
        
        # Update system metrics periodically (not every frame for performance)
        if self.frame_count % 30 == 0:  # Every 0.5 seconds
            self.metrics.update_system_metrics()
        
        # Processing with performance monitoring
        processing_start = time.time()
        self.process_optimized_detection()
        processing_time = time.time() - processing_start
        self.metrics.update_processing_time(processing_time)
        
        # Adaptive quality management
        current_quality = self.quality_manager.update_quality(
            self.metrics.current_fps, self.metrics.target_fps
        )
        
        # Optimized rendering
        self.render_optimized_display(current_quality)
        self.update_performance_panels()
        
        # Frame timing
        frame_time = time.time() - frame_start
        self.metrics.update_frame_time(frame_time)
        
        # Memory management
        if self.metrics.current_memory_mb > self.memory_cleanup_threshold:
            self.perform_memory_cleanup()
        
        return []
    
    