# Draw sweep beam
        if mode_config['sweep_rate'] > 0:
            beam_width = np.radians(mode_config['beam_width'] * quality_settings['detail_level'])
            beam = Wedge((0, 0), max_range,
                        np.degrees(sweep_rad - beam_width/2),
                        np.degrees(sweep_rad + beam_width/2),
                        alpha=0.3, color=sweep_color)
            ax.add_patch(beam)
            
            # Bright sweep line
            ax.plot([sweep_rad, sweep_rad], [0, max_range], 
                   color=sweep_color, linewidth=3, alpha=0.9)
        
        # Optimized sweep trail
        trail_points = quality_settings['trail_points']
        if len(self.sweep_history) > 0:
            display_trail = min(len(self.sweep_history), trail_points)
            for i, (angle, timestamp) in enumerate(list(self.sweep_history)[-display_trail:]):
                age_factor = (i + 1) / display_trail
                alpha = 0.05 + 0.15 * age_factor * quality_settings['detail_level']
                trail_rad = np.radians(angle)
                ax.plot([trail_rad, trail_rad], [0, max_range], 
                       color=sweep_color, linewidth=1, alpha=alpha)
        
        self.sweep_history.append((self.sweep_angle, self.current_time))
        
        # Display tracks with LOD
        confirmed_tracks = self.tracker.get_confirmed_tracks()
        for track in confirmed_tracks:
            self.draw_ultimate_track(ax, track, quality_settings, sweep_color)
        
        # Status overlay
        fps = self.profiler.get_average_fps(10)
        quality = self.quality_manager.current_quality
        
        status_text = (f'MODE: {self.current_config.current_mode.value} | '
                      f'FPS: {fps:.1f} | Q: {quality:.2f} | '
                      f'PWR: {self.current_config.transmitter_power_kw:.0f}kW | '
                      f'TRACKS: {len(confirmed_tracks)}')
        
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, 
               color='#ffff00', fontsize=12, weight='bold', verticalalignment='top')
        
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.grid(True, color='#00ff00', alpha=0.2)
        ax.set_rticks([])
        ax.set_thetagrids([])
    
    def draw_ultimate_track(self, ax, track, quality_settings, mode_color):
        """Draw track with ultimate feature set"""
        range_km = np.sqrt(track.state.x**2 + track.state.y**2)
        bearing_rad = np.arctan2(track.state.x, track.state.y)
        
        if range_km > self.current_config.max_range_km:
            return
            
        detail_level = quality_settings['detail_level']
        
        # Classification-based styling
        if track.classification == 'aircraft':
            marker, color, base_size = '^', '#ffff00', 150
        elif track.classification == 'ship':
            marker, color, base_size = 's', '#00ffff', 130
        else:
            marker, color, base_size = 'o', '#ff8800', 110
        
        # LOD rendering
        if detail_level > 0.8:
            # High detail - full track with info and effects
            size = int(base_size * detail_level)
            ax.scatter(bearing_rad, range_km, s=size, c=color, marker=marker, 
                      alpha=0.9, edgecolors='white', linewidths=2, zorder=20)
            
            # Detailed info
            info_text = f'T{track.id[-3:]}\n{track.classification[:4].upper()}\n{track.state.speed_kmh:.0f}kt\nQ:{track.quality_score:.1f}'
            ax.text(bearing_rad, range_km + self.current_config.max_range_km*0.05, 
                   info_text, color=color, fontsize=9, ha='center', va='bottom', weight='bold')
            
            # Velocity vector
            speed = np.sqrt(track.state.vx**2 + track.state.vy**2)
            if speed > 0.5:
                vel_scale = self.current_config.max_range_km * 0.1
                end_x = track.state.x + track.state.vx * vel_scale
                end_y = track.state.y + track.state.vy * vel_scale
                end_range = np.sqrt(end_x**2 + end_y**2)
                end_bearing = np.arctan2(end_x, end_y)
                
                if end_range <= self.current_config.max_range_km:
                    ax.annotate('', xy=(end_bearing, end_range),
                               xytext=(bearing_rad, range_km),
                               arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8))
        
        elif detail_level > 0.5:
            # Medium detail
            ax.scatter(bearing_rad, range_km, s=80, c=color, marker='o', alpha=0.8, zorder=15)
            ax.text(bearing_rad, range_km + 8, f'T{track.id[-3:]}', 
                   color=color, fontsize=8, ha='center', va='bottom')
        
        else:
            # Low detail - just dots
            ax.scatter(bearing_rad, range_km, s=20, c='#888888', marker='.', alpha=0.6, zorder=10)
        
        # Trail rendering
        if detail_level > 0.4:
            self.target_trails[track.id].append((bearing_rad, range_km, self.current_time))
            
            trail = list(self.target_trails[track.id])
            if len(trail) > 1:
                trail_length = int(len(trail) * detail_level)
                for i in range(len(trail) - trail_length, len(trail) - 1):
                    if i >= 0:
                        b1, r1, t1 = trail[i]
                        b2, r2, t2 = trail[i + 1]
                        age = self.current_time - t1
                        alpha = max(0.1, 1.0 - age / 20.0) * detail_level
                        ax.plot([b1, b2], [r1, r2], color=color, alpha=alpha, linewidth=1)
    
    def manage_memory(self):
        """Efficient memory management"""
        self.gc_counter += 1
        
        if self.gc_counter >= 100:
            gc.collect()
            self.gc_counter = 0
        
        # Clean old trail data
        for track_id in list(self.target_trails.keys()):
            trail = self.target_trails[track_id]
            while trail and self.current_time - trail[0][2] > 30.0:
                trail.popleft()
            if not trail:
                del self.target_trails[track_id]
    
    def update_all_panels(self):
        """Update all display panels"""
        self.update_modes_panel()
        self.update_presets_panel()
        self.update_performance_panel()
        self.update_status_panel()
        self.update_tracks_panel()
        self.update_quality_panel()
        self.update_fps_graph()
        self.update_resources_panel()
        self.update_timing_panel()
        self.update_optimization_panel()
        self.update_threading_panel()
        self.update_controls_panel()
        self.update_alerts_panel()
        self.update_filters_panel()
        self.update_config_info_panel()
        self.update_system_info_panel()
    
    def update_static_displays(self):
        """Update displays when system is stopped"""
        self.update_modes_panel()
        self.update_presets_panel()
        self.update_controls_panel()
        self.update_config_info_panel()
        self.update_system_info_panel()
    
    def update_modes_panel(self):
        """Update radar modes panel"""
        ax = self.axes['modes']
        ax.clear()
        ax.set_title('RADAR MODES', color='#00ff00', fontsize=11, weight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        modes = [
            (RadarMode.SEARCH, (0.5, 8, 9, 1.2)),
            (RadarMode.TRACK, (0.5, 6.5, 9, 1.2)),
            (RadarMode.TWS, (0.5, 5, 9, 1.2)),
            (RadarMode.WEATHER, (0.5, 3.5, 9, 1.2)),
            (RadarMode.STANDBY, (0.5, 2, 9, 1.2))
        ]
        
        for mode, (x, y, w, h) in modes:
            if mode == self.current_config.current_mode:
                color = '#006600'
                text_color = '#00ff00'
            else:
                color = '#333333'
                text_color = '#888888'
                
            rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='#00ff00', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, mode.value, ha='center', va='center',
                   color=text_color, fontsize=9, weight='bold')
        ax.axis('off')
    
    def update_presets_panel(self):
        """Update configuration presets panel"""
        ax = self.axes['presets']
        ax.clear()
        ax.set_title('CONFIG PRESETS', color='#00ff00', fontsize=11, weight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        presets = [
            (ConfigPreset.AIRPORT_CONTROL, (0.5, 8.5, 9, 1)),
            (ConfigPreset.NAVAL_SURVEILLANCE, (0.5, 7.2, 9, 1)),
            (ConfigPreset.MILITARY_DEFENSE, (0.5, 5.9, 9, 1)),
            (ConfigPreset.WEATHER_MONITORING, (0.5, 4.6, 9, 1)),
            (ConfigPreset.COASTAL_PATROL, (0.5, 3.3, 9, 1)),
            (ConfigPreset.CUSTOM, (0.5, 2, 9, 1))
        ]
        
        for preset, (x, y, w, h) in presets:
            rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                facecolor='#004444', edgecolor='#00ff00', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, preset.value, ha='center', va='center',
                   color='#00ff00', fontsize=8, weight='bold')
        ax.axis('off')
    
    def update_performance_panel(self):
        """Update main performance panel"""
        ax = self.axes['performance']
        ax.clear()
        ax.set_title('PERFORMANCE', color='#00ff00', fontsize=11, weight='bold')
        
        fps = self.profiler.get_average_fps(10)
        quality = self.quality_manager.current_quality
        
        perf_text = f"""
TARGET: {self.target_fps:.0f} FPS
CURRENT: {fps:.1f} FPS
EFFICIENCY: {(fps/self.target_fps*100):.1f}%

QUALITY: {quality:.2f}
LEVEL: {self.get_quality_level_name(quality)}

FRAMES: {self.profiler.total_frames}
OPTIMIZATION: {'ACTIVE' if quality < 0.95 else 'STABLE'}
        """.strip()
        
        color = '#00ff00' if fps >= self.target_fps * 0.9 else '#ffff00'
        if fps < self.target_fps * 0.7:
            color = '#ff4400'
        
        ax.text(0.05, 0.95, perf_text, transform=ax.transAxes,
               color=color, fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def update_status_panel(self):
        """Update system status panel"""
        ax = self.axes['status']
        ax.clear()
        ax.set_title('SYSTEM STATUS', color='#00ff00', fontsize=11, weight='bold')
        
        status_text = f"""
STATUS: {'ACTIVE' if self.is_running else 'STANDBY'}
MODE: {self.current_config.current_mode.value}
RANGE: {self.current_config.max_range_km:.0f}km
POWER: {self.current_config.transmitter_power_kw:.0f}kW
SWEEP: {self.current_config.sweep_rate_rpm:.0f}RPM

UPTIME: {self.current_time:.0f}s
HEALTH: OPTIMAL
        """.strip()
        
        ax.text(0.05, 0.95, status_text, transform=ax.transAxes,
               color='#00ff00', fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def update_tracks_panel(self):
        """Update tracks panel"""
        ax = self.axes['tracks']
        ax.clear()
        ax.set_title('ACTIVE TRACKS', color='#00ff00', fontsize=11, weight='bold')
        
        tracks = self.tracker.get_confirmed_tracks()
        
        if tracks:
            tracks_text = f"CONFIRMED: {len(tracks)}\n\n"
            for i, track in enumerate(tracks[:5]):
                range_km = np.sqrt(track.state.x**2 + track.state.y**2)
                tracks_text += f"T{track.id[-3:]}: {track.classification[:4].upper()}\n"
                tracks_text += f"  {range_km:.1f}km, {track.state.speed_kmh:.0f}kt\n"
                if i < 4:
                    tracks_text += "\n"
        else:
            tracks_text = "NO CONFIRMED\nTRACKS"
            
        ax.text(0.05, 0.95, tracks_text, transform=ax.transAxes,
               color='#00ff00', fontsize=8, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def update_quality_panel(self):
        """Update adaptive quality panel"""
        ax = self.axes['quality']
        ax.clear()
        ax.set_title('ADAPTIVE QUALITY', color='#00ff00', fontsize=11, weight='bold')
        
        quality = self.quality_manager.current_quality
        level_name = self.get_quality_level_name(quality)
        
        quality_text = f"""
CURRENT: {quality:.2f}
LEVEL: {level_name}

SETTINGS:
Trail: {int(50 * quality)}
History: {int(30 * quality)}
Effects: {'ON' if quality > 0.5 else 'OFF'}

STATUS: {'ADJUSTING' if len(self.quality_manager.quality_history) >= 3 else 'STABLE'}
        """.strip()
        
        color = '#00ff00' if quality >= 0.75 else '#ffff00' if quality >= 0.5 else '#ff4400'
        
        ax.text(0.05, 0.95, quality_text, transform=ax.transAxes,
               color=color, fontsize=8, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def update_fps_graph(self):
        """Update FPS graph"""
        ax = self.axes['fps_graph']
        ax.clear()
        ax.set_title('REAL-TIME FPS MONITORING', color='#00ff00', fontsize=11, weight='bold')
        
        if len(self.performance_history) > 10:
            fps_data = [entry['frame_rate'] for entry in list(self.performance_history)[-60:]]
            time_data = list(range(len(fps_data)))
            
            ax.plot(time_data, fps_data, color='#00ff00', linewidth=2)
            ax.axhline(y=self.target_fps, color='#ffff00', linestyle='--', alpha=0.7)
            ax.axhline(y=self.target_fps * 0.8, color='#ff4400', linestyle='--', alpha=0.5)
            
            ax.set_ylim(0, max(80, max(fps_data) * 1.1))
            ax.set_ylabel('FPS', color='#00ff00', fontsize=9)
            ax.tick_params(colors='#00ff00', labelsize=8)
            ax.grid(True, alpha=0.3, color='#00ff00')
            ax.fill_between(time_data, fps_data, alpha=0.3, color='#00ff00')
    
    def update_resources_panel(self):
        """Update system resources panel"""
        ax = self.axes['resources']
        ax.clear()
        ax.set_title('RESOURCES', color='#00ff00', fontsize=11, weight='bold')
        
        metrics = self.profiler.current_metrics
        
        resource_text = f"""
CPU: {metrics.cpu_usage:.1f}%
MEMORY: {metrics.memory_usage_mb:.1f}MB

DETECTION: {metrics.detection_time_ms:.1f}ms
RENDERING: {metrics.rendering_time_ms:.1f}ms

TOTAL: {metrics.frame_time_ms:.1f}ms
        """.strip()
        
        ax.text(0.05, 0.95, resource_text, transform=ax.transAxes,
               color='#00ff00', fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def update_timing_panel(self):
        """Update timing breakdown panel"""
        ax = self.axes['timing']
        ax.clear()
        ax.set_title('TIMING', color='#00ff00', fontsize=11, weight='bold')
        
        if len(self.performance_history) > 0:
            recent = list(self.performance_history)[-10:]
            avg_frame = np.mean([e.get('frame_time_ms', 0) for e in recent])
            
            timing_text = f"""
FRAME: {avg_frame:.1f}ms
TARGET: {1000/self.target_fps:.1f}ms

OVERHEAD: {max(0, avg_frame - 10):.1f}ms
EFFICIENCY: {min(100, 1000/self.target_fps/avg_frame*100):.1f}%

BOTTLENECK: {'CPU' if avg_frame > 20 else 'NONE'}
            """.strip()
        else:
            timing_text = "No timing\ndata available"
        
        ax.text(0.05, 0.95, timing_text, transform=ax.transAxes,
               color='#00ff00', fontsize=8, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def update_optimization_panel(self):
        """Update optimization status panel"""
        ax = self.axes['optimization']
        ax.clear()
        ax.set_title('OPTIMIZATION', color='#00ff00', fontsize=11, weight='bold')
        
        opt_text = f"""
ADAPTIVE QUALITY: ON
LOD RENDERING: ON
MEMORY MGMT: ON
ASYNC PROC: {'ON' if self.async_processing else 'OFF'}

OPTIMIZATIONS: {len(self.optimization_log)}
GC CYCLES: {self.gc_counter}

STATUS: ACTIVE
        """.strip()
        
        ax.text(0.05, 0.95, opt_text, transform=ax.transAxes,
               color='#00ff00', fontsize=8, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def update_threading_panel(self):
        """Update threading status panel"""
        ax = self.axes['threading']
        ax.clear()
        ax.set_title('THREADING', color='#00ff00', fontsize=11, weight='bold')
        
        if self.async_processing:
            queue_size = self.processing_queue.qsize()
            
            thread_text = f"""
ASYNC: ENABLED
WORKER: {'ACTIVE' if self.worker_thread and self.worker_thread.is_alive() else 'INACTIVE'}

QUEUE: {queue_size}/10
UTILIZATION: {queue_size*10}%

STATUS: {'OPTIMAL' if queue_size < 8 else 'BUSY'}
            """.strip()
        else:
            thread_text = """
ASYNC: DISABLED
MODE: SYNCHRONOUS

All processing on
main thread.
            """.strip()
        
        ax.text(0.05, 0.95, thread_text, transform=ax.transAxes,
               color='#00ff00', fontsize=8, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def update_controls_panel(self):
        """Update system controls panel"""
        ax = self.axes['controls']
        ax.clear()
        ax.set_title('SYSTEM CONTROLS', color='#00ff00', fontsize=12, weight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        buttons = [
            ('START', (0.5, 7.5, 4, 1.5), '#006600' if not self.is_running else '#333333'),
            ('STOP', (5, 7.5, 4, 1.5), '#660000' if self.is_running else '#333333'),
            ('RESET', (0.5, 5.5, 4, 1.5), '#444444'),
            ('AUTO DEMO', (5, 5.5, 4, 1.5), '#004466'),
            ('SAVE CONFIG', (0.5, 3.5, 4, 1.5), '#440044'),
            ('OPTIMIZE', (5, 3.5, 4, 1.5), '#444400')
        ]
        
        for label, (x, y, w, h), color in buttons:
            rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='#00ff00', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                   color='#00ff00', fontsize=9, weight='bold')
        ax.axis('off')
    
    def update_alerts_panel(self):
        """Update alerts panel"""
        ax = self.axes['alerts']
        ax.clear()
        ax.set_title('ALERTS', color='#00ff00', fontsize=11, weight='bold')
        
        alerts = []
        
        # Performance alerts
        fps = self.profiler.get_average_fps(10)
        if fps < self.target_fps * 0.8:
            alerts.append("PERFORMANCE LOW")
        
        # Configuration alerts
        if self.current_config.transmitter_power_kw > 400:
            alerts.append("HIGH POWER")
        if self.current_config.detection_threshold < 0.05:
            alerts.append("HIGH SENSITIVITY")
        
        # Track alerts
        tracks = self.tracker.get_confirmed_tracks()
        if len(tracks) > 20:
            alerts.append("HIGH TRAFFIC")
        
        if not alerts:
            alerts.append("ALL SYSTEMS NORMAL")
        
        alert_text = "\n".join(alerts[:6])
        color = '#ffff00' if len(alerts) > 1 else '#00ff00'
        
        ax.text(0.05, 0.95, alert_text, transform=ax.transAxes,
               color=color, fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def update_filters_panel(self):
        """Update filters panel"""
        ax = self.axes['filters']
        ax.clear()
        ax.set_title('FILTERS', color='#00ff00', fontsize=11, weight='bold')
        
        filter_text = f"""
CLUTTER: {'ON' if self.current_config.clutter_rejection else 'OFF'}
WEATHER: {'ON' if self.current_config.weather_filtering else 'OFF'}
MTI: {'ON' if self.current_config.moving_target_indicator else 'OFF'}
SEA: {'ON' if self.current_config.sea_clutter_suppression else 'OFF'}
        """.strip()
        
        ax.text(0.05, 0.95, filter_text, transform=ax.transAxes,
               color='#00ff00', fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def update_config_info_panel(self):
        """Update configuration info panel"""
        ax = self.axes['config_info']
        ax.clear()
        ax.set_title('CONFIG INFO', color='#00ff00', fontsize=11, weight='bold')
        
        config_text = f"""
RANGE: {self.current_config.min_range_km:.0f}-{self.current_config.max_range_km:.0f}km
THRESHOLD: {self.current_config.detection_threshold:.3f}
POWER: {self.current_config.transmitter_power_kw:.0f}kW
SWEEP: {self.current_config.sweep_rate_rpm:.0f}RPM
BEAM: {self.current_config.beam_width_deg:.1f}°
        """.strip()
        
        ax.text(0.05, 0.95, config_text, transform=ax.transAxes,
               color='#00ff00', fontsize=8, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def update_system_info_panel(self):
        """Update system info panel"""
        ax = self.axes['system_info']
        ax.clear()
        ax.set_title('SYSTEM INFO', color='#00ff00', fontsize=11, weight='bold')
        
        targets = len(self.data_generator.targets)
        aircraft = sum(1 for t in self.data_generator.targets if t.target_type == 'aircraft')
        ships = sum(1 for t in self.data_generator.targets if t.target_type == 'ship')
        weather = sum(1 for t in self.data_generator.targets if t.target_type == 'weather')
        
        info_text = f"""
TARGETS: {targets}
AIRCRAFT: {aircraft}
SHIPS: {ships}
WEATHER: {weather}

VERSION: Day 7 Ultimate
BUILD: Complete Integration
        """.strip()
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
               color='#00ff00', fontsize=8, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def get_quality_level_name(self, quality: float) -> str:
        """Get quality level name"""
        if quality >= 0.9:
            return "MAXIMUM"
        elif quality >= 0.75:
            return "HIGH"
        elif quality >= 0.5:
            return "BALANCED"
        elif quality >= 0.25:
            return "PERFORMANCE"
        else:
            return "MINIMUM"
    
    def on_slider_change(self, slider_name: str, value: float):
        """Handle slider changes"""
        param_map = {
            'range': 'max_range_km',
            'threshold': 'detection_threshold',
            'power': 'transmitter_power_kw',
            'sweep_rate': 'sweep_rate_rpm'
        }
        
        if slider_name in param_map:
            param_name = param_map[slider_name]
            is_valid, error_msg = self.config_manager.validate_parameter(param_name, value)
            
            if is_valid:
                setattr(self.current_config, param_name, value)
                self.apply_configuration(self.current_config)
                
                if slider_name == 'range':
                    self.setup_radar_scope()
            else:
                print(f"Parameter validation failed: {error_msg}")
    
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes == self.axes['modes']:
            self.handle_mode_click(event)
        elif event.inaxes == self.axes['presets']:
            self.handle_preset_click(event)
        elif event.inaxes == self.axes['controls']:
            self.handle_control_click(event)
    
    def handle_mode_click(self, event):
        """Handle radar mode selection"""
        x, y = event.xdata, event.ydata
        if x is not None and y is not None and 0.5 <= x <= 9.5:
            if 8 <= y <= 9.2:
                self.switch_mode(RadarMode.SEARCH)
            elif 6.5 <= y <= 7.7:
                self.switch_mode(RadarMode.TRACK)
            elif 5 <= y <= 6.2:
                self.switch_mode(RadarMode.TWS)
            elif 3.5 <= y <= 4.7:
                self.switch_mode(RadarMode.WEATHER)
            elif 2 <= y <= 3.2:
                self.switch_mode(RadarMode.STANDBY)
    
    def handle_preset_click(self, event):
        """Handle preset selection"""
        x, y = event.xdata, event.ydata
        if x is not None and y is not None and 0.5 <= x <= 9.5:
            if 8.5 <= y <= 9.5:
                self.apply_preset(ConfigPreset.AIRPORT_CONTROL)
            elif 7.2 <= y <= 8.2:
                self.apply_preset(ConfigPreset.NAVAL_SURVEILLANCE)
            elif 5.9 <= y <= 6.9:
                self.apply_preset(ConfigPreset.MILITARY_DEFENSE)
            """
Ultimate Advanced Radar System - Day 7 Complete Integration
===========================================================
This is the culmination of Day 7 development, integrating:
• Advanced radar modes (Search, Track, TWS, Weather, Standby)
• Real-time configuration management with presets
• High-performance optimization with 60+ FPS targeting
• Professional operator interface and controls

Features:
• Multi-mode radar operation with mode-specific behaviors
• Interactive parameter adjustment with live validation
• Adaptive quality management for optimal performance
• Real-time performance monitoring and profiling
• Professional configuration presets for different missions
• Advanced memory management and threading
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Wedge, FancyBboxPatch
from matplotlib.widgets import Slider, Button, CheckButtons
import time
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
import gc
import psutil
from collections import deque, defaultdict

# Import radar components
try:
    from src.radar_data_generator import RadarDataGenerator
    from src.signal_processing import SignalProcessor
    from src.target_detection import TargetDetector
    from src.multi_target_tracker import MultiTargetTracker
    print("All radar components imported successfully")
except ImportError as e:
    print(f"Warning: Some components not found: {e}")

class RadarMode(Enum):
    """Advanced radar operating modes"""
    SEARCH = "SEARCH"
    TRACK = "TRACK" 
    TWS = "TWS"
    WEATHER = "WEATHER"
    STANDBY = "STANDBY"

class ConfigPreset(Enum):
    """Configuration presets for different missions"""
    AIRPORT_CONTROL = "Airport Control"
    NAVAL_SURVEILLANCE = "Naval Surveillance"
    MILITARY_DEFENSE = "Military Defense"
    WEATHER_MONITORING = "Weather Monitoring"
    COASTAL_PATROL = "Coastal Patrol"
    CUSTOM = "Custom"

class PerformanceLevel(Enum):
    """Adaptive quality levels"""
    MAXIMUM = "Maximum"
    HIGH = "High"
    BALANCED = "Balanced" 
    PERFORMANCE = "Performance"
    MINIMUM = "Minimum"

@dataclass
class RadarConfiguration:
    """Complete radar system configuration"""
    # Detection parameters
    max_range_km: float = 200.0
    min_range_km: float = 5.0
    detection_threshold: float = 0.08
    false_alarm_rate: float = 0.05
    
    # Sweep parameters
    sweep_rate_rpm: float = 30.0
    beam_width_deg: float = 2.0
    antenna_gain_db: float = 35.0
    
    # Tracking parameters
    max_association_distance: float = 10.0
    min_hits_for_confirmation: int = 1
    max_missed_detections: int = 15
    track_aging_time: float = 45.0
    
    # Filter settings
    clutter_rejection: bool = True
    weather_filtering: bool = True
    moving_target_indicator: bool = True
    sea_clutter_suppression: bool = False
    
    # Display settings
    trail_length_sec: float = 30.0
    update_rate_hz: float = 10.0
    brightness: float = 1.0
    contrast: float = 1.0
    
    # System settings
    transmitter_power_kw: float = 100.0
    current_mode: RadarMode = RadarMode.SEARCH

@dataclass 
class PerformanceMetrics:
    """Real-time performance metrics"""
    frame_rate: float = 0.0
    frame_time_ms: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    detection_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    rendering_time_ms: float = 0.0
    quality_level: float = 1.0
    confirmed_tracks: int = 0
    total_detections: int = 0

class ConfigurationManager:
    """Manages radar configurations and presets"""
    
    def __init__(self):
        self.presets = self._create_presets()
        self.current_config = RadarConfiguration()
        self.validation_rules = {
            'max_range_km': (10.0, 500.0),
            'min_range_km': (0.1, 50.0),
            'detection_threshold': (0.01, 1.0),
            'sweep_rate_rpm': (5.0, 120.0),
            'transmitter_power_kw': (10.0, 1000.0)
        }
    
    def _create_presets(self) -> Dict[ConfigPreset, RadarConfiguration]:
        """Create default configuration presets"""
        return {
            ConfigPreset.AIRPORT_CONTROL: RadarConfiguration(
                max_range_km=150.0, detection_threshold=0.06, sweep_rate_rpm=60.0,
                beam_width_deg=1.5, transmitter_power_kw=50.0, current_mode=RadarMode.TWS
            ),
            ConfigPreset.NAVAL_SURVEILLANCE: RadarConfiguration(
                max_range_km=300.0, detection_threshold=0.10, sweep_rate_rpm=20.0,
                beam_width_deg=3.0, sea_clutter_suppression=True, transmitter_power_kw=200.0
            ),
            ConfigPreset.MILITARY_DEFENSE: RadarConfiguration(
                max_range_km=400.0, detection_threshold=0.04, sweep_rate_rpm=90.0,
                beam_width_deg=1.0, transmitter_power_kw=500.0, current_mode=RadarMode.TRACK
            ),
            ConfigPreset.WEATHER_MONITORING: RadarConfiguration(
                max_range_km=250.0, detection_threshold=0.15, sweep_rate_rpm=15.0,
                beam_width_deg=4.0, weather_filtering=False, current_mode=RadarMode.WEATHER
            ),
            ConfigPreset.COASTAL_PATROL: RadarConfiguration(
                max_range_km=180.0, detection_threshold=0.08, sweep_rate_rpm=40.0,
                beam_width_deg=2.5, sea_clutter_suppression=True, current_mode=RadarMode.TWS
            )
        }
    
    def validate_parameter(self, param_name: str, value: float) -> Tuple[bool, str]:
        """Validate parameter values"""
        if param_name not in self.validation_rules:
            return True, ""
        min_val, max_val = self.validation_rules[param_name]
        if not (min_val <= value <= max_val):
            return False, f"{param_name} must be between {min_val} and {max_val}"
        return True, ""
    
    def apply_preset(self, preset: ConfigPreset) -> RadarConfiguration:
        """Apply configuration preset"""
        if preset in self.presets:
            config_dict = self.presets[preset].__dict__.copy()
            self.current_config = RadarConfiguration(**config_dict)
            return self.current_config
        return self.current_config

class PerformanceProfiler:
    """Advanced performance monitoring"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.frame_times = deque(maxlen=history_size)
        self.frame_start_time = 0.0
        self.current_metrics = PerformanceMetrics()
        self.total_frames = 0
        self.process = psutil.Process()
    
    def start_frame(self):
        """Mark frame start"""
        self.frame_start_time = time.perf_counter()
        
    def end_frame(self) -> PerformanceMetrics:
        """Calculate frame metrics"""
        frame_time = time.perf_counter() - self.frame_start_time
        
        self.current_metrics.frame_time_ms = frame_time * 1000
        self.current_metrics.frame_rate = 1.0 / max(frame_time, 0.001)
        self.current_metrics.cpu_usage = self.process.cpu_percent()
        self.current_metrics.memory_usage_mb = self.process.memory_info().rss / 1024 / 1024
        
        self.frame_times.append(frame_time)
        self.total_frames += 1
        
        return self.current_metrics
    
    def get_average_fps(self, window_size: int = 30) -> float:
        """Get average FPS"""
        if len(self.frame_times) < window_size:
            return 0.0
        recent_times = list(self.frame_times)[-window_size:]
        avg_time = sum(recent_times) / len(recent_times)
        return 1.0 / max(avg_time, 0.001)

class AdaptiveQualityManager:
    """Manages adaptive quality based on performance"""
    
    def __init__(self, target_fps: float = 60.0):
        self.target_fps = target_fps
        self.current_quality = 1.0
        self.adjustment_rate = 0.1
        self.quality_history = deque(maxlen=5)
        
    def update_quality(self, current_fps: float) -> float:
        """Update quality based on performance"""
        fps_ratio = current_fps / self.target_fps
        
        if fps_ratio < 0.8:
            target_quality = max(0.25, self.current_quality - self.adjustment_rate)
        elif fps_ratio > 1.1:
            target_quality = min(1.0, self.current_quality + self.adjustment_rate * 0.5)
        else:
            target_quality = self.current_quality
        
        self.quality_history.append(target_quality)
        if len(self.quality_history) >= 3:
            avg_target = np.mean(list(self.quality_history))
            if abs(avg_target - self.current_quality) > 0.05:
                self.current_quality = avg_target
        
        return self.current_quality
    
    def get_quality_settings(self) -> Dict[str, Any]:
        """Get settings based on current quality"""
        quality = self.current_quality
        return {
            'trail_points': int(50 * quality),
            'sweep_history': int(30 * quality),
            'detail_level': quality,
            'effects_enabled': quality > 0.5,
            'anti_aliasing': quality > 0.8
        }

class UltimateRadarSystem:
    """
    Ultimate Advanced Radar System - Day 7 Complete Integration
    
    This system represents the culmination of advanced radar development,
    combining multiple operating modes, real-time configuration, and
    high-performance optimization into a professional radar platform.
    """
    
    def __init__(self):
        print("Initializing Ultimate Advanced Radar System...")
        
        # Performance management
        self.target_fps = 60.0
        self.profiler = PerformanceProfiler()
        self.quality_manager = AdaptiveQualityManager(self.target_fps)
        
        # Configuration management
        self.config_manager = ConfigurationManager()
        self.current_config = self.config_manager.current_config
        
        # Core radar components
        self.data_generator = RadarDataGenerator(max_range_km=200)
        self.signal_processor = SignalProcessor()
        self.target_detector = TargetDetector()
        self.tracker = MultiTargetTracker()
        
        # Apply initial optimizations
        self.apply_configuration(self.current_config)
        
        # System state
        self.is_running = False
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.frame_count = 0
        
        # Optimized data structures
        self.sweep_history = deque(maxlen=50)
        self.target_trails = defaultdict(lambda: deque(maxlen=30))
        self.performance_history = deque(maxlen=300)
        
        # Memory management
        self.gc_counter = 0
        self.optimization_log = []
        
        # Threading for async processing
        self.processing_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.async_processing = True
        
        if self.async_processing:
            self.start_worker_thread()
        
        # Display components
        self.fig = None
        self.axes = {}
        self.sliders = {}
        self.animation = None
        
        self.setup_ultimate_display()
        self.load_comprehensive_scenario()
        
    def apply_configuration(self, config: RadarConfiguration):
        """Apply configuration to all radar components"""
        # Update signal processor
        self.signal_processor.detection_threshold = config.detection_threshold
        self.signal_processor.false_alarm_rate = config.false_alarm_rate
        
        # Update target detector
        self.target_detector.min_detections_for_confirmation = config.min_hits_for_confirmation
        self.target_detector.association_distance_threshold = config.max_association_distance
        
        # Update tracker
        self.tracker.max_association_distance = config.max_association_distance
        self.tracker.min_hits_for_confirmation = config.min_hits_for_confirmation
        self.tracker.max_missed_detections = config.max_missed_detections
        self.tracker.max_track_age_without_update = config.track_aging_time
        
        # Update data generator
        self.data_generator.max_range_km = config.max_range_km
        
        print(f"Configuration applied: {config.current_mode.value} mode, {config.max_range_km}km range, {config.transmitter_power_kw}kW")
    
    def start_worker_thread(self):
        """Start async processing thread"""
        self.worker_thread = threading.Thread(target=self.worker_function, daemon=True)
        self.worker_thread.start()
        print("Async processing thread started")
    
    def worker_function(self):
        """Background processing worker"""
        while True:
            try:
                task = self.processing_queue.get(timeout=1.0)
                if task is None:
                    break
                    
                task_type, data = task
                if task_type == 'detection':
                    result = self.process_detections_async(data)
                    self.result_queue.put(('detection_result', result))
                
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker thread error: {e}")
    
    def process_detections_async(self, detection_data):
        """Process detections in background thread"""
        detections, current_time = detection_data
        targets = self.target_detector.process_raw_detections(detections)
        
        if targets:
            active_tracks = self.tracker.update(targets, current_time)
            return self.tracker.get_confirmed_tracks()
        return []
    
    def setup_ultimate_display(self):
        """Setup comprehensive display with all features"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(24, 16))
        self.fig.patch.set_facecolor('black')
        
        # Complex grid layout for all features
        gs = self.fig.add_gridspec(4, 6, height_ratios=[3, 1, 1, 1], width_ratios=[3, 1, 1, 1, 1, 1])
        
        # Main radar display
        self.axes['radar'] = self.fig.add_subplot(gs[0, :3], projection='polar')
        self.setup_radar_scope()
        
        # Mode and preset controls
        self.axes['modes'] = self.fig.add_subplot(gs[0, 3])
        self.axes['presets'] = self.fig.add_subplot(gs[0, 4])
        self.axes['performance'] = self.fig.add_subplot(gs[0, 5])
        
        # Real-time parameter sliders
        self.axes['sliders'] = self.fig.add_subplot(gs[1, :3])
        self.setup_parameter_sliders()
        
        # Status and monitoring panels
        self.axes['status'] = self.fig.add_subplot(gs[1, 3])
        self.axes['tracks'] = self.fig.add_subplot(gs[1, 4])
        self.axes['quality'] = self.fig.add_subplot(gs[1, 5])
        
        # Performance monitoring
        self.axes['fps_graph'] = self.fig.add_subplot(gs[2, :2])
        self.axes['resources'] = self.fig.add_subplot(gs[2, 2])
        self.axes['timing'] = self.fig.add_subplot(gs[2, 3])
        self.axes['optimization'] = self.fig.add_subplot(gs[2, 4])
        self.axes['threading'] = self.fig.add_subplot(gs[2, 5])
        
        # System controls and info
        self.axes['controls'] = self.fig.add_subplot(gs[3, :2])
        self.axes['alerts'] = self.fig.add_subplot(gs[3, 2])
        self.axes['filters'] = self.fig.add_subplot(gs[3, 3])
        self.axes['config_info'] = self.fig.add_subplot(gs[3, 4])
        self.axes['system_info'] = self.fig.add_subplot(gs[3, 5])
        
        # Style all panels
        for name, ax in self.axes.items():
            if name not in ['radar', 'sliders']:
                ax.set_facecolor('#001122')
                for spine in ax.spines.values():
                    spine.set_color('#00ff00')
                    spine.set_linewidth(1)
                ax.tick_params(colors='#00ff00', labelsize=8)
        
        # Title
        self.fig.suptitle('ULTIMATE ADVANCED RADAR SYSTEM - DAY 7 COMPLETE INTEGRATION', 
                         fontsize=20, color='#00ff00', weight='bold', y=0.96)
    
    def setup_radar_scope(self):
        """Configure main radar display"""
        ax = self.axes['radar']
        ax.set_facecolor('black')
        ax.set_ylim(0, self.current_config.max_range_km)
        ax.set_title('ADVANCED RADAR PPI SCOPE\nMulti-Mode Operation with Performance Optimization', 
                    color='#00ff00', pad=20, fontsize=14, weight='bold')
        
        # Dynamic range rings
        max_range = self.current_config.max_range_km
        for r in [max_range*0.25, max_range*0.5, max_range*0.75, max_range]:
            circle = Circle((0, 0), r, fill=False, color='#00ff00', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
            ax.text(np.pi/4, r-max_range*0.05, f'{r:.0f}km', color='#00ff00', fontsize=10, ha='center')
        
        # Bearing lines
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            ax.plot([rad, rad], [0, max_range], color='#00ff00', alpha=0.2, linewidth=0.5)
            ax.text(rad, max_range*1.05, f'{angle}°', color='#00ff00', fontsize=9, ha='center')
        
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.grid(True, color='#00ff00', alpha=0.2)
        ax.set_rticks([])
        ax.set_thetagrids([])
    
    def setup_parameter_sliders(self):
        """Setup interactive parameter adjustment sliders"""
        ax = self.axes['sliders']
        ax.set_title('REAL-TIME PARAMETER ADJUSTMENT', color='#00ff00', fontsize=12, weight='bold')
        ax.axis('off')
        
        # Range slider
        slider_ax1 = plt.axes([0.1, 0.7, 0.3, 0.02], facecolor='#001122')
        self.sliders['range'] = Slider(
            slider_ax1, 'Range (km)', 50, 500, 
            valinit=self.current_config.max_range_km, 
            color='#00ff00'
        )
        
        # Sensitivity slider
        slider_ax2 = plt.axes([0.1, 0.65, 0.3, 0.02], facecolor='#001122')
        self.sliders['threshold'] = Slider(
            slider_ax2, 'Sensitivity', 0.01, 0.3, 
            valinit=self.current_config.detection_threshold,
            color='#ffff00'
        )
        
        # Power slider
        slider_ax3 = plt.axes([0.1, 0.6, 0.3, 0.02], facecolor='#001122')
        self.sliders['power'] = Slider(
            slider_ax3, 'Power (kW)', 10, 500, 
            valinit=self.current_config.transmitter_power_kw,
            color='#ff4400'
        )
        
        # Sweep rate slider
        slider_ax4 = plt.axes([0.1, 0.55, 0.3, 0.02], facecolor='#001122')
        self.sliders['sweep_rate'] = Slider(
            slider_ax4, 'Sweep (RPM)', 5, 120, 
            valinit=self.current_config.sweep_rate_rpm,
            color='#ff8800'
        )
        
        # Connect slider events
        for name, slider in self.sliders.items():
            slider.on_changed(lambda val, n=name: self.on_slider_change(n, val))
    
    def load_comprehensive_scenario(self):
        """Load comprehensive test scenario"""
        print("Loading comprehensive radar test scenario...")
        
        # Diverse aircraft at various ranges and speeds
        aircraft_data = [
            (-150, 180, 90, 450),   (-80, 160, 45, 380),   (120, 140, 225, 520),
            (-180, -120, 135, 420), (160, -100, 315, 360), (0, 200, 180, 600),
            (-120, 80, 270, 280),   (180, 60, 225, 480),   (-100, -160, 45, 340),
            (140, 120, 135, 380),   (-200, 40, 90, 520),   (200, -80, 270, 400),
            (-60, 220, 180, 950),   (80, -180, 315, 150),  (-140, 160, 45, 680)
        ]
        
        for x, y, heading, speed in aircraft_data:
            self.data_generator.add_aircraft(x, y, heading, speed)
            
        # Naval vessels
        ship_data = [
            (-180, -160, 45, 25), (160, -180, 270, 18), (-120, -200, 90, 35),
            (200, -140, 225, 22), (-160, -120, 135, 28), (180, -160, 315, 15)
        ]
        
        for x, y, heading, speed in ship_data:
            self.data_generator.add_ship(x, y, heading, speed)
            
        # Weather phenomena
        self.data_generator.add_weather_returns(-120, 120, 45)
        self.data_generator.add_weather_returns(140, 160, 35)
        self.data_generator.add_weather_returns(-80, -140, 25)
        
        total = len(self.data_generator.targets)
        print(f"Comprehensive scenario loaded: {total} targets (aircraft, ships, weather)")
    
    def animate_ultimate(self, frame):
        """Ultimate animation loop with all features"""
        self.profiler.start_frame()
        
        if not self.is_running:
            self.update_static_displays()
            self.profiler.end_frame()
            return []
        
        # Update system
        self.current_time += 1.0 / self.target_fps
        self.frame_count += 1
        
        # Mode-specific sweep behavior
        mode_config = self.get_mode_sweep_config()
        self.sweep_angle = (self.sweep_angle + mode_config['sweep_rate'] * 0.1) % 360
        
        # Update targets
        detection_start = time.perf_counter()
        self.data_generator.update_targets(1.0 / self.target_fps)
        
        # Process detections with current configuration
        self.process_ultimate_detection(mode_config)
        detection_time = (time.perf_counter() - detection_start) * 1000
        
        # Update displays
        render_start = time.perf_counter()
        self.update_ultimate_radar_display()
        self.update_all_panels()
        render_time = (time.perf_counter() - render_start) * 1000
        
        # Memory management
        self.manage_memory()
        
        # Performance tracking
        metrics = self.profiler.end_frame()
        metrics.detection_time_ms = detection_time
        metrics.rendering_time_ms = render_time
        metrics.confirmed_tracks = len(self.tracker.get_confirmed_tracks())
        
        # Adaptive quality management
        new_quality = self.quality_manager.update_quality(metrics.frame_rate)
        metrics.quality_level = new_quality
        
        # Store performance data
        self.performance_history.append({
            'frame_rate': metrics.frame_rate,
            'frame_time_ms': metrics.frame_time_ms,
            'cpu_usage': metrics.cpu_usage,
            'memory_mb': metrics.memory_usage_mb,
            'quality': new_quality
        })
        
        return []
    
    def get_mode_sweep_config(self) -> Dict[str, float]:
        """Get sweep configuration based on current mode"""
        mode_configs = {
            RadarMode.SEARCH: {'sweep_rate': 30.0, 'beam_width': 30.0},
            RadarMode.TRACK: {'sweep_rate': 60.0, 'beam_width': 10.0},
            RadarMode.TWS: {'sweep_rate': 45.0, 'beam_width': 20.0},
            RadarMode.WEATHER: {'sweep_rate': 15.0, 'beam_width': 40.0},
            RadarMode.STANDBY: {'sweep_rate': 0.0, 'beam_width': 0.0}
        }
        return mode_configs.get(self.current_config.current_mode, mode_configs[RadarMode.SEARCH])
    
    def process_ultimate_detection(self, mode_config):
        """Process detections with mode-specific and performance optimizations"""
        if mode_config['sweep_rate'] == 0:
            return
            
        # Get quality settings
        quality_settings = self.quality_manager.get_quality_settings()
        
        # Adjust beam width based on quality
        beam_width = mode_config['beam_width'] * quality_settings['detail_level']
        
        # Get detections
        detections = self.data_generator.simulate_radar_detection(
            self.sweep_angle, sweep_width_deg=beam_width
        )
        
        if not detections:
            return
            
        # Apply configuration filters
        filtered_detections = [
            d for d in detections 
            if self.current_config.min_range_km <= d.get('range', 0) <= self.current_config.max_range_km
        ]
        
        # Limit detections based on quality
        max_detections = int(100 * quality_settings['detail_level'])
        if len(filtered_detections) > max_detections:
            filtered_detections = sorted(filtered_detections, 
                                       key=lambda d: d.get('signal_strength', 0), 
                                       reverse=True)[:max_detections]
        
        # Process detections
        if self.async_processing and not self.processing_queue.full():
            try:
                self.processing_queue.put_nowait(('detection', (filtered_detections, self.current_time)))
            except queue.Full:
                pass
        else:
            targets = self.target_detector.process_raw_detections(filtered_detections)
            if targets:
                self.tracker.update(targets, self.current_time)
    
    def update_ultimate_radar_display(self):
        """Ultimate radar display with all features"""
        ax = self.axes['radar']
        ax.clear()
        
        # Reconfigure scope for current range
        max_range = self.current_config.max_range_km
        ax.set_ylim(0, max_range)
        
        # Draw range rings
        for r in [max_range*0.25, max_range*0.5, max_range*0.75, max_range]:
            circle = Circle((0, 0), r, fill=False, color='#00ff00', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
            ax.text(np.pi/4, r-max_range*0.05, f'{r:.0f}km', color='#00ff00', fontsize=10, ha='center')
        
        # Mode-specific sweep display
        sweep_rad = np.radians(self.sweep_angle)
        mode_config = self.get_mode_sweep_config()
        quality_settings = self.quality_manager.get_quality_settings()
        
        # Mode-specific colors
        mode_colors = {
            RadarMode.SEARCH: '#00ff00',
            RadarMode.TRACK: '#ff4400', 
            RadarMode.TWS: '#00ffff',
            RadarMode.WEATHER: '#ffff00',
            RadarMode.STANDBY: '#404040'
        }
        sweep_color = mode_colors[self.current_config.current_mode]
