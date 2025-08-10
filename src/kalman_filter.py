"""
Kalman filter implementation for radar target tracking
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class TrackState:
    """State of a tracked target"""
    x: float          # x position (km)
    y: float          # y position (km)
    vx: float         # x velocity (km/s)
    vy: float         # y velocity (km/s)
    timestamp: float  # last update time
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    @property
    def velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)
    
    @property
    def speed_kmh(self) -> float:
        speed_ms = np.sqrt(self.vx**2 + self.vy**2)
        return speed_ms * 3.6  # convert m/s to km/h
    
    @property
    def heading_deg(self) -> float:
        heading_rad = np.arctan2(self.vx, self.vy)
        heading_deg = np.degrees(heading_rad)
        return heading_deg % 360
