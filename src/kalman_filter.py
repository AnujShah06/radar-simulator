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
    x: float          #x position (km)
    y: float          #y position (km)
    vx: float         #x velocity (km/s)
    vy: float         #y velocity (km/s)
    timestamp: float  #last update time
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    @property
    def velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)
    
    @property
    def speed_kmh(self) -> float:
        speed_ms = np.sqrt(self.vx**2 + self.vy**2)
        return speed_ms * 3.6  #convert m/s to km/h
    
    @property
    def heading_deg(self) -> float:
        heading_rad = np.arctan2(self.vx, self.vy)
        heading_deg = np.degrees(heading_rad)
        return heading_deg % 360

class KalmanFilter:
    """
    Kalman filter for tracking radar targets in 2D space
    State vector: [x, y, vx, vy] (position and velocity)
    """
    
    def __init__(self, dt: float = 1.0):
        """
        Initialize Kalman filter
        
        Args:
            dt: Time step in seconds
        """
        self.dt = dt
        
        #state vector: [x, y, vx, vy]
        self.state = np.zeros(4)  #[x, y, vx, vy]
        
        #state covariance matrix (uncertainty in state)
        self.P = np.eye(4) * 1000  #high initial uncertainty
        
        #state transition matrix (how state evolves)
        self.F = np.array([
            [1, 0, dt, 0 ],  #x = x + vx*dt
            [0, 1, 0,  dt],  #y = y + vy*dt  
            [0, 0, 1,  0 ],  #vx = vx (constant velocity)
            [0, 0, 0,  1 ]   #vy = vy (constant velocity)
        ])
        
        #process noise covariance (model uncertainty)
        #targets can accelerate/change course
        q = 0.1  #process noise parameter
        self.Q = np.array([
            [dt**4/4, 0,       dt**3/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0      ],
            [0,       dt**3/2, 0,       dt**2  ]
        ]) * q
        #measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],  #measure x
            [0, 1, 0, 0]   #measure y
        ])
        
        #measurement noise covariance (sensor uncertainty)
        r = 0.5  #measurement noise parameter (km)
        self.R = np.eye(2) * r**2
        
        #track quality metrics
        self.innovation_history = []
        self.likelihood_history = []
        
    def predict(self, dt: Optional[float] = None) -> TrackState:
        """
        Predict next state (time update)
        
        Args:
            dt: Time step override
            
        Returns:
            Predicted track state
        """
        if dt is not None and dt != self.dt:
            # update matrices for different time step
            self.update_time_step(dt)
        
        # predict state: x_pred = F * x
        self.state = self.F @ self.state
        
        # predict covariance: P_pred = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return TrackState(
            x=self.state[0],
            y=self.state[1], 
            vx=self.state[2],
            vy=self.state[3],
            timestamp=0  # will be set by caller
        )
    
    def update(self, measurement: Tuple[float, float], measurement_cov: Optional[np.ndarray] = None) -> TrackState:
        """
        Update state with measurement (measurement update)
        
        Args:
            measurement: (x, y) position measurement
            measurement_cov: Optional measurement covariance override
            
        Returns:
            Updated track state
        """
        z = np.array(measurement)  #measurement vector
        
        #use provided covariance or default
        R = measurement_cov if measurement_cov is not None else self.R
        
        #innovation (measurement residual): y = z - H*x
        innovation = z - self.H @ self.state
        
        #innovation covariance: S = H*P*H^T + R
        S = self.H @ self.P @ self.H.T + R
        
        #kalman gain: K = P*H^T*S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        #update state: x = x + K*y
        self.state = self.state + K @ innovation
        
        #update covariance: P = (I - K*H)*P
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        #store quality metrics
        self.innovation_history.append(np.linalg.norm(innovation))
        
        #calculate likelihood (how well measurement fits prediction)
        likelihood = self.calculate_likelihood(innovation, S)
        self.likelihood_history.append(likelihood)
        
        return TrackState(
            x=self.state[0],
            y=self.state[1],
            vx=self.state[2], 
            vy=self.state[3],
            timestamp=0  #will be set by caller
        )
    
    def calculate_likelihood(self, innovation: np.ndarray, innovation_cov: np.ndarray) -> float:
        """Calculate likelihood of measurement given prediction"""
        #multivariate normal probability density
        det_S = np.linalg.det(innovation_cov)
        if det_S <= 0:
            return 0.0
        
        exp_term = -0.5 * innovation.T @ np.linalg.inv(innovation_cov) @ innovation
        likelihood = np.exp(exp_term) / np.sqrt((2 * np.pi)**2 * det_S)
        
        return float(likelihood)