import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from src.coordinate_utils import polar_to_cartesian

#figure with radar-like colors
plt.style.use('dark_background')  #dark theme
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')

#look like a radar screen
ax.set_facecolor('black')
ax.grid(True, color='green', alpha=0.3)
ax.set_theta_zero_location('N')  # 0 degrees at top (North)
ax.set_theta_direction(-1)  # Clockwise

# Add range rings every 50km
max_range = 200
range_rings = np.arange(50, max_range + 1, 50)
theta_full = np.linspace(0, 2*np.pi, 100)

for r in range_rings:
    ax.plot(theta_full, np.full_like(theta_full, r), 'g-', alpha=0.5, linewidth=0.5)
    ax.text(0, r, f'{r}km', color='green', fontsize=8)

# Add bearing lines every 30 degrees
for angle in range(0, 360, 30):
    theta_rad = np.radians(angle)
    ax.plot([theta_rad, theta_rad], [0, max_range], 'g-', alpha=0.5, linewidth=0.5)
    ax.text(theta_rad, max_range * 1.1, f'{angle}°', color='green', 
            ha='center', va='center', fontsize=8)

# Add targets
target_ranges = [75, 120, 150, 180]
target_bearings = [30, 120, 200, 310]

for r, bearing in zip(target_ranges, target_bearings):
    theta_rad = np.radians(bearing)
    ax.plot(theta_rad, r, 'ro', markersize=8, alpha=0.8)
    ax.text(theta_rad, r + 10, f'T{len(target_ranges)}', 
            color='red', ha='center', fontsize=8)

ax.set_ylim(0, max_range)
ax.set_title('Radar Display Practice', color='green', pad=20)

plt.show()