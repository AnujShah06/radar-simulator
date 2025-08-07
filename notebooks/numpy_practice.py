import numpy as np
import matplotlib.pyplot as plt

#create sine waves with different frequencies (radar signals)
print("=== Radar Signal Practice ===")
t = np.linspace(0, 1, 1000)  # 1 second of time
frequency1 = 5  # Hz
frequency2 = 20  # Hz

signal1 = np.sin(2 * np.pi * frequency1 * t)
signal2 = np.sin(2 * np.pi * frequency2 * t)
combined = signal1 + signal2

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(t[:100], signal1[:100])  # Show first 0.1 seconds
plt.title('Low Frequency Signal')

plt.subplot(1, 3, 2)
plt.plot(t[:100], signal2[:100])
plt.title('High Frequency Signal')

plt.subplot(1, 3, 3)
plt.plot(t[:100], combined[:100])
plt.title('Combined Signals')
plt.tight_layout()
plt.show()

# Practice 2: Add realistic noise
print("=== Adding Noise ===")
clean_signal = np.sin(2 * np.pi * 10 * t)
noise_levels = [0.1, 0.3, 0.5, 1.0]

plt.figure(figsize=(15, 3))
for i, noise_level in enumerate(noise_levels):
    noise = np.random.normal(0, noise_level, len(t))
    noisy_signal = clean_signal + noise
    
    plt.subplot(1, 4, i+1)
    plt.plot(t[:200], noisy_signal[:200])
    plt.title(f'Noise Level: {noise_level}')
plt.tight_layout()
plt.show()

#trigonometry for radar
print("Radar Trigonometry")
# Convert between polar (range, bearing) and Cartesian (x, y)
ranges = np.array([50, 100, 150, 200])  # kilometers
bearings_deg = np.array([0, 45, 90, 180])  # degrees
bearings_rad = np.radians(bearings_deg)  # convert to radians

# Convert to x, y coordinates
x = ranges * np.cos(bearings_rad)
y = ranges * np.sin(bearings_rad)

print("Range | Bearing | X | Y")
for i in range(len(ranges)):
    print(f"{ranges[i]:3.0f}km | {bearings_deg[i]:3.0f}° | {x[i]:6.1f} | {y[i]:6.1f}")