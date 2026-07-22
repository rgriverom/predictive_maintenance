import matplotlib.pyplot as plt
from data_loading import load_train, COLUMN_NAMES

train = load_train()

# Identify constant (zero-variance) sensors — known issue in CMAPSS,
# confirming it with real data instead of assuming which ones.
sensor_cols = [c for c in COLUMN_NAMES if c.startswith('sensor_')]
variances = train[sensor_cols].var()
constant_sensors = variances[variances == 0].index.tolist()
print('Constant sensors (no variance, safe to drop):', constant_sensors)
print('Informative sensors:', [s for s in sensor_cols if s not in constant_sensors])

# Plot a handful of informative sensors over time for one engine, to see
# what a degradation trajectory actually looks like.
engine_1 = train[train['unit_number'] == 1]
informative_sensors = [c for c in sensor_cols if c not in constant_sensors][:6]

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, sensor in zip(axes.flat, informative_sensors):
    ax.plot(engine_1['time_in_cycles'], engine_1[sensor])
    ax.set_title(sensor)
    ax.set_xlabel('cycle')
plt.tight_layout()
plt.show()