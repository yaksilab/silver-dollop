import numpy as np

import matplotlib.pyplot as plt

masks = np.load(
    r"C:\YaksiData\astrolglialAnalysis\tests\data\combined_mean_image_seg.npy",
    allow_pickle=True,
).item()["masks"]
# Example set of points (x, y)

selected_region = np.where(masks == 237)
selected_region = np.transpose(selected_region)
selected_region = np.flip(selected_region, axis=1)
selected_region2 = np.where(masks == 235)
selected_region2 = np.transpose(selected_region2)
selected_region2 = np.flip(selected_region2, axis=1)


points = selected_region


# Calculate pairwise distances
distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))

# Calculate cumulative distances
cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

# New points on the x-axis
new_points = np.zeros_like(points)
new_points[:, 0] = cumulative_distances

# Plot original and translated points
plt.figure(figsize=(10, 5))

# Original points
plt.subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], color="blue")
plt.plot(points[:, 0], points[:, 1], linestyle="--", color="blue")
for i, txt in enumerate(points):
    plt.annotate(f"{txt}", (points[i, 0], points[i, 1]))

plt.title("Original Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# Translated points
plt.subplot(1, 2, 2)
plt.scatter(new_points[:, 0], new_points[:, 1], color="red")
plt.plot(new_points[:, 0], new_points[:, 1], linestyle="--", color="red")
for i, txt in enumerate(new_points):
    plt.annotate(f"{txt}", (new_points[i, 0], new_points[i, 1]))

plt.title("Translated Points to X-axis")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

plt.tight_layout()
plt.show()
