#!/usr/bin/env python3
"""Debug wall proximity field."""

import numpy as np
from efi.core import wall_proximity_field, diffuse_masked

# Create simple wall mask
walls = np.zeros((10, 10), dtype=bool)
walls[0, :] = True  # Top wall
walls[-1, :] = True  # Bottom wall
walls[:, 0] = True  # Left wall
walls[:, -1] = True  # Right wall

print("Wall mask:")
for y in range(10):
    row = []
    for x in range(10):
        row.append('#' if walls[y, x] else '.')
    print(''.join(row))

print("\nTesting wall_proximity_field function:")
W_prox = wall_proximity_field(walls, radius=1.5)

print("\nWall proximity values (should be high near walls):")
# Show a slice
for y in range(10):
    row = []
    for x in range(10):
        val = W_prox[y, x]
        if walls[y, x]:
            row.append('#')
        elif val > 0.5:
            row.append('H')  # High proximity
        elif val > 0.2:
            row.append('M')  # Medium proximity
        elif val > 0.0:
            row.append('L')  # Low proximity
        else:
            row.append('.')  # Zero
    print(''.join(row))

print("\nActual values at key positions:")
print(f"  (1,1) corner: {W_prox[1,1]:.3f}")
print(f"  (1,5) near wall: {W_prox[1,5]:.3f}")
print(f"  (5,5) center: {W_prox[5,5]:.3f}")

# Let's trace through the function manually
print("\n" + "="*50)
print("Manual trace of wall_proximity_field:")

# Start with walls as source
W = walls.astype(np.float32)
print(f"Initial W sum: {np.sum(W)}")

# Diffuse wall presence
steps = max(1, int(1.5 * 2))  # radius=1.5 -> 3 steps
print(f"Diffusion steps: {steps}")

W_prox_manual = diffuse_masked(W, walls, diff=0.25, decay=0.05, steps=steps)
print(f"After diffusion sum: {np.sum(W_prox_manual)}")

# Scale
W_prox_manual = np.clip(W_prox_manual * 2.0, 0, 1.0)
print(f"After scaling sum: {np.sum(W_prox_manual)}")

print("\nManual values at key positions:")
print(f"  (1,1) corner: {W_prox_manual[1,1]:.3f}")
print(f"  (1,5) near wall: {W_prox_manual[1,5]:.3f}")
print(f"  (5,5) center: {W_prox_manual[5,5]:.3f}")