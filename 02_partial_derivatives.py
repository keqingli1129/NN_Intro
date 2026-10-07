"""
Section 2 - Partial Derivatives ("one knob at a time").

With many variables you can't ask for "the" slope. Instead ask for the slope
along each variable separately, pretending every other variable is frozen.

Example bowl:  f(x1, x2) = x1^2 + 2*x2^2      (lowest point at (0, 0))
  df/dx1: freeze x2, so 2*x2^2 is a constant -> derivative 0 -> left with 2*x1
  df/dx2: freeze x1, so x1^2 is a constant   -> derivative 0 -> left with 4*x2
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                   # save images instead of opening windows

import matplotlib.pyplot as plt
import numpy as np

PLOTS = Path(__file__).parent / "plots"
PLOTS.mkdir(exist_ok=True)


def f(x1, x2):
    """The oval bowl - our 'error' surface."""
    return x1 ** 2 + 2 * x2 ** 2


def grad_x1(x1, x2):
    """Partial derivative w.r.t. x1 (x2 treated as a constant)."""
    return 2 * x1


def grad_x2(x1, x2):
    """Partial derivative w.r.t. x2 (x1 treated as a constant)."""
    return 4 * x2


x1, x2 = 3.0, 2.0                       # starting point (3, 2): error = 9 + 8 = 17
lr = 0.1                                # learning rate
trail = [(x1, x2)]                      # remember the path for the contour plot

print("Gradient descent on f(x1,x2) = x1^2 + 2*x2^2, start (3, 2), lr = 0.1")
print(f"{'iter':>4} | {'x1':>8} {'x2':>8} | {'error':>9} | {'g_x1':>8} {'g_x2':>8}")
for i in range(100):
    g1 = grad_x1(x1, x2)                # slope in the x1 direction only
    g2 = grad_x2(x1, x2)                # slope in the x2 direction only
    if i < 10:
        print(f"{i:>4} | {x1:>8.4f} {x2:>8.4f} | {f(x1, x2):>9.4f} | {g1:>8.4f} {g2:>8.4f}")
    # Update BOTH knobs at the same time, each with its own gradient.
    # (Compute both gradients first, then update - so x2's update doesn't see a half-updated x1.)
    x1 = x1 - lr * g1
    x2 = x2 - lr * g2
    trail.append((x1, x2))

print(f"After 100 iterations: x1 = {x1:.2e}, x2 = {x2:.2e}, error = {f(x1, x2):.2e}")
print("Note: the first step cut the error from 17 to 8.64 - almost in half.")

# Contour plot: a top-down view of the bowl with our path drawn on it.
g1s, g2s = np.meshgrid(np.linspace(-3.5, 3.5, 200), np.linspace(-2.5, 2.5, 200))  # grid of points
plt.figure(figsize=(6, 4.5))
plt.contour(g1s, g2s, f(g1s, g2s), levels=20)          # rings of equal height ("elevation lines")
t = np.array(trail[:25])                                # first 25 positions
plt.plot(t[:, 0], t[:, 1], "o-", color="red", ms=3)     # our walk down into the bowl
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("Descending the 3D bowl (top view)")
plt.savefig(PLOTS / "02_bowl.png", dpi=120)
plt.close()
print(f"Plot saved to {PLOTS / '02_bowl.png'}")
