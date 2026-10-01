"""
Section 1 - Gradient Descent ("the foggy valley").

Idea: you can't see the whole map, only the slope under your feet.
  1. Feel the slope (compute the derivative / gradient).
  2. Take a small step downhill (move opposite to the slope).
  3. Repeat until the ground is flat (slope ~ 0).
"""

import matplotlib                      # plotting library we installed with `uv add matplotlib`
matplotlib.use("Agg")                  # "Agg" = draw to image files instead of opening a window
import matplotlib.pyplot as plt        # the plotting interface (plt.plot, plt.savefig, ...)
import numpy as np                     # numerical arrays, used here only to draw smooth curves
from pathlib import Path               # convenient, OS-independent file paths

PLOTS = Path(__file__).parent / "plots"  # folder next to this script where images will be saved
PLOTS.mkdir(exist_ok=True)               # create it if it doesn't already exist


# ---------------------------------------------------------------------------
# Part A: the perfect U-shaped valley  f(x) = x^2
# ---------------------------------------------------------------------------

def f(x):
    """The valley itself: the 'error' we want to make as small as possible."""
    return x ** 2                       # x squared -> a U shape whose bottom is at x = 0


def df(x):
    """The slope detector: derivative of x^2 is 2x."""
    return 2 * x                        # positive slope -> downhill is LEFT, negative -> RIGHT


x = 3.0                                 # start at a "random" spot on the hill, x = 3
learning_rate = 0.1                     # step size: how far we move each time
history = [x]                           # remember every position so we can plot the path

print("Gradient descent on f(x) = x^2, start x = 3, learning rate = 0.1")
print(f"{'iter':>4} | {'x':>10} | {'error x^2':>10} | {'grad 2x':>10} | {'new x':>10}")
for i in range(100):                    # the famous loop: runs 100 times
    grad = df(x)                        # 1) feel the slope where we're standing
    new_x = x - learning_rate * grad    # 2) update rule: step AGAINST the slope (downhill)
    if i < 10:                          # print only the first 10 rows so the table stays readable
        print(f"{i:>4} | {x:>10.4f} | {f(x):>10.4f} | {grad:>10.4f} | {new_x:>10.4f}")
    x = new_x                           # 3) we've moved; repeat from the new spot
    history.append(x)                   # record the position for the plot

print(f"After 100 iterations: x = {x:.10f}, error = {f(x):.2e}  (the bottom is x = 0)\n")

xs = np.linspace(-3.5, 3.5, 200)        # 200 evenly spaced x values to draw the curve smoothly
plt.figure(figsize=(6, 4))              # new blank figure, 6x4 inches
plt.plot(xs, f(xs), label="f(x) = x²")  # draw the valley
path = np.array(history[:15])           # the first 15 positions of our walk
plt.plot(path, f(path), "o-", color="red", label="gradient descent steps")  # dots = each step
plt.title("Sliding down the parabola")  # chart title
plt.legend()                            # show the labels
plt.savefig(PLOTS / "01_parabola.png", dpi=120)  # write the image to disk
plt.close()                             # free the figure's memory


# ---------------------------------------------------------------------------
# Part B: the plot twist - local minima ("tunnel vision")
# The transcript doesn't name the bumpy function, so we use a classic one:
#   g(x) = x^4 - 3x^2 + x
# It has a deep global minimum near x = -1.30 and a shallow trap near x = +1.13.
# ---------------------------------------------------------------------------

def g(x):
    """A bumpy landscape with two valleys."""
    return x ** 4 - 3 * x ** 2 + x


def dg(x):
    """Its derivative (power rule on each term): 4x^3 - 6x + 1."""
    return 4 * x ** 3 - 6 * x + 1


def descend(start, lr=0.01, steps=500):
    """Run plain gradient descent on g from a given starting point."""
    x = start                           # where we drop the ball
    path = [x]                          # keep the trail
    for _ in range(steps):              # same loop as before...
        x = x - lr * dg(x)              # ...same update rule, just a different slope function
        path.append(x)
    return x, path                      # final resting place, plus the whole trail


end_a, path_a = descend(+0.5)           # start on the right side of the central hump
end_b, path_b = descend(-0.5)           # start on the left side of the central hump
print("Bumpy landscape g(x) = x^4 - 3x^2 + x")
print(f"  start x = +0.5 -> stuck at x = {end_a:.4f}, g = {g(end_a):.4f}  (local minimum, a trap)")
print(f"  start x = -0.5 -> found    x = {end_b:.4f}, g = {g(end_b):.4f}  (global minimum, the best)")
print("Same algorithm, different starting points, different results.")

xs = np.linspace(-2.1, 2.1, 300)        # range wide enough to show both valleys
plt.figure(figsize=(6, 4))
plt.plot(xs, g(xs), label="g(x) = x⁴ − 3x² + x")
plt.plot(path_a, [g(p) for p in path_a], ".", color="orange", label="start +0.5 (trapped)")
plt.plot(path_b, [g(p) for p in path_b], ".", color="green", label="start −0.5 (global)")
plt.title("Local vs. global minimum")
plt.legend()
plt.savefig(PLOTS / "01_local_minima.png", dpi=120)
plt.close()
print(f"\nPlots saved to {PLOTS}")
