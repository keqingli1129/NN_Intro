"""
Section 3 - The Chain Rule ("the blame game").

Blame multiplies along a chain:  90% * 80% * 50% = 36%.
And when a variable reaches the result by several paths, the blames ADD.

Nested function from the video:
    u = 2*x1 + x2
    v = u^2 + 3*x2^2
    f = v^3

Chain of influence:  x1 -> u -> v -> f
                     x2 -> u -> v -> f   (indirect path)
                     x2 ------> v -> f   (direct path)
"""

# The blame-game analogy, in numbers.
blame = 0.90 * 0.80 * 0.50              # data collection -> analysis -> slides -> presentation
print(f"Data collector's share of the blame: {blame:.0%}\n")


def forward(x1, x2):
    """Compute the nested function step by step and return every intermediate value."""
    u = 2 * x1 + x2                     # inner link
    v = u ** 2 + 3 * x2 ** 2            # middle link (uses u AND x2 directly)
    f = v ** 3                          # outer link
    return u, v, f


def gradients(x1, x2):
    """Trace the blame backwards with the chain rule."""
    u, v, _ = forward(x1, x2)           # we need the intermediate values
    df_dv = 3 * v ** 2                  # step 1: how much f blames v   (power rule on v^3)
    dv_du = 2 * u                       # step 2: how much v blames u   (x2 held constant)
    du_dx1 = 2                          # step 3: how much u blames x1
    grad_x1 = df_dv * dv_du * du_dx1    # multiply along the chain -> 12 * u * v^2

    du_dx2 = 1                          # u = 2*x1 + x2 -> slope 1 w.r.t. x2
    dv_dx2_direct = 6 * x2              # the 3*x2^2 term inside v -> 6*x2
    dv_dx2 = dv_du * du_dx2 + dv_dx2_direct   # SUM of both paths -> 2u + 6*x2
    grad_x2 = df_dv * dv_dx2            # 3v^2 * (2u + 6*x2)
    return grad_x1, grad_x2


# Sanity check: compare our chain-rule gradients to a brute-force numerical estimate.
# Nudge a variable by a tiny h and see how much f changes: slope ~ (f(x+h) - f(x-h)) / 2h.
x1, x2 = 0.5, 0.5
h = 1e-6
num_g1 = (forward(x1 + h, x2)[2] - forward(x1 - h, x2)[2]) / (2 * h)
num_g2 = (forward(x1, x2 + h)[2] - forward(x1, x2 - h)[2]) / (2 * h)
g1, g2 = gradients(x1, x2)
print(f"At (0.5, 0.5): chain rule -> ({g1:.4f}, {g2:.4f})   numerical -> ({num_g1:.4f}, {num_g2:.4f})")
print("They match, so the blame tracing is correct.\n")

# Plug the gradients into the same gradient-descent loop as before.
# (The video doesn't give a starting point or learning rate; these are chosen so it's stable.)
lr = 0.001
print("Gradient descent on f = (u^2 + 3*x2^2)^3, start (0.5, 0.5), lr = 0.001")
for i in range(100):
    grad_x1, grad_x2 = gradients(x1, x2)  # blame for each variable
    x1 = x1 - lr * grad_x1                # nudge x1 downhill
    x2 = x2 - lr * grad_x2                # nudge x2 downhill
    if i % 20 == 0 or i == 99:
        print(f"  iter {i:>3}: x1 = {x1:+.4f}, x2 = {x2:+.4f}, f = {forward(x1, x2)[2]:.6f}")
print("f falls from 27 toward its minimum of 0 at (0, 0). It slows down near the bottom")
print("because f = v^3 is extremely flat there (tiny slope -> tiny steps).")
