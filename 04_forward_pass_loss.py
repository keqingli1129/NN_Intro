"""
Section 4 - Forward Pass & Loss.

Feed every training example through the untrained network and measure how wrong it is.
No learning happens here - this is pure calculation.
"""

from tiny_net import (  # the network defined in tiny_net.py
    DATA,
    INITIAL_WEIGHTS,
    forward,
    loss,
)

w = INITIAL_WEIGHTS                     # w1=1, w2=2, w3=1, w4=1, w5=0
print(f"Weights: {w}\n")
print(f"{'input':>8} | {'h1':>5} {'h2':>5} | {'pred':>5} {'true':>5} | {'diff':>5} | {'sq err':>6}")

total = 0                               # running total of squared errors
for (x1, x2), y_true in DATA:           # loop over the three examples
    h1, h2, y = forward(w, x1, x2)      # forward pass for this example
    diff = y_true - y                   # raw error (can be negative)
    sq = loss(y, y_true)                # squared error (always positive)
    total += sq                         # add to the total
    print(f"{str((x1, x2)):>8} | {h1:>5.0f} {h2:>5.0f} | {y:>5.0f} {y_true:>5} | {diff:>5.0f} | {sq:>6.0f}")

print(f"\nTotal squared error = {total:.0f}   <- our enemy. Training should push this toward 0.")
print("Why square? A plain sum of diffs (-13 - 19 - 2) could cancel out if signs were mixed;")
print("squaring makes every error positive and punishes the big miss (19 -> 361) far more")
print("than the small one (2 -> 4).")
