"""
Section 6 - Online vs. Batch vs. Mini-batch learning.

Online : look at one example -> update immediately -> next example. (impulsive, noisy)
Batch  : compute every example's blame -> SUM them -> update once.   (deliberate, stable, slow)
Mini-batch: the real-world compromise - sum the blame over a small group (e.g. 32), update, repeat.
"""

import random
from tiny_net import DATA, INITIAL_WEIGHTS, forward, loss, backward, step, total_loss

lr = 0.0001                             # same learning rate as section 5 (matches the video's numbers)
names = ["w1", "w2", "w3", "w4", "w5"]

# ---------------------------------------------------------------------------
# One full-batch update, step by step.
# ---------------------------------------------------------------------------
w = INITIAL_WEIGHTS
print(f"Step 1 - forward pass on everyone. Total error = {total_loss(w):.0f}\n")

print("Step 2 - blame per example (no updating yet):")
total_grad = [0.0] * 5                  # one running sum per weight
for (x1, x2), y_true in DATA:
    g = backward(w, x1, x2, y_true)     # this example's blame for each weight
    print(f"  {str((x1, x2)):>6}: " + "  ".join(f"{n}={v:>6.0f}" for n, v in zip(names, g)))
    total_grad = [t + gi for t, gi in zip(total_grad, g)]   # add it to the column totals

# Note: the video shows 380 for example (1,4)'s w1 blame (total 924). The correct value is
# 38 * w3 * 2*(1 + 1*4) * 4 = 1520 (it dropped the final "* x2"), so our totals differ from the video's.
print("  total : " +"  ".join(f"{n}={v:>6.0f}" for n, v in zip(names, total_grad)))

w_batch = step(w, total_grad, lr)       # Step 3 - ONE update using the summed blame
print("\nStep 3 - single update: " + ", ".join(f"{n}={v:.4f}" for n, v in zip(names, w_batch)))

print("\nDid it help every example?")
for (x1, x2), y_true in DATA:
    before = loss(forward(w, x1, x2)[2], y_true)
    after_pred = forward(w_batch, x1, x2)[2]
    print(f"  {str((x1, x2)):>6}: error {before:>5.0f} -> {loss(after_pred, y_true):>7.2f}  (pred {after_pred:.2f})")
print(f"Total error: {total_loss(w):.0f} -> {total_loss(w_batch):.2f}\n")


# ---------------------------------------------------------------------------
# The three strategies side by side, trained for a while.
# ---------------------------------------------------------------------------
def train_online(w, epochs):
    for _ in range(epochs):
        for (x1, x2), y in DATA:
            w = step(w, backward(w, x1, x2, y), lr)       # update inside the loop
    return w


def train_batch(w, epochs):
    for _ in range(epochs):
        g = [0.0] * 5
        for (x1, x2), y in DATA:
            g = [a + b for a, b in zip(g, backward(w, x1, x2, y))]  # accumulate only
        w = step(w, g, lr)                                 # update once, outside the loop
    return w


def train_minibatch(w, epochs, batch_size=2, seed=0):
    rng = random.Random(seed)            # fixed seed so the run is reproducible
    data = list(DATA)
    for _ in range(epochs):
        rng.shuffle(data)                # new random order every epoch
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]          # a small handful of examples
            g = [0.0] * 5
            for (x1, x2), y in batch:
                g = [a + b for a, b in zip(g, backward(w, x1, x2, y))]
            w = step(w, g, lr)           # update after each mini-batch
    return w


epochs = 2000
print(f"Training each strategy for {epochs} epochs (passes over the data):")
for label, trainer in [("online", train_online), ("batch", train_batch), ("mini-batch", train_minibatch)]:
    w_final = trainer(INITIAL_WEIGHTS, epochs)
    print(f"  {label:>10}: total error = {total_loss(w_final):.4f}")
print("(This 5-weight net can't represent 2*x1^2 + 3*x2 exactly - it has no plain '3*x2' term -")
print(" so it only approximates the pattern; the error keeps creeping down slowly, see section 7.)")
