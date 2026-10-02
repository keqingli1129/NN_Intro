"""
The tiny "AI brain" from the video: 2 inputs -> 2 hidden neurons -> 1 output, 5 weights.

    h1 = (x1 + w1*x2)^2          hidden neuron 1 (uses a squaring "activation")
    h2 = w2 * x1 * x2            hidden neuron 2
    y  = w3*h1 + w4*h2 + w5      output neuron (w5 acts as a bias)

Task: learn  f(x1, x2) = 2*x1^2 + 3*x2  from three examples.
Shared by sections 4-7 so the forward/backward code lives in one place.
"""

# Training data: (x1, x2) -> true output, computed from 2*x1^2 + 3*x2
DATA = [
    ((3, 2), 24),                       # 2*9 + 3*2 = 18 + 6 = 24
    ((1, 4), 14),                       # 2*1 + 3*4 =  2 + 12 = 14
    ((2, 1), 11),                       # 2*4 + 3*1 =  8 + 3  = 11
]

# The random-ish starting weights used in the video: [w1, w2, w3, w4, w5]
INITIAL_WEIGHTS = [1.0, 2.0, 1.0, 1.0, 0.0]


def forward(w, x1, x2):
    """Forward pass: push the inputs left-to-right through the network."""
    w1, w2, w3, w4, w5 = w              # unpack the five knobs
    h1 = (x1 + w1 * x2) ** 2            # hidden neuron 1
    h2 = w2 * x1 * x2                   # hidden neuron 2
    y = w3 * h1 + w4 * h2 + w5          # final prediction
    return h1, h2, y                    # return intermediates too - backprop reuses them


def loss(y_pred, y_true):
    """Squared error: always positive, and it punishes big mistakes much harder."""
    return (y_pred - y_true) ** 2


def backward(w, x1, x2, y_true):
    """Backpropagation: trace the blame right-to-left and return dLoss/dw for all 5 weights."""
    w1, _w2, w3, w4, _w5 = w            # w2, w5 never appear in a gradient formula
    h1, h2, y = forward(w, x1, x2)      # we need the forward values to compute slopes

    # The shared starting point of every chain - computed ONCE and reused below.
    dL_dy = 2 * (y - y_true)            # derivative of (y - y_true)^2 w.r.t. y

    # Output-layer weights: one chain-rule step from the error.
    dL_dw5 = dL_dy * 1                  # dy/dw5 = 1
    dL_dw4 = dL_dy * h2                 # dy/dw4 = h2
    dL_dw3 = dL_dy * h1                 # dy/dw3 = h1

    # Propagate the blame back into the hidden neurons.
    dL_dh1 = dL_dy * w3                 # dy/dh1 = w3
    dL_dh2 = dL_dy * w4                 # dy/dh2 = w4

    # Hidden-layer weights: one more chain-rule step.
    dL_dw1 = dL_dh1 * 2 * (x1 + w1 * x2) * x2   # dh1/dw1 = 2*(x1 + w1*x2) * x2
    dL_dw2 = dL_dh2 * x1 * x2                   # dh2/dw2 = x1 * x2

    return [dL_dw1, dL_dw2, dL_dw3, dL_dw4, dL_dw5]


def step(w, grads, lr):
    """Gradient descent update: new weight = old weight - learning_rate * gradient."""
    return [wi - lr * gi for wi, gi in zip(w, grads)]


def total_loss(w, data=DATA):
    """Sum of squared errors over a set of examples."""
    return sum(loss(forward(w, x1, x2)[2], t) for (x1, x2), t in data)
