"""
LLM Pre-training, capstone - the whole pipeline on a tiny model.

  raw text -> GPT-2 subword tokens -> (input, next-token) pairs     [Parts 1 & 2]
  -> logits -> softmax -> cross-entropy loss                         [Parts 3 & 4]
  -> gradients -> gradient descent, repeated                         [lessons 01-07]
  -> generate new text with temperature + top-p                      [Part 5]

Our "model" is a bigram model: a table W where W[a, b] is the logit for "token b comes next
after token a". It only looks at the LAST token of the context. A real LLM uses a transformer
to look at the whole context (the next video), but the training loop is exactly the same.
"""

import numpy as np                          # fast arrays: lets us process every example at once
import tiktoken

# ---------------------------------------------------------------------------
# 1. Data: a (very) small "internet".
# ---------------------------------------------------------------------------
corpus = """The cat sat on the mat. The dog sat on the rug. The cat chased the dog.
The dog chased the ball. The cat slept on the mat. The dog slept on the rug.
In Paris the people speak French. In Berlin the people speak German.
The weather today is sunny. The weather today is cloudy. The weather today is cold."""

# ---------------------------------------------------------------------------
# 2. Tokenize with GPT-2's real subword tokenizer.
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
gpt2_ids = enc.encode(corpus)               # text -> GPT-2 token IDs (values up to 50,256)

# GPT-2's vocabulary has 50,257 tokens but our corpus uses only a few dozen. To keep the
# weight table small we re-number just the tokens we actually see: 0, 1, 2, ...
vocab = sorted(set(gpt2_ids))               # the distinct GPT-2 IDs in our corpus
to_local = {g: i for i, g in enumerate(vocab)}     # GPT-2 ID -> our compact ID
V = len(vocab)                              # vocabulary size = number of outputs (logits) per step
ids = np.array([to_local[g] for g in gpt2_ids])    # the corpus as compact IDs
print(f"Corpus: {len(corpus)} characters -> {len(ids)} tokens, vocabulary of {V} distinct tokens")

# ---------------------------------------------------------------------------
# 3. Self-supervised pairs: input = current token, target = next token.
# ---------------------------------------------------------------------------
X = ids[:-1]                                # every token except the last is an input
Y = ids[1:]                                 # ... and the token right after it is its label
print(f"Training pairs created for free: {len(X)}\n")

# ---------------------------------------------------------------------------
# 4. The model's weights: one row of V logits for each possible previous token.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)              # seeded random generator -> reproducible run
W = rng.normal(0, 0.01, size=(V, V))        # start with tiny random logits: a blank-slate brain


def softmax_rows(logits):
    """Softmax applied to each row independently (one row = one prediction)."""
    z = logits - logits.max(axis=1, keepdims=True)   # stability trick from lesson 10
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 5. The training loop: forward -> loss -> backprop -> gradient descent -> repeat.
# ---------------------------------------------------------------------------
lr = 5.0                                    # learning rate (large is fine for this simple convex model)
N = len(X)
print(f"Loss if guessing uniformly: ln({V}) = {np.log(V):.3f}")
for step in range(301):
    logits = W[X]                           # FORWARD: look up the logit row for every input token (N x V)
    probs = softmax_rows(logits)            # turn every row into probabilities
    loss = -np.log(probs[np.arange(N), Y]).mean()   # LOSS: average cross-entropy (surprise) over all pairs

    grad_logits = probs.copy()              # BACKPROP: dLoss/dlogits = p - one_hot (lesson 11) ...
    grad_logits[np.arange(N), Y] -= 1       # ... subtract 1 at each correct token
    grad_logits /= N                        # ... and average over the batch, matching the mean loss
    grad_W = np.zeros_like(W)
    np.add.at(grad_W, X, grad_logits)       # send each example's gradient to the row it came from
    W -= lr * grad_W                        # GRADIENT DESCENT: nudge every weight downhill

    if step in (0, 10, 50, 100, 300):
        print(f"  step {step:>3}: loss = {loss:.3f}")


# ---------------------------------------------------------------------------
# 6. What did it learn? Peek at a few predictions.
# ---------------------------------------------------------------------------
def show_next(word):
    """Print the model's top predictions after a given GPT-2 token."""
    local = to_local[enc.encode(word)[0]]
    p = softmax_rows(W[[local]])[0]
    top = np.argsort(p)[::-1][:3]           # indices of the 3 highest probabilities
    guesses = ", ".join(f"{enc.decode([vocab[i]])!r} {p[i]:.0%}" for i in top)
    print(f"  after {word!r:<8}: {guesses}")


print("\nLearned next-token predictions:")
for w in [" cat", " weather", " speak", " sat"]:
    show_next(w)


# ---------------------------------------------------------------------------
# 7. Generation: same probabilities, but now we SAMPLE with temperature and top-p.
# ---------------------------------------------------------------------------
def generate(prompt, n_tokens, temperature=1.0, top_p=1.0, seed=0):
    rng = np.random.default_rng(seed)
    out = [to_local[g] for g in enc.encode(prompt)]  # start from the prompt's tokens
    for _ in range(n_tokens):
        logits = W[out[-1]] / temperature           # bigram: only the last token matters
        p = softmax_rows(logits[None, :])[0]
        order = np.argsort(p)[::-1]                 # tokens sorted from most to least likely
        cutoff = np.searchsorted(np.cumsum(p[order]), top_p) + 1   # size of the nucleus
        nucleus = order[:cutoff]
        p_nucleus = p[nucleus] / p[nucleus].sum()   # renormalise inside the nucleus
        out.append(rng.choice(nucleus, p=p_nucleus))  # draw one token
    return enc.decode([vocab[i] for i in out])     # compact IDs -> GPT-2 IDs -> text


print("\nGeneration from the prompt 'The':")
print(f"  T=0.3, top_p=0.9 : {generate('The', 12, 0.3, 0.9)!r}")
print(f"  T=1.0, top_p=0.9 : {generate('The', 12, 1.0, 0.9)!r}")
print(f"  T=2.0, top_p=1.0 : {generate('The', 12, 2.0, 1.0)!r}")
print("\nIt has learned local word patterns, but not facts: it can't link 'Paris' to 'French'")
print("because it only sees one token back. Fixing that is the transformer's job (self-attention).")
