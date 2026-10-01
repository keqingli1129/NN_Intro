"""
LLM Pre-training, Part 1 - Self-supervised data ("the data labels itself").

Supervised learning needs a human to write every label (expensive, slow, limited).
Next-token prediction needs no humans: in any text, the "label" for a sequence of words
is simply the word that comes next. One document becomes many training examples for free.
"""


def make_training_pairs(tokens):
    """Walk through a document one token at a time, producing (input-so-far, next-token) pairs."""
    pairs = []                              # the training examples we're about to create
    for i in range(1, len(tokens)):         # start at 1: we need at least one token of input
        context = tokens[:i]                # INPUT: everything seen so far (tokens 0 .. i-1)
        target = tokens[i]                  # LABEL: the very next token - taken from the text itself
        pairs.append((context, target))     # store the example
    return pairs                            # n tokens -> n-1 examples


# ---------------------------------------------------------------------------
# "The cat sat on the mat" -> 5 training examples, zero human labelling.
# ---------------------------------------------------------------------------
sentence = "The cat sat on the mat"
tokens = sentence.split()                   # simple word-level split for now (Part 2 does better)
pairs = make_training_pairs(tokens)

print(f"Sentence: {sentence!r}  ({len(tokens)} words)\n")
print(f"{'#':>2} | {'input (context)':<22} | target")
for n, (context, target) in enumerate(pairs, start=1):
    print(f"{n:>2} | {' '.join(context):<22} | {target}")
print(f"\n{len(tokens)} words -> {len(pairs)} free training examples.\n")

# ---------------------------------------------------------------------------
# Scale: a ~2,000-word article becomes ~2,000 examples. The internet becomes a textbook.
# ---------------------------------------------------------------------------
article_words = 2000                        # length of a typical Wikipedia article
print(f"A {article_words:,}-word article -> {article_words - 1:,} examples")
gpt2_bytes = 40e9                           # GPT-2 was trained on ~40 GB of text (WebText)
bytes_per_token = 4                         # rough rule of thumb for English text
print(f"~40 GB of text (GPT-2 era) -> roughly {gpt2_bytes / bytes_per_token:,.0f} examples, "
      "all labelled for free.\n")

# ---------------------------------------------------------------------------
# Why such a simple task teaches so much: look at what one prediction requires.
# ---------------------------------------------------------------------------
prompt = "In Paris , the capital of France , the primary language spoken is"
context, target = prompt.split(), "French"
print(f"Input : {prompt}")
print(f"Target: {target}")
print("To get this right the model must learn:")
print("  - grammar          : 'is' is likely followed by a noun / adjective")
print(f"  - long context     : link the end back to 'Paris', {len(context) - 1} words earlier")
print("  - world facts      : Paris is in France; people there speak French")
print("  - ignore distractors: 'France' is the clue, 'primary' is not")
