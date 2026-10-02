"""
LLM Pre-training, Part 2 - Tokenization ("bridging text and numbers").

A neural network only understands numbers, so text must become a list of integer IDs.
  Approach 1: word-level      -> short sequences, but huge vocabulary and breaks on new words
  Approach 2: character-level -> tiny vocabulary, any word works, but very long sequences
  Approach 3: subword (BPE)   -> the best of both; what GPT actually uses
"""

from collections import Counter

import tiktoken  # OpenAI's tokenizer library (installed with `uv add tiktoken`)

text = "The cat quickly jumped"

# ---------------------------------------------------------------------------
# Approach 1: word-level tokens - split on spaces.
# ---------------------------------------------------------------------------
training_text = "the cat jumped . the dog jumps . the cat is jumping"
word_vocab = sorted(set(training_text.split()))          # every distinct word seen in training
word_to_id = {w: i for i, w in enumerate(word_vocab)}    # word -> integer ID lookup table
print("WORD-LEVEL")
print(f"  vocabulary ({len(word_vocab)}): {word_vocab}")
print("  note: 'jumped', 'jumps', 'jumping' each need their own slot")
new_sentence = "the cat jumper".split()                  # 'jumper' was never seen in training
ids = [word_to_id.get(w, "<UNK>") for w in new_sentence] # unknown words have no ID at all
print(f"  'the cat jumper' -> {ids}   <- unseen word breaks it\n")

# ---------------------------------------------------------------------------
# Approach 2: character-level tokens - every letter is a token.
# ---------------------------------------------------------------------------
chars = list(text.lower())                  # 'the cat ...' -> ['t', 'h', 'e', ' ', 'c', ...]
print("CHARACTER-LEVEL")
print(f"  tokens: {chars}")
print(f"  vocabulary: tiny (a-z, digits, punctuation), and ANY word can be spelled")
print(f"  but sequence length: {len(text.split())} words -> {len(chars)} characters "
      "(much longer, much more expensive)\n")

# ---------------------------------------------------------------------------
# Approach 3: subword tokens, learned with Byte-Pair Encoding (BPE) - built from scratch.
# Idea: start from characters, then repeatedly glue together the most frequent
# neighbouring pair. Common chunks ("jump", "ing", "ly") become single tokens - Lego bricks.
# ---------------------------------------------------------------------------

def merge_pair(symbols, pair):
    """Replace every adjacent occurrence of `pair` in a word with one glued symbol."""
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])  # glue the two pieces together
            i += 2                                   # skip past both
        else:
            out.append(symbols[i])                   # keep this piece as-is
            i += 1
    return tuple(out)


def train_bpe(corpus, num_merges):
    """Learn a list of merge rules from a training corpus."""
    words = Counter(tuple(w) for w in corpus.split())    # each word as a tuple of characters, with its count
    merges = []                                          # the rules we learn, in order
    for _ in range(num_merges):
        pairs = Counter()                                # how often does each neighbouring pair occur?
        for word, freq in words.items():
            for a, b in zip(word, word[1:]):             # every adjacent pair inside the word
                pairs[(a, b)] += freq                    # weighted by how often the word appears
        if not pairs:                                    # every word is already a single token
            break
        best = max(pairs, key=pairs.get)                 # the most frequent pair wins
        merges.append(best)                              # remember the rule ...
        words = Counter({merge_pair(w, best): f for w, f in words.items()})  # ... and apply it
    return merges


def bpe_tokenize(word, merges):
    """Split a (possibly never-seen) word by replaying the learned merges in order."""
    symbols = tuple(word)                                # start from single characters
    for pair in merges:
        symbols = merge_pair(symbols, pair)              # apply each rule in the order it was learned
    # Mark pieces that continue a word with '##' (the notation used in the video).
    return [symbols[0]] + ["##" + s for s in symbols[1:]]


corpus = ("jump jumps jumping jumped jumper jump jumping quick quickly quick quickly "
          "slow slowly slow slowly read reading reads thread threads threading "
          "the cat the cat the dog play playing played plays") * 5
# The number of merges sets the vocabulary size. Too many and every training word becomes one
# token (word-level again); too few and we're back near characters. 22 shows the sweet spot.
merges = train_bpe(corpus, num_merges=22)
print("SUBWORD (BPE, trained from scratch on a tiny corpus)")
print(f"  first 10 learned merges: {merges[:10]}")
for w in ["cat", "quickly", "jumping", "jumped", "hyperthreading", "replaying"]:
    print(f"  {w:<15} -> {bpe_tokenize(w, merges)}")
print("  'hyperthreading' and 'replaying' were never in the corpus - built from known bricks.")
print("  'cat' is less frequent here than jump/read/play, so it hasn't earned a brick yet -> letters.\n")

# Final step: look each subword up in the vocabulary to get its integer ID.
pieces = [p for w in text.lower().split() for p in bpe_tokenize(w, merges)]
vocab = {p: i for i, p in enumerate(sorted(set(pieces)))}  # toy vocabulary just for this sentence
print(f"  {text!r} -> pieces {pieces}")
print(f"  -> IDs {[vocab[p] for p in pieces]}   <- this list of numbers is what the network sees\n")

# ---------------------------------------------------------------------------
# The real thing: GPT-2's tokenizer (50,257 subword tokens learned with BPE on web text).
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")         # downloads the vocabulary once, then caches it
print(f"GPT-2 TOKENIZER (vocabulary size {enc.n_vocab:,})")
for s in [text, "hyperthreading", "awesommme"]:
    ids = enc.encode(s)                     # text -> list of integer IDs
    pieces = [enc.decode([i]) for i in ids] # decode each ID back to its text chunk to see the split
    print(f"  {s!r:<26} -> {pieces}  IDs {ids}")
print("  (GPT-2's vocabulary is large, so common words like 'quickly' are a single token.)\n")

# ---------------------------------------------------------------------------
# The dirty secret: why LLMs struggle to count the r's in "strawberry".
# ---------------------------------------------------------------------------
ids = enc.encode("strawberry")
pieces = [enc.decode([i]) for i in ids]
print("WHY 'how many r's in strawberry?' IS HARD")
print(f"  you see    : 'strawberry'  -> count is {'strawberry'.count('r')}")
print(f"  model sees : {ids}  (three opaque symbols: {pieces})")
print("  The letters are hidden inside the token IDs; the model must 'remember' each token's")
print("  spelling and add up r's across token boundaries:",
      " + ".join(f"{p.count('r')}" for p in pieces), f"= {sum(p.count('r') for p in pieces)}")
