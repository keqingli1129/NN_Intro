"""
PyTorch, Part 3 - Operations, the verbs of PyTorch.

  *  vs  @         element-wise multiply vs matrix multiply (the #1 beginner mix-up)
  .mean(dim=...)   reductions, and which direction `dim` collapses
  [ : , 2 ]        basic indexing
  argmax, gather   picking the best index, and picking a different column per row
"""

import torch

# ---------------------------------------------------------------------------
# Element-wise multiplication:  *   (shapes must match; multiply matching positions)
# ---------------------------------------------------------------------------
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[10, 20], [30, 40]])
print("A * B (element-wise):")
print(A * B, "\n")                              # [[1*10, 2*20], [3*30, 4*40]] = [[10, 40], [90, 160]]

# ---------------------------------------------------------------------------
# Matrix multiplication:  @   (columns of the first must equal rows of the second)
# ---------------------------------------------------------------------------
M1 = torch.tensor([[1, 2, 3], [4, 5, 6]])       # shape 2x3
M2 = torch.tensor([[7, 8], [9, 10], [11, 12]])  # shape 3x2  -> inner dims 3 and 3 match
product = M1 @ M2                               # result takes the outer dims: 2x2
print(f"M1 {tuple(M1.shape)} @ M2 {tuple(M2.shape)} -> {tuple(product.shape)}:")
print(product)                                  # [[58, 64], [139, 154]]
print("  e.g. top-left = 1*7 + 2*9 + 3*11 =", 1 * 7 + 2 * 9 + 3 * 11)
try:
    M1 @ M1                                     # 2x3 @ 2x3: inner dims 3 and 2 don't match
except RuntimeError as e:
    print("  M1 @ M1 fails:", str(e).splitlines()[0])
print("  A linear layer y = xW + b always uses @.\n")

# ---------------------------------------------------------------------------
# Reductions and the dim argument.
# Rows = 2 students, columns = 3 assignments.
# ---------------------------------------------------------------------------
scores = torch.tensor([[10.0, 20.0, 30.0],      # student 1
                       [5.0, 10.0, 15.0]])      # student 2
print("scores (rows = students, columns = assignments):")
print(scores)
print("scores.mean()       =", scores.mean().item(), " (all 6 numbers)")
print("scores.mean(dim=0)  =", scores.mean(dim=0), " <- collapse rows (vertical): per ASSIGNMENT")
print("scores.mean(dim=1)  =", scores.mean(dim=1), " <- collapse columns (horizontal): per STUDENT")
print("Rule: `dim` is the dimension that DISAPPEARS.\n")

# ---------------------------------------------------------------------------
# Basic indexing.
# ---------------------------------------------------------------------------
X = torch.arange(12).reshape(3, 4)              # 0..11 laid out as 3 rows x 4 columns
print("X:")
print(X)
print("X[:, 2] (all rows, column 2) =", X[:, 2], "\n")

# ---------------------------------------------------------------------------
# argmax: index of the highest value - how you read off a model's prediction.
# ---------------------------------------------------------------------------
pred_scores = torch.tensor([[5, 10, 15, 20],    # best is 20 at index 3
                            [10, 30, 20, 5]])   # best is 30 at index 1
print("argmax per row:", torch.argmax(pred_scores, dim=1), "\n")

# ---------------------------------------------------------------------------
# gather: a different column for each row, in one vectorised operation (no Python loop).
# ---------------------------------------------------------------------------
data = torch.tensor([[10, 11, 12, 13],
                     [20, 21, 22, 23],
                     [30, 31, 32, 33]])
indices_to_select = torch.tensor([[2],          # from row 0 take column 2
                                  [0],          # from row 1 take column 0
                                  [3]])         # from row 2 take column 3
selected = torch.gather(data, dim=1, index=indices_to_select)  # dim=1: the indices are columns
print("gather picks:", selected.flatten().tolist(), " (row0[2]=12, row1[0]=20, row2[3]=33)")
# Where you'll meet it: cross-entropy needs p(correct token) for each row - exactly a gather.
# Lesson 13 did the same thing with NumPy: probs[np.arange(N), Y].
