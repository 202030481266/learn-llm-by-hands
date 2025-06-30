import torch
from utils import arange


def cumsum_spec(a, out):
    total = 0
    for i in range(len(out)):
        out[i] = total + a[i]
        total += a[i]

def cumsum(a):
    row_idx = arange(a.shape[0])[:,None] 
    col_idx = arange(a.shape[0])
    mask = row_idx <= col_idx
    ones = arange(a.shape[0])[:, None] * 0 + arange(a.shape[0])[None, :] * 0 + 1
    ones = ones * mask
    return a @ (ones.to(a.dtype))


test = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
test_output = torch.zeros_like(test)
cumsum_spec(test, test_output)
print(test_output)
print(cumsum(test))