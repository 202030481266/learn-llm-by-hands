import torch
from utils import arange

def triu_spec(out):
    for i in range(len(out)):
        for j in range(len(out)):
            if i <= j:
                out[i][j] = 1
            else:
                out[i][j] = 0

def triu(a):
    # [n, m]
    row_idx = arange(a.shape[0])[:, None] # [n, 1]
    col_idx = arange(a.shape[1])
    a = a * 0 + 1
    return a * (row_idx <= col_idx)


test = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
test_output = torch.zeros_like(test)
triu_spec(test_output)
print(test_output)
print(triu(test))