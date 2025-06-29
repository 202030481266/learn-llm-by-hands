from utils import arange
import torch

def ones_spec(out):
    for i in range(len(out)):
        out[i] = 1
        
def ones(i: int): 
    t = arange(i)[None, :]
    return (t + 1. - t)[0]

test = torch.zeros(3)
ones_spec(test)
print(test)
print(ones(3))