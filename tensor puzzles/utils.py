import torch


def arange(i: int):
    "Use this function to replace a for-loop."
    return torch.tensor(range(i))


def where(q, a, b):
    "Use this function to replace an if-statement."
    return (q * a) + (~q) * b