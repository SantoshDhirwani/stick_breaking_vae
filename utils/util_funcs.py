import torch


def beta_func(a, b):
    return (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)).exp()


def logistic_func(x):
    return 1 / (1 + torch.exp(-x))
