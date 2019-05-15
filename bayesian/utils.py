import torch

def heteroscedastic_loss(mean, log_var, true):
    precision = torch.exp(-log_var)
    true = true.type('torch.cuda.FloatTensor')
    sum_ = torch.sum(precision * (true - mean)**2 + log_var, 1)
    out = torch.mean(sum_, 0)
    return out
