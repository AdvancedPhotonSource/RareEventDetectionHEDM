import torch 

def regression_loss(x, y):
    x = torch.nn.functional.normalize(x, dim=-1, p=2)
    y = torch.nn.functional.normalize(y, dim=-1, p=2)
    
    return 2 - 2 * (x * y).sum(dim=-1)