# various custom loss functions in pytorch

import torch
from torch.autograd import Variable

def weighted_mse_loss(input, target):
    """
    weighted mse loss with input and target
    from pytorch community
    """

    # alpha of 0.5 means half weight goes to first, remaining half split by remaining 15
    weights = Variable(torch.Tensor([0.5,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15])).cuda()  
    pct_var = (input-target)**2
    out = pct_var * weights.expand_as(target)
    loss = out.mean() 
    return loss
