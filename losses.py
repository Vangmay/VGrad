from tensor import Tensor
from modules import Module 
# Loss functions
class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor):
        loss = ((pred - target) ** 2).sum()
        return loss