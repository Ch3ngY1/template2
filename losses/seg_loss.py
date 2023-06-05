import torch.nn as nn
import torch.nn.functional as F
import torch

class seg_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lossfunc = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, target, pred):
        # target.shape = B * 1 * 224 * 224
        # pred.shape = B * 1 * 224 * 224
        B, _ , H, W = target.shape
        target = target.reshape(B, -1)
        pred = pred.reshape(B, -1)
        loss = self.lossfunc(input=pred, target=target)

        return loss
    


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs.cuda() * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    

class EntropyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lossfunc = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, targets, inputs):
        # inputs.shape = B * 1 * 224 * 224
        # targets.shape = B * 1 * 224 * 224
        B, _ , H, W = targets.shape
        targets = targets.reshape(B, -1).float().cuda()
        inputs = inputs.reshape(B, -1).cuda()
        loss = self.lossfunc(input=inputs, target=targets)

        return loss