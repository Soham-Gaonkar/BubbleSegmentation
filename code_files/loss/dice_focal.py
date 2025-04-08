# loss/dice_focal.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, smooth=1e-5):
        super().__init__()
        if dice_weight + focal_weight != 1.0:
            warnings.warn("Dice and Focal weights do not sum to 1. Adjust if needed.")
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_prob = torch.sigmoid(inputs).clamp(min=1e-7, max=1 - 1e-7)
        targets = targets.float()

        # Dice Loss
        intersection = (inputs_prob * targets).sum(dim=(2, 3))
        total = (inputs_prob + targets).sum(dim=(2, 3))
        dice_coeff = (2. * intersection + self.smooth) / (total + self.smooth)
        dice_loss = 1. - dice_coeff.mean()

        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss
