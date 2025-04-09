# model/fpn_binary.py

import torch.nn as nn
import segmentation_models_pytorch as smp

class HRNetBinary(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(HRNetBinary, self).__init__()
        self.model = smp.FPN(
            encoder_name="resnet34",        # or "resnet18" or "efficientnet-b0"
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )

    def forward(self, x):
        return self.model(x)
