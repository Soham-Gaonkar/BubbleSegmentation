import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNetMini(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout_prob=0.2):
        super(SimpleUNetMini, self).__init__()

        self.dropout = nn.Dropout2d(dropout_prob)  # ✅ Define once

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder
        self.up3 = self.up_block(256, 128)
        self.up2 = self.up_block(128, 64)
        self.up1 = self.up_block(64, 32)

        # Final output
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))

        x_b = self.bottleneck(F.max_pool2d(x3, 2))
        x_b = self.dropout(x_b)  # ✅ Dropout after bottleneck

        x = self.up3(x_b)
        x = self.dropout(x)      # ✅ Dropout after up3

        x = self.up2(x)
        x = self.dropout(x)      # ✅ Dropout after up2

        x = self.up1(x)

        out = self.final(x)
        return out  # ✅ Let loss handle sigmoid
