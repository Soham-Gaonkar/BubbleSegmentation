# ================================================
# File: code_files/model/resnet18.py
# ================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from config import Config

# --- Building Blocks ---

class ConvBlock(nn.Module):
    """Standard Double Convolution Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpConvResnet(nn.Module):
    """Upsampling block for ResNet-UNet"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # Use ConvTranspose2d for upsampling
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # Conv block takes concatenated channels (skip_ch + in_ch // 2)
        self.conv = ConvBlock(skip_ch + in_ch // 2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# --- ResNet18 U-Net Model ---

class ResNet18CNN(nn.Module):
    """U-Net architecture with a ResNet18 encoder."""
    def __init__(self, in_channels=3, num_classes=1, pretrained=False, dropout_prob=0.3):
        super().__init__()

        self.dropout = nn.Dropout2d(p=dropout_prob)

        # --- Encoder (ResNet18 Backbone) ---
        if pretrained:
            print("Loading pretrained ResNet18 weights.")
            weights = ResNet18_Weights.DEFAULT
        else:
            print("Initializing ResNet18 backbone weights from scratch (or using base random init).")
            weights = None

        resnet_model = resnet18(weights=weights)

        if in_channels != 3:
            print(f"Adapting ResNet18 first conv layer from 3 to {in_channels} channels.")
            original_conv1 = resnet_model.conv1
            self.encoder_conv1 = nn.Conv2d(in_channels, original_conv1.out_channels,
                                           kernel_size=original_conv1.kernel_size,
                                           stride=original_conv1.stride,
                                           padding=original_conv1.padding,
                                           bias=original_conv1.bias is not None)
        else:
            self.encoder_conv1 = resnet_model.conv1

        self.encoder_bn1 = resnet_model.bn1
        self.encoder_relu = resnet_model.relu
        self.encoder_maxpool = resnet_model.maxpool
        self.encoder_layer1 = resnet_model.layer1
        self.encoder_layer2 = resnet_model.layer2
        self.encoder_layer3 = resnet_model.layer3
        self.encoder_layer4 = resnet_model.layer4

        # Optionally freeze early layers
        if Config.FREEZE_BACKBONE:
            self._freeze_layers(until=Config.FREEZE_UNTIL)

        self.up4 = UpConvResnet(in_ch=512, skip_ch=256, out_ch=256)
        self.up3 = UpConvResnet(in_ch=256, skip_ch=128, out_ch=128)
        self.up2 = UpConvResnet(in_ch=128, skip_ch=64, out_ch=64)
        self.up1_conv = ConvBlock(in_channels=64 + 64, out_channels=64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _freeze_layers(self, until="layer2"):
        freeze = True
        for name, module in self.named_children():
            if freeze:
                for param in module.parameters():
                    param.requires_grad = False
            if name == until:
                freeze = False

    def forward(self, x):
        x0 = self.encoder_conv1(x)
        x0 = self.encoder_bn1(x0)
        x0 = self.encoder_relu(x0)

        p0 = self.encoder_maxpool(x0)
        e1 = self.encoder_layer1(p0)
        e2 = self.encoder_layer2(e1)
        e3 = self.encoder_layer3(e2)
        e4 = self.encoder_layer4(e3)

        d4 = self.up4(x1=e4, x2=e3)
        d3 = self.up3(x1=d4, x2=e2)
        d2 = self.up2(x1=d3, x2=e1)

        d2_upsampled = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        diffY = x0.size()[2] - d2_upsampled.size()[2]
        diffX = x0.size()[3] - d2_upsampled.size()[3]
        if diffY > 0 or diffX > 0:
            d2_upsampled = F.pad(d2_upsampled, [diffX // 2, diffX - diffX // 2,
                                                diffY // 2, diffY - diffY // 2])

        d1 = torch.cat([x0, d2_upsampled], dim=1)
        d1 = self.up1_conv(d1)
        d1 = self.dropout(d1)
        d0 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        logits = self.out_conv(d0)
        return logits

# Example Usage Block
if __name__ == '__main__':
    model_gray = ResNet18CNN(in_channels=1, num_classes=1, pretrained=False, dropout_prob=0.3)
    input_gray = torch.randn(2, 1, 256, 256)
    output_gray = model_gray(input_gray)
    print("ResNet18CNN (Gray Input, No Pretrained ImageNet)")
    print(f"Input shape: {input_gray.shape}")
    print(f"Output shape: {output_gray.shape}")
    num_params_gray = sum(p.numel() for p in model_gray.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params_gray:,}")

    print("-" * 20)

    model_rgb = ResNet18CNN(in_channels=3, num_classes=5, pretrained=False, dropout_prob=0.3)
    input_rgb = torch.randn(1, 3, 224, 224)
    output_rgb = model_rgb(input_rgb)
    print("ResNet18CNN (RGB Input, No Pretrained ImageNet)")
    print(f"Input shape: {input_rgb.shape}")
    print(f"Output shape: {output_rgb.shape}")
    num_params_rgb = sum(p.numel() for p in model_rgb.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params_rgb:,}")