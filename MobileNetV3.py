# 256*256
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MobileNetV3', 'mobilenetv3']

def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, input_channels=1, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 96  # Last channel from mobile_setting for 'small' mode
        if mode == 'large':
            raise NotImplementedError("Large mode not supported for this reconstruction task")
        elif mode == 'small':
            mobile_setting = [
                [3, 16,  16,  True,  'RE', 2],  # 256 -> 128
                [3, 72,  24,  False, 'RE', 2],  # 128 -> 64
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],  # 64 -> 32
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 2],  # 32 -> 16
                [3, 144, 48,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # Building first layer
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(input_channels, input_channel, 1, nlin_layer=Hswish)]  # Keep original 256x256
        self.classifier = []

        # Building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # Building encoder (remove pooling and classification layers)
        self.encoder = nn.Sequential(*self.features)

        # Decoder blocks (upsampling path)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 40, kernel_size=4, stride=2, padding=1, bias=False),  # 16 -> 32
            nn.BatchNorm2d(40),
            Hswish(inplace=True),
            nn.ConvTranspose2d(40, 24, kernel_size=4, stride=2, padding=1, bias=False),  # 32 -> 64
            nn.BatchNorm2d(24),
            Hswish(inplace=True),
            nn.ConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 64 -> 128
            nn.BatchNorm2d(16),
            Hswish(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),  # 128 -> 256
            nn.BatchNorm2d(8),
            Hswish(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=True),  # 256 -> 256
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        # Encoder feature extraction
        x = self.encoder(x)
        # Decoder image reconstruction
        x = self.decoder(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def mobilenetv3(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV3 model for image reconstruction
    """
    model = MobileNetV3(**kwargs)
    if pretrained:
        raise NotImplementedError("Pretrained weights not available for reconstruction model")
    return model

if __name__ == '__main__':
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build model and move to specified device
    model = mobilenetv3(input_channels=1, mode='small').to(device)
    # Test with manual input to verify output shape
    sample_input = torch.randn(1, 1, 256, 256).to(device)  # Sample 256x256 grayscale input
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [1, 1, 256, 256]
    # Assert output size to ensure 256x256
    h, w = output.shape[2], output.shape[3]
    assert h == 256 and w == 256, f"Expected 256x256 output, got {h}x{w}"
    # Print model summary with input shape (1, 256, 256)
    from torchsummary import summary
    print(summary(model, (1, 256, 256), device=device.type))
