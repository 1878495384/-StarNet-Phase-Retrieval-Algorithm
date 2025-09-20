import torch
import torch.nn  as nn
import torch.nn.functional  as F
from torchsummary import summary

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x) 

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 编码器部分
        self.encoder1  = DoubleConv(n_channels, 64)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2  = DoubleConv(64, 128)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3  = DoubleConv(128, 256)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4  = DoubleConv(256, 512)
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2)

        # 中间部分
        self.middle  = DoubleConv(512, 1024)

        # 解码器部分
        self.upconv4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4  = DoubleConv(1024, 512)
        self.upconv3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3  = DoubleConv(512, 256)
        self.upconv2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2  = DoubleConv(256, 128)
        self.upconv1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1  = DoubleConv(128, 64)

        # 输出层
        self.out_conv  = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器前向传播
        enc1 = self.encoder1(x) 
        enc2 = self.encoder2(self.pool1(enc1)) 
        enc3 = self.encoder3(self.pool2(enc2)) 
        enc4 = self.encoder4(self.pool3(enc3)) 

        # 中间部分
        middle = self.middle(self.pool4(enc4)) 

        # 解码器前向传播
        dec4 = self.upconv4(middle) 
        dec4 = torch.cat([dec4,  enc4], dim=1)
        dec4 = self.decoder4(dec4) 
        dec3 = self.upconv3(dec4) 
        dec3 = torch.cat([dec3,  enc3], dim=1)
        dec3 = self.decoder3(dec3) 
        dec2 = self.upconv2(dec3) 
        dec2 = torch.cat([dec2,  enc2], dim=1)
        dec2 = self.decoder2(dec2) 
        dec1 = self.upconv1(dec2) 
        dec1 = torch.cat([dec1,  enc1], dim=1)
        dec1 = self.decoder1(dec1) 

        # 输出层
        out = self.out_conv(dec1) 
        return out

if __name__ == "__main__":
    device = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")
    model = UNet().to(device)
    print(summary(model, (1, 256, 256)))