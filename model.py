# 33125（star）
import torch
import torch.nn  as nn
from timm.layers import DropPath, trunc_normal_
# 定义 ConvBN 模块，用于卷积和批量归一化操作
from torchsummary import summary


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv',  torch.nn.Conv2d(in_planes,  out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn',  torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight,  1)
            torch.nn.init.constant_(self.bn.bias,  0)


# 定义 Block 模块，包含多个卷积和激活函数操作
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv  = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2  = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act  = nn.ReLU6()
        self.drop_path  = nn.Identity() if drop_path <= 0. else DropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1)  * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


# 定义 StarNet 模块，用于图片恢复
class StarNet(nn.Module):# 1262
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # 茎层，用于对输入图片进行初步处理
        self.stem  = nn.Sequential(ConvBN(1, self.in_channel,  kernel_size=3, stride=1, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0,  drop_path_rate, sum(depths))]
        # 构建不同阶段的模块
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            if i_layer > 0:
                down_sampler = ConvBN(self.in_channel,  embed_dim, 3, 2, 1)
            else:
                down_sampler = nn.Identity()
            self.in_channel  = embed_dim
            blocks = [Block(self.in_channel,  mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler,  *blocks))
        # 上采样层，用于将特征图恢复到原始尺寸
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256,  128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU6(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU6(),

            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight,  std=.02)
            if hasattr(m, 'bias') and m.bias  is not None:
                nn.init.constant_(m.bias,  0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias,  0)
            nn.init.constant_(m.weight,  1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.upsampling(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StarNet().to(device)
    print(summary(model, (1, 256, 256)))
    print(device)



