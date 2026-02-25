import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        KERNEL_SIZE = 3
        PADDING = KERNEL_SIZE // 2  # to maintain spatial dimensions
        
        # NOTE: each layer learns its own parameters, thus conv1 and conv2 will be learned differently

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=KERNEL_SIZE, padding=PADDING)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # shortcut connection with batch normalization for proper residual learning
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(residual) # residual block
        out = self.relu2(out)
        
        return out

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        KERNEL_SIZE = 4 # 2 may cause checkerboard artifacts due to no overlap, 6 may be too large that it can include too much
        STRIDE = 2 # do not change, need to upsample by factor of 2
        PADDING = (KERNEL_SIZE - STRIDE) // 2

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING) 
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.out_channels = out_channels

        BASE_CHANNELS = 48
        self.inc = DoubleConv(in_channels, BASE_CHANNELS)

        self.down1 = Down(BASE_CHANNELS, BASE_CHANNELS * 2)
        self.down2 = Down(BASE_CHANNELS * 2, BASE_CHANNELS * 4)
        self.down3 = Down(BASE_CHANNELS * 4, BASE_CHANNELS * 8)
        self.down4 = Down(BASE_CHANNELS * 8, BASE_CHANNELS * 16)

        self.up1 = Up(BASE_CHANNELS * 16, BASE_CHANNELS * 8)
        self.up2 = Up(BASE_CHANNELS * 8, BASE_CHANNELS * 4)
        self.up3 = Up(BASE_CHANNELS * 4, BASE_CHANNELS * 2)
        self.up4 = Up(BASE_CHANNELS * 2, BASE_CHANNELS)

        self.outc = OutConv(BASE_CHANNELS, out_channels)
        
    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # bottleneck

        x = self.up1(x5, x4) # skip connection: x4 from encoder
        x = self.up2(x, x3) # skip connection: x3 from encoder
        x = self.up3(x, x2) # skip connection: x2 from encoder
        x = self.up4(x, x1) # skip connection: x1 from encoder

        out = self.outc(x)
        
        return out
