import torch
import torch.nn as nn

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None

class MinkowskiDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, D=3):
        super().__init__()
        KERNEL_SIZE = 3
        
        # first convolution: in_channels -> out_channels
        self.conv1 = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=KERNEL_SIZE, stride=1, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(out_channels)
        self.relu1 = ME.MinkowskiReLU()
        
        # second convolution: out_channels -> out_channels
        self.conv2 = ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=KERNEL_SIZE, stride=1, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)
        
        # shortcut connection for residual learning
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, stride=1, dimension=D),
                ME.MinkowskiBatchNorm(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.relu2 = ME.MinkowskiReLU()

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(residual) # residual connection
        out = self.relu2(out)
        
        return out

class MinkowskiDown(nn.Module):
    def __init__(self, in_channels, out_channels, D=3):
        super().__init__()
        self.pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D)
        self.double_conv = MinkowskiDoubleConv(in_channels, out_channels, D)

    def forward(self, x):
        x = self.pool(x)
        return self.double_conv(x)

class MinkowskiUp(nn.Module):
    def __init__(self, in_channels, out_channels, D=3):
        super().__init__()
        KERNEL_SIZE = 4 # larger kernel to avoid checkerboard artifacts
        STRIDE = 2
        PADDING = (KERNEL_SIZE - STRIDE) // 2
        
        # transposed convolution for upsampling
        self.up = ME.MinkowskiConvolutionTranspose(
            in_channels, in_channels // 2, 
            kernel_size=KERNEL_SIZE, stride=STRIDE, 
            dimension=D
        )
        self.double_conv = MinkowskiDoubleConv(in_channels, out_channels, D)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = ME.cat(x2, x1) # skip connection concatenation
        return self.double_conv(x)

class MinkowskiOutConv(nn.Module):
    def __init__(self, in_channels, out_channels, D=3):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, stride=1, dimension=D)
        self.sigmoid = ME.MinkowskiSigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class MinkowskiUNet(nn.Module):
    def __init__(self, in_channels, out_channels, D=3): # D: Dimension
        super(MinkowskiUNet, self).__init__()
        
        BASE_CHANNELS = 64 
        
        # initial convolution
        self.inc = MinkowskiDoubleConv(in_channels, BASE_CHANNELS, D)
        
        # encoder (downsampling path)
        self.down1 = MinkowskiDown(BASE_CHANNELS, BASE_CHANNELS * 2, D)
        self.down2 = MinkowskiDown(BASE_CHANNELS * 2, BASE_CHANNELS * 4, D)
        self.down3 = MinkowskiDown(BASE_CHANNELS * 4, BASE_CHANNELS * 8, D)
        # self.down4 = MinkowskiDown(BASE_CHANNELS * 8, BASE_CHANNELS * 16, D)

        # decoder (upsampling path)
        # self.up1 = MinkowskiUp(BASE_CHANNELS * 16, BASE_CHANNELS * 8, D)
        self.up1 = MinkowskiUp(BASE_CHANNELS * 8, BASE_CHANNELS * 4, D)
        self.up2 = MinkowskiUp(BASE_CHANNELS * 4, BASE_CHANNELS * 2, D)
        self.up3 = MinkowskiUp(BASE_CHANNELS * 2, BASE_CHANNELS, D)
        # self.up4 = MinkowskiUp(BASE_CHANNELS * 2, BASE_CHANNELS, D)
        
        # output convolution
        self.outc = MinkowskiOutConv(BASE_CHANNELS, out_channels, D)
        
    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)

        # decoder with skip connections
        # x = self.up1(x5, x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)

        # Output
        out = self.outc(x)

        return out
