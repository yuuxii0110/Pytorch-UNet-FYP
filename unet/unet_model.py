""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        fac = 2

        self.inc = DoubleConv(n_channels, 64//fac)
        self.down1 = Down(64//fac, 128//fac)
        self.down2 = Down(128//fac, 256//fac)
        self.down3 = Down(256//fac, 512//fac)
        self.down4 = Down(512//fac, 1024//fac)
        factor = 2 if bilinear else 1
        self.down5 = Down(1024//fac, 2048//fac//factor)
        self.up1 = Up(2048//fac, 1024//fac // factor, bilinear)
        self.up2 = Up(1024//fac, 512//fac // factor, bilinear)
        self.up3 = Up(512//fac, 256//fac // factor, bilinear)
        self.up4 = Up(256//fac, 128//fac // factor, bilinear)
        self.up5 = Up(128//fac, 64//fac, bilinear)
        if bilinear:
            self.outc = OutConv((512)//fac, n_classes)
        else:
            self.outc = OutConv((64+128+256+512)//fac, n_classes)

        self.up_s8 = customUp(8)
        self.up_s4 = customUp(4)
        self.up_s2 = customUp(2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)

        x = self.up2(x, x4)
        s1 = self.up_s8(x)

        x = self.up3(x, x3)
        s2 = self.up_s4(x)

        x = self.up4(x, x2)
        s3 = self.up_s2(x)

        x = self.up5(x, x1)

        x = torch.cat([x,s3,s2,s1],dim=1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)