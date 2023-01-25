import torch
import torch.nn as nn


def up_layer(x,in_ch,out_ch):
    return nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2, padding=0)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm3d(out_ch)
        self.relu = nn.PReLU() #nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

class Fuse_model(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=16, surv_nc=2):
        super(Fuse_model, self).__init__()

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        self.surv_out_dim = surv_nc

        # define the network
        self.encode = nn.Sequential(
            Block(self.in_dim,self.out_dim),
            nn.MaxPool3d(kernel_size=2, stride=2), # shrink image to half
            Block(self.out_dim,self.out_dim*2),
            nn.MaxPool3d(kernel_size=2, stride=2),
            Block(self.out_dim*2,self.out_dim*4),
            nn.MaxPool3d(kernel_size=2, stride=2),
            Block(self.out_dim*4,self.out_dim*8),
            nn.MaxPool3d(kernel_size=2, stride=2),
            Block(self.out_dim*8,self.out_dim*16),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.out_dim*16, out_channels=self.out_dim*8, kernel_size=2, stride=2, padding=0),
            Block(self.out_dim*8,self.out_dim*8),
            nn.ConvTranspose3d(in_channels=self.out_dim * 8, out_channels=self.out_dim * 4, kernel_size=2, stride=2, padding=0),
            Block(self.out_dim * 4, self.out_dim * 4),
            nn.ConvTranspose3d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 2, kernel_size=2, stride=2, padding=0),
            Block(self.out_dim * 2, self.out_dim * 2),
            nn.ConvTranspose3d(in_channels=self.out_dim * 2, out_channels=self.out_dim, kernel_size=2, stride=2, padding=0),
            Block(self.out_dim, self.out_dim),
            nn.Conv3d(in_channels=self.out_dim, out_channels=self.final_out_dim, kernel_size=3, stride=1, padding=1),
        )

        # the survival outcome net
        self.surv_net = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            Block(self.out_dim*16,self.out_dim*16),

            # Fully convolutional kernel size is same of my image
            nn.Conv3d(in_channels=self.out_dim*16, out_channels=512, kernel_size=3),
            nn.BatchNorm3d(512),
            nn.PReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm3d(512),
            nn.PReLU(),
            nn.Conv3d(in_channels=512, out_channels=self.surv_out_dim, kernel_size=1),
            nn.Flatten(),
        )

    def forward(self, x): #define network
        encoded = self.encode(x)
        out = self.decode(encoded)
        phi = self.surv_net(encoded)
        return out, phi

class Unet_model(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=16):
        super(Unet_model, self).__init__()

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        # define the network
        self.encode1 = Block(self.in_dim,self.out_dim)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) # shrink image to half
        self.encode2 = Block(self.out_dim,self.out_dim*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encode3 = Block(self.out_dim*2,self.out_dim*4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encode4 = Block(self.out_dim*4,self.out_dim*8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encode5 = Block(self.out_dim*8,self.out_dim*16)

        self.up1 = nn.ConvTranspose3d(in_channels=self.out_dim*16, out_channels=self.out_dim*8, kernel_size=2, stride=2, padding=0)
        self.decode1 = Block(self.out_dim*16,self.out_dim*8)
        self.up2 = nn.ConvTranspose3d(in_channels=self.out_dim * 8, out_channels=self.out_dim * 4, kernel_size=2, stride=2, padding=0)
        self.decode2 = Block(self.out_dim * 8, self.out_dim * 4)
        self.up3 = nn.ConvTranspose3d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 2, kernel_size=2, stride=2, padding=0)
        self.decode3 = Block(self.out_dim * 4, self.out_dim * 2)
        self.up4 = nn.ConvTranspose3d(in_channels=self.out_dim * 2, out_channels=self.out_dim, kernel_size=2, stride=2, padding=0)
        self.decode4 = Block(self.out_dim, self.out_dim)
        self.decode5 = nn.Conv3d(in_channels=self.out_dim, out_channels=self.final_out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x): #define network
        x = self.encode1(x)
        x = self.pool1(x)
        x1 = self.encode2(x)
        x = self.pool2(x1)
        x2 = self.encode3(x)
        x = self.pool3(x2)
        x3 = self.encode4(x)
        x = self.pool4(x3)
        x = self.encode5(x)

        x = self.up1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decode1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decode2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decode3(x)
        x = self.up4(x)
        x = self.decode4(x)
        x = self.decode5(x)
        return x