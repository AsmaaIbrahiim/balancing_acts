import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channel, feature_d):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channel,feature_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self.CNNBlock(feature_d, feature_d*2, 4, 2, 1),
            self.CNNBlock(feature_d*2, feature_d*4, 4, 2, 1),
            self.CNNBlock(feature_d*4, feature_d*8, 4, 2, 1),
            nn.Conv2d(feature_d*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )
    
    def CNNBlock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.disc(x)