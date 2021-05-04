import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channel_noise, img_channel, feature_g):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            self.CNNBlock(channel_noise,feature_g*16, 4, 1,0),
            self.CNNBlock(feature_g*16, feature_g*8, 4, 2, 1),
            self.CNNBlock(feature_g*8, feature_g*4, 4, 2, 1),
            self.CNNBlock(feature_g*4, feature_g*2, 4, 2, 1),
            nn.ConvTranspose2d(feature_g*2,img_channel,kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def CNNBlock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self,x):
        return self.gen(x)