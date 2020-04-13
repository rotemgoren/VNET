import torch
import torchvision
from torchsummary import summary
from torch import nn
import torch.utils.data as data
import numpy as np

from torchsummary import summary
import torch.nn.functional as F


NUM_SLICES=3
class Dataset(data.Dataset):
    def __init__(self, x,y,step = int(NUM_SLICES/2)):



        
            self.x = torch.cat([torch.FloatTensor(x[:,:,i:i+NUM_SLICES]).unsqueeze(0).unsqueeze(1) for i in range(0,int(x.shape[2]/NUM_SLICES)*NUM_SLICES-2,step)])
            self.y = torch.cat([torch.FloatTensor(y[:,:,i:i+NUM_SLICES]).unsqueeze(0).unsqueeze(1) for i in range(0,int(x.shape[2]/NUM_SLICES)*NUM_SLICES-2,step)])


    def __getitem__(self, index):
        return self.x[index,:,:,:,:], self.y[index,:,:,:,:]

    def __len__(self):
        return self.x.size(0)


class DownConvBlock(nn.Module):
    def __init__(self,in_channels, mid_channels,out_channels,kernel_size=3,num_layers=1,device='cpu'):
        super(DownConvBlock, self).__init__()
        self.device=device
        self.num_layers=num_layers
        self.conv1=nn.Conv3d(in_channels=in_channels,out_channels=mid_channels,padding=1,kernel_size=kernel_size,stride=1).to(self.device).float()
        self.conv2 = []
        for _ in range(1,self.num_layers):
            self.conv2.append(nn.Conv3d(in_channels=mid_channels,out_channels=mid_channels,padding=1,kernel_size=kernel_size,stride=1).to(self.device).float())
        self.conv3 = nn.Conv3d(in_channels=mid_channels, out_channels=out_channels,padding=1, kernel_size=kernel_size, stride=1).to(self.device).float()

        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1))
        self.bn1 = nn.BatchNorm3d(mid_channels).to(self.device).float()
        self.bn2 = nn.BatchNorm3d(out_channels).to(self.device).float()

        self.prelu=nn.ReLU().to(self.device).float()

    def forward(self,x):
        x1=self.conv1(x)
        x1=self.bn1(x1)
        x1=self.prelu(x1)

        x2=x1
        for i in range(1,self.num_layers):
            x2 = self.conv2[i-1](x2)
            x2 = self.bn1(x2)
            x2 = self.prelu(x2)

        del x1
        torch.cuda.empty_cache()
        x3 = x2 + x
        x4 = self.conv3(x3)
        x4 = self.bn2(x4)
        x4 = self.prelu(x4)
        x4 = self.pool4(x4)

        return x3,x4

class UpConvBlock(nn.Module):
    def __init__(self,in_channels, mid_channels, out_channels,kernel_size=3,num_layers=1,device='cpu'):
        super(UpConvBlock, self).__init__()
        self.num_layers=num_layers
        self.device=device
        self.conv1=nn.Conv3d(in_channels,mid_channels,padding=1,kernel_size=kernel_size,stride=1).to(self.device).float()
        self.conv2 = []
        for _ in range(1,self.num_layers):
            self.conv2.append(nn.Conv3d(mid_channels,mid_channels,padding=1,kernel_size=kernel_size,stride=1).to(self.device).float())
        self.conv3 = nn.ConvTranspose3d(mid_channels, out_channels, kernel_size=(2,2,1),stride=(2,2,1)).to(self.device).float()
        self.bn1 = nn.BatchNorm3d(mid_channels).to(self.device).float()
        self.bn2 = nn.BatchNorm3d(out_channels).to(self.device).float()
        self.prelu = nn.ReLU().to(self.device).float()

    def forward(self,x):
        x1=self.conv1(x)
        x1=self.bn1(x1)
        x1=self.prelu(x1)

        x2=x1
        for i in range(1,self.num_layers):
            x2 = self.conv2[i-1](x2)
            x2 = self.bn1(x2)
            x2 = self.prelu(x2)

        x3 = x2 + x
        x4 = self.conv3(x3)
        x4 = self.bn2(x4)
        x4 = self.prelu(x4)

        return x4



class VNet(nn.Module):
    def __init__(self,device):
        super(VNet,self).__init__()
        self.down_conv1 = DownConvBlock(in_channels=1,mid_channels=16,out_channels=32,num_layers=1,device=device)
        self.down_conv2 = DownConvBlock(in_channels=32,mid_channels=32, out_channels=64, num_layers=3,device=device)
        self.down_conv3 = DownConvBlock(in_channels=64,mid_channels=64, out_channels=128, num_layers=3,device=device)
        self.down_conv4 = DownConvBlock(in_channels=128,mid_channels=128, out_channels=256, num_layers=3,device=device)

        self.up_conv5 = UpConvBlock(in_channels=256,mid_channels=256, out_channels=128, num_layers=3,device=device)
        self.up_conv6 = UpConvBlock(in_channels=128+128,mid_channels=128+128, out_channels=64, num_layers=3,device=device)
        self.up_conv7 = UpConvBlock(in_channels=64+64,mid_channels=64+64, out_channels=32, num_layers=3,device=device)
        self.up_conv8 = UpConvBlock(in_channels=32 + 32, mid_channels=32 + 32, out_channels=32, num_layers=3,device=device)

        self.conv9=nn.Conv3d(in_channels=32+16,out_channels=1,padding=1,kernel_size=3,stride=1).to(device).float()
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        x1,x = self.down_conv1(x)
        x3,x = self.down_conv2(x)
        x5,x = self.down_conv3(x)
        x7,x = self.down_conv4(x)

        # x8, x = self.down_conv4_(x)
        #
        # x = self.up_conv5_(x)
        # x = torch.cat((x8,x),dim=1)

        x = self.up_conv5(x)
        x = torch.cat((x7,x),dim=1)

        x= self.up_conv6(x)
        x = torch.cat((x5,x),dim=1)

        x = self.up_conv7(x)
        x = torch.cat((x3,x),dim=1)

        x = self.up_conv8(x)
        x = torch.cat((x1, x), dim=1)

        x= self.conv9(x)
        x = self.sigmoid(x)

        return x


#model = UNetWithMobileNet().cuda()
#summary(model,input_size=(3,224,224))
#print(model)

#inp = torch.rand((2, 3, 512, 512)).cuda()
#out = model(inp)
