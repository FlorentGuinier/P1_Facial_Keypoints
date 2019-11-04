import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # -- conv1 (input = 1x100x100) --
        self.conv1 = nn.Conv2d( 1,  16, 5) # 1x100x100 -> 16x96x96
        self.pool = nn.MaxPool2d(2, 2)     #  16x96x96 -> 16x48x48
        self.bnconv1 = nn.BatchNorm2d(16)
        
        # -- conv2 --
        self.conv2 = nn.Conv2d(16,  32, 3) #  16x48x48 -> 32x46x46
        #self.pool = nn.MaxPool2d(2, 2)    #  32x46x46 -> 32x23x23
        self.bnconv2 = nn.BatchNorm2d(32)
        
        # -- conv3 --
        self.conv3 = nn.Conv2d(32,  64, 3) #  32x23x23 -> 64x21x21
        #self.pool = nn.MaxPool2d(2, 2)    #  64x21x21 -> 64x10x10
        self.bnconv3 = nn.BatchNorm2d(64)
        
        # -- conv4 --
        self.conv4 = nn.Conv2d(64, 128, 3) #  64x10x10 -> 128x8x8
        #self.pool = nn.MaxPool2d(2, 2)    #  128x8x8  -> 128x4x4
        self.bnconv4 = nn.BatchNorm2d(128)
        
        # -- flatten --                    # 128x4x4   -> 2048
                                           
        # -- fc1 --
        self.fc1 = nn.Linear(128*4*4, 1024)# 2048      -> 1024
        self.bncfc1 = nn.BatchNorm1d(1024)
        
        # -- fc2 --
        self.fc2 = nn.Linear(1024, 256)    # 2048      -> 256
        self.bncfc2 = nn.BatchNorm1d(256)
        
        # -- fc3 (output = 136 ie 68*2)--
        self.fc3 = nn.Linear(256, 68*2)   # 256       -> 136
       
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bnconv1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bnconv2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bnconv3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bnconv4(x)
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.bncfc1(x)
        x = F.relu(self.fc2(x))
        x = self.bncfc2(x)
        x = self.fc3(x)
        return x
