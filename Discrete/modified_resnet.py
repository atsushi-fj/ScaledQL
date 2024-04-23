
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import models


class ModifiedResNet50(nn.Module):
    def __init__(self, n_groups):
        super().__init__()
        self.resnet = models.resnet50()
        
        # input layer
        self.resnet.conv1 = nn.Conv2d(4, self.resnet.bn1.num_features, kernel_size=(7, 7),
                                      stride=(2, 2), padding=(3, 3), bias=False)
        
        # BatchNorm -> GroupNorm
        self.resnet.bn1 = nn.GroupNorm(n_groups, self.resnet.conv1.out_channels)
        for i in range(len(self.resnet.layer1)):
            self.resnet.layer1[i].bn1 = nn.GroupNorm(n_groups, self.resnet.layer1[i].conv1.out_channels)
            self.resnet.layer1[i].bn2 = nn.GroupNorm(n_groups, self.resnet.layer1[i].conv2.out_channels)
            self.resnet.layer1[i].bn3 = nn.GroupNorm(n_groups, self.resnet.layer1[i].conv3.out_channels)
            if i == 0: self.resnet.layer1[i].downsample[1] = nn.GroupNorm(n_groups, self.resnet.layer1[i].downsample[0].out_channels)
        
        for i in range(len(self.resnet.layer2)):
            self.resnet.layer2[i].bn1 = nn.GroupNorm(n_groups, self.resnet.layer2[i].conv1.out_channels)
            self.resnet.layer2[i].bn2 = nn.GroupNorm(n_groups, self.resnet.layer2[i].conv2.out_channels)
            self.resnet.layer2[i].bn3 = nn.GroupNorm(n_groups, self.resnet.layer2[i].conv3.out_channels)
            if i == 0: self.resnet.layer2[i].downsample[1] = nn.GroupNorm(n_groups, self.resnet.layer2[i].downsample[0].out_channels)
        
        for i in range(len(self.resnet.layer3)):
            self.resnet.layer3[i].bn1 = nn.GroupNorm(n_groups, self.resnet.layer3[i].conv1.out_channels)
            self.resnet.layer3[i].bn2 = nn.GroupNorm(n_groups, self.resnet.layer3[i].conv2.out_channels)
            self.resnet.layer3[i].bn3 = nn.GroupNorm(n_groups, self.resnet.layer3[i].conv3.out_channels)
            if i == 0: self.resnet.layer3[i].downsample[1] = nn.GroupNorm(n_groups, self.resnet.layer3[i].downsample[0].out_channels)
        
        for i in range(len(self.resnet.layer4)):
            self.resnet.layer4[i].bn1 = nn.GroupNorm(n_groups, self.resnet.layer4[i].conv1.out_channels)
            self.resnet.layer4[i].bn2 = nn.GroupNorm(n_groups, self.resnet.layer4[i].conv2.out_channels)
            self.resnet.layer4[i].bn3 = nn.GroupNorm(n_groups, self.resnet.layer4[i].conv3.out_channels)
            if i == 0: self.resnet.layer4[i].downsample[1] = nn.GroupNorm(n_groups, self.resnet.layer4[i].downsample[0].out_channels)
        
        self.resnet.avgpool = nn.Sequential()
        self.resnet.fc = nn.Sequential()
        
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))
        
    def forward(self, x):
        return self.resnet(x)
    
    
class ModifiedResNet18(nn.Module):
    def __init__(self, n_groups):
        super().__init__()
        self.resnet = models.resnet18()
        
        # input layer
        self.resnet.conv1 = nn.Conv2d(4, self.resnet.bn1.num_features, kernel_size=(7, 7),
                                      stride=(2, 2), padding=(3, 3), bias=False)
        
        # BatchNorm -> GroupNorm
        self.resnet.bn1 = nn.GroupNorm(n_groups, self.resnet.conv1.out_channels)
        for i in range(len(self.resnet.layer1)):
            self.resnet.layer1[i].bn1 = nn.GroupNorm(n_groups, self.resnet.layer1[i].conv1.out_channels)
            self.resnet.layer1[i].bn2 = nn.GroupNorm(n_groups, self.resnet.layer1[i].conv2.out_channels)
        
        for i in range(len(self.resnet.layer2)):
            self.resnet.layer2[i].bn1 = nn.GroupNorm(n_groups, self.resnet.layer2[i].conv1.out_channels)
            self.resnet.layer2[i].bn2 = nn.GroupNorm(n_groups, self.resnet.layer2[i].conv2.out_channels)
            if i == 0: self.resnet.layer2[i].downsample[1] = nn.GroupNorm(n_groups, self.resnet.layer2[i].downsample[0].out_channels)
        
        for i in range(len(self.resnet.layer3)):
            self.resnet.layer3[i].bn1 = nn.GroupNorm(n_groups, self.resnet.layer3[i].conv1.out_channels)
            self.resnet.layer3[i].bn2 = nn.GroupNorm(n_groups, self.resnet.layer3[i].conv2.out_channels)
            if i == 0: self.resnet.layer3[i].downsample[1] = nn.GroupNorm(n_groups, self.resnet.layer3[i].downsample[0].out_channels)
        
        for i in range(len(self.resnet.layer4)):
            self.resnet.layer4[i].bn1 = nn.GroupNorm(n_groups, self.resnet.layer4[i].conv1.out_channels)
            self.resnet.layer4[i].bn2 = nn.GroupNorm(n_groups, self.resnet.layer4[i].conv2.out_channels)
            if i == 0: self.resnet.layer4[i].downsample[1] = nn.GroupNorm(n_groups, self.resnet.layer4[i].downsample[0].out_channels)
        
        self.resnet.avgpool = nn.Sequential()
        self.resnet.fc = nn.Sequential()
        
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))
        
    def forward(self, x):
        return self.resnet(x)
        
  