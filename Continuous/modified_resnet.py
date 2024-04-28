import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models


class CustomCNN(nn.Module):
    def __init__(self, input_dims):
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dims[0], 8, 4, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1)
        self.conv6 = nn.Conv2d(128, 256, 3, stride=1)
        
        self.apply(self._weights_init)
        
        self.encoder_output_dims = self.calculate_conv_output_dims(input_dims)
        
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        
    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        dims = self.conv4(dims)
        dims = self.conv5(dims)
        dims = self.conv6(dims)
        return int(np.prod(dims.size()))
    
    def forward(self, state):
        state_out = F.relu(self.conv1(state))
        state_out = F.relu(self.conv2(state_out))
        state_out = F.relu(self.conv3(state_out))
        state_out = F.relu(self.conv4(state_out))
        state_out = F.relu(self.conv5(state_out))
        state_out = F.relu(self.conv6(state_out))
        return state_out


class ModifiedResNet34(nn.Module):
    def __init__(self, n_groups, input_dims):
        super().__init__()
        self.resnet = models.resnet50()
        
        # input layer
        self.resnet.conv1 = nn.Conv2d(input_dims[0], self.resnet.bn1.num_features, kernel_size=(7, 7),
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
        
        self.apply(self._weights_init)
        
        self.resnet.avgpool = nn.Sequential()
        self.resnet.fc = nn.Sequential()
        
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))
        
        self.encoder_output_dims = self.calculate_encoder_output_dims(input_dims)
        
    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
    
    def calculate_encoder_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.resnet(state)
        return dims.size()
    
    def forward(self, x):
        return self.resnet(x) 