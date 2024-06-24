"""
EECS 445 - Introduction to Machine Learning
Fall 2022 - Project 2
Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import target
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Target(nn.Module):
    def __init__(self):
        """Define the architecture, i.e. what layers our network contains. 
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions."""
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (5,5), stride = (2,2), padding = (2, 2))
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = (5,5), stride = (2,2), padding = (2, 2))
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 8, kernel_size = (5,5), stride = (2,2), padding = (2, 2))
        self.fc_1 = nn.Linear(in_features = 32, out_features = 2)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""        
        torch.manual_seed(42)

        ## weights and biases for convolution layers (3)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## weights and biases for fully connected layer (1)
        f_in = self.fc_1.weight.size(1)
        nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(f_in))
        nn.init.constant_(self.fc_1.bias, 0.0) 

    def forward(self, x):
        """This function defines the forward propagation for a batch of input examples, by
            successively passing output of the previous layer as the input into the next layer (after applying
            activation functions), and returning the final output as a torch.Tensor object

            You may optionally use the x.shape variables below to resize/view the size of
            the input matrix at different points of the forward pass"""        
        N, C, H, W = x.shape

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc_1(x)

        return x
