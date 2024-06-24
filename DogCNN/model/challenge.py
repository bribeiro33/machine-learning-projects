"""
EECS 445 - Introduction to Machine Learning
Fall 2022 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config

class Challenge(nn.Module):
    def __init__(self):
        """Define the architecture, i.e. what layers our network contains. 
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions."""
        super().__init__()

        self.conv1=Conv2d(64,64,kernel_size=(5,5), stride=(2,2))
        self.conv2=Conv2d(64,64,kernel_size=(5,5),stride=(2,2))
        self.conv3=Conv2d(64,64,kernel_size=(5,5),stride=(2,2))
        self.conv4=Conv2d(64,64,kernel_size=(5,5),stride=(2,2))

        self.fc1 = Linear(6195, 128)
        self.fc2 = Linear(1428, 128)
        self.fc3 = Linear(288, 128)
        self.fc4 = Linear(2560, 128)
        self.fc5 = Linear(2560, 1024)
        self.fc6 = Linear(1024,512)
        self.fc7 = Linear(512,256)
        self.fc8 = Linear(256,128)
        self.fc9 = Linear(258,1)
        
        self.drop = nn.Dropout()
        ##

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)

        nn.init.normal_(self.fc__2.weight, 0.0, 1 / sqrt(256))
        nn.init.normal_(self.fc__3.weight, 0.0, 1 / sqrt(128))
        nn.init.normal_(self.fc__4.weight, 0.0, 1 / sqrt(64))
        nn.init.normal_(self.fc__5.weight, 0.0, 1 / sqrt(32))

        nn.init.constant(self.fc__2.bias, 0.0)
        nn.init.constant(self.fc__3.bias, 0.0)
        nn.init.constant(self.fc__4.bias, 0.0)
        nn.init.constant(self.fc__5.bias, 0.0)
        ##

    def forward(self, x):
        """This function defines the forward propagation for a batch of input examples, by
            successively passing output of the previous layer as the input into the next layer (after applying
            activation functions), and returning the final output as a torch.Tensor object

            You may optionally use the x.shape variables below to resize/view the size of
            the input matrix at different points of the forward pass"""
        N, C, H, W = x.shape

        x = F.relu(self.conv1(x))
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = self.drop(x)
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.drop(self.relu(self.fc3(x)))
        x = self.drop(self.relu(self.fc4(x)))
        x = self.drop(self.relu(self.fc5(x)))
        x = self.drop(self.relu(self.fc6(x)))
        x = self.drop(self.relu(self.fc7(x)))
        x = self.fc8(x)

        return x
