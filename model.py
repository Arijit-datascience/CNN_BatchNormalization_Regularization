#!/usr/bin/env python
# coding: utf-8

# # Python program to create a CNN model with Normalization to detect MNIST digits

# ### Import the required libraries

# In[1]:

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


# # Model Architecture

# ## Class for Model with Batch Normalization

# In[2]:

# Set the dropout probability to be used in the network
DROPOUT_VALUE = 0.04


# In[3]:

# In Batch Normalization, each channel in a layer is picked up across the batch on images and normalized.
# Here, we use nn.BatchNorm2d(num_channels) after each Conv2d layer to implement Batch Normalization.
class BatchNormalization(nn.Module):
    def __init__(self):
        super(BatchNormalization, self).__init__()

        # Input Block
        self.conv1 = nn.Sequential(
            #input - RF:1x1, Channel Size: 28x28; Output RF: 3x3, Channel Size: 26x26
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(8) 
        )
        
        # CONVOLUTION BLOCK 1
        self.conv2 = nn.Sequential(
            #input - RF:3x3, Channel Size: 26x26; Output RF: 5x5, Channel Size: 24x24
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16) 
        )

        # TRANSITION BLOCK 1
        self.trans = nn.Sequential(
            #input - RF:5x5, Channel Size: 24x24; Output RF: 5x5, Channel Size: 24x24
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            #input - RF:5x5, Channel Size: 24x24; Output RF: 6x6, Channel Size: 12x12
            nn.MaxPool2d(2, 2) 
        )
        
        # CONVOLUTION BLOCK 2
        self.conv3 = nn.Sequential(
            #input - RF:6x6, Channel Size: 12x12; Output RF: 10x10, Channel Size: 10x10
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            #input - RF:10x10, Channel Size: 10x10; Output RF: 14x14, Channel Size: 8x8
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(DROPOUT_VALUE)
        )

        # CONVOLUTION BLOCK 3
        self.conv4 = nn.Sequential(
            #input - RF:14x14, Channel Size: 8x8; Output RF: 18x18, Channel Size: 6x6
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(DROPOUT_VALUE)
        )

        # CONVOLUTION BLOCK 4
        self.conv5 = nn.Sequential(
            #input - RF:18x18, Channel Size: 6x6; Output RF: 22x22, Channel Size: 6x6
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            #input - RF:22x22, Channel Size: 6x6; Output RF: 32x32, Channel Size: 1x1
            nn.AvgPool2d(kernel_size=6) 
        )

        self.conv6 = nn.Sequential(
            #input - RF:32x32, Channel Size: 1x1; Output RF: 32x32, Channel Size: 1x1
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)        
        x = self.conv6(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


# ## Class for Model with Group Normalization

# In[4]:

# In Group Normalization, we forms groups of channels within each layer and normalize the element values in these channel groups.
# Here, we use nn.GroupNorm(num_groups,num_channels) after each Conv2d layer to implement Group Normalization.
class GroupNormalization(nn.Module):
    def __init__(self):
        super(GroupNormalization, self).__init__()

        # Input Block
        self.conv1 = nn.Sequential(
            #input - RF:1x1, Channel Size: 28x28; Output RF: 3x3, Channel Size: 26x26
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.GroupNorm(num_groups=2, num_channels=8)
        )
        
        # CONVOLUTION BLOCK 1
        self.conv2 = nn.Sequential(
            #input - RF:3x3, Channel Size: 26x26; Output RF: 5x5, Channel Size: 24x24
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=4, num_channels=16)
        )

        # TRANSITION BLOCK 1
        self.trans = nn.Sequential(
            #input - RF:5x5, Channel Size: 24x24; Output RF: 5x5, Channel Size: 24x24
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            #input - RF:5x5, Channel Size: 24x24; Output RF: 6x6, Channel Size: 12x12
            nn.MaxPool2d(2, 2)
        )
        
        # CONVOLUTION BLOCK 2
        self.conv3 = nn.Sequential(
            #input - RF:6x6, Channel Size: 12x12; Output RF: 10x10, Channel Size: 10x10 
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(num_groups=3, num_channels=12),
            #input - RF:10x10, Channel Size: 10x10; Output RF: 14x14, Channel Size: 8x8
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.Dropout(DROPOUT_VALUE)
        )

        # CONVOLUTION BLOCK 3
        self.conv4 = nn.Sequential(
            #input - RF:14x14, Channel Size: 8x8; Output RF: 18x18, Channel Size: 6x6
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.Dropout(DROPOUT_VALUE)
        )

        # CONVOLUTION BLOCK 4
        self.conv5 = nn.Sequential(
            #input - RF:18x18, Channel Size: 6x6; Output RF: 22x22, Channel Size: 6x6
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(num_groups=4, num_channels=16),
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            #input - RF:22x22, Channel Size: 6x6; Output RF: 32x32, Channel Size: 1x1
            nn.AvgPool2d(kernel_size=6)
        )

        self.conv6 = nn.Sequential(
            #input - RF:32x32, Channel Size: 1x1; Output RF: 32x32, Channel Size: 1x1
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)        
        x = self.conv6(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


# ## Class for Model with Layer Normalization

# In[5]:

# Layer Normalization is a special case of Group Normalization wherein we selct the group count as 1.
# This will result in the entire channels in the layer to be normalized at once.
# Here, we use nn.GroupNorm(1,num_channels) after each Conv2d layer to implement Layer Normalization.
class LayerNormalization(nn.Module):
    def __init__(self):
        super(LayerNormalization, self).__init__()

        # Input Block
        self.conv1 = nn.Sequential(
            #input - RF:1x1, Channel Size: 28x28; Output RF: 3x3, Channel Size: 26x26
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), 
            nn.ReLU(),
            # Layer Normalization
            nn.GroupNorm(1,8)
        )
        
        # CONVOLUTION BLOCK 1
        self.conv2 = nn.Sequential(
            #input - RF:3x3, Channel Size: 26x26; Output RF: 5x5, Channel Size: 24x24
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            # Layer Normalization
            nn.GroupNorm(1,16)
        )

        # TRANSITION BLOCK 1
        self.trans = nn.Sequential(
            #input - RF:5x5, Channel Size: 24x24; Output RF: 5x5, Channel Size: 24x24
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            #input - RF:5x5, Channel Size: 24x24; Output RF: 6x6, Channel Size: 12x12
            nn.MaxPool2d(2, 2)
        )
        
        # CONVOLUTION BLOCK 2
        self.conv3 = nn.Sequential(
            #input - RF:6x6, Channel Size: 12x12; Output RF: 10x10, Channel Size: 10x10 
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            # Layer Normalization
            nn.GroupNorm(1,12),
            #input - RF:10x10, Channel Size: 10x10; Output RF: 14x14, Channel Size: 8x8
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            # Layer Normalization
            nn.GroupNorm(1,16),
            nn.Dropout(DROPOUT_VALUE)
        )
        
        # CONVOLUTION BLOCK 3
        self.conv4 = nn.Sequential(
            #input - RF:14x14, Channel Size: 8x8; Output RF: 18x18, Channel Size: 6x6
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            # Layer Normalization
            nn.GroupNorm(1,16),
            nn.Dropout(DROPOUT_VALUE)
        )

        # CONVOLUTION BLOCK 4
        self.conv5 = nn.Sequential(
            #input - RF:18x18, Channel Size: 6x6; Output RF: 22x22, Channel Size: 6x6
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            # Layer Normalization
            nn.GroupNorm(1,16),
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            #input - RF:22x22, Channel Size: 6x6; Output RF: 32x32, Channel Size: 1x1
            nn.AvgPool2d(kernel_size=6)
        )

        self.conv6 = nn.Sequential(
            #input - RF:32x32, Channel Size: 1x1; Output RF: 32x32, Channel Size: 1x1
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)        
        x = self.conv6(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


# ## build_model: This function accepts type of Normalization as input and returns the model object with corresponding normalization applied
# ##              Raises an exception in case of an incorrect Normalization type is given in input
# In[18]:


def build_model(model_type):
    # Check if the input is Batch Normalization
    if model_type == 'BN':
        model = BatchNormalization()
    # Check if the input is Group Normalization
    elif model_type == 'GN':
        model = GroupNormalization()
    # Check if the input is Layer Normalization
    elif model_type == 'LN':
        model = LayerNormalization()
    # Raise an exception in case the input is invalid
    else:
        raise Exception("Invalid Normalization: model_type should be 'BN' or 'GN' or 'LN'")
    
    # Return the model object to the calling program
    return model

