# -*- coding: utf-8 -*-
"""MNIST Digit Recognition Batch Normalization and Regularization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U3fdaRyFHBVyh2aRczVS_RPY4aVZUmR_
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import StepLR,OneCycleLR
import numpy as np
import os

!pip install torchsummary
from torchsummary import summary

"""# Reading MNIST data"""

# Train Phase transformations
train_transforms = transforms.Compose([
                                       transforms.RandomRotation((-7.0, 7.0), fill=(1,)),                
                                       transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.40, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])

# Test Phase transformations
test_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])

train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True,transform=test_transforms)

n_train = len(train_data)
n_test = len(test_data)

print('Number of training samples: {0}'.format(n_train))
print('Number of test samples: {0}'.format(n_test))

"""# Model Architecture

## Batch Normalization
"""

DROPOUT_VALUE = 0.04

class BatchNormalization(nn.Module):
    def __init__(self):
        super(BatchNormalization, self).__init__()

        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        
        # CONVOLUTION BLOCK 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # TRANSITION BLOCK 1
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2)
        )
        
        # CONVOLUTION BLOCK 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(DROPOUT_VALUE)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(DROPOUT_VALUE)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )

        self.conv6 = nn.Sequential(
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

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

model = BatchNormalization().to(device)
summary(model, input_size=(1, 28, 28))

"""## Group Normalization"""

class GroupNormalization(nn.Module):
    def __init__(self):
        super(GroupNormalization, self).__init__()

        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.GroupNorm(num_groups=2, num_channels=8)
        )
        
        # CONVOLUTION BLOCK 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=4, num_channels=16)
        )

        # TRANSITION BLOCK 1
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2)
        )
        
        # CONVOLUTION BLOCK 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(num_groups=3, num_channels=12),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.Dropout(DROPOUT_VALUE)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.Dropout(DROPOUT_VALUE)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(num_groups=4, num_channels=16),
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )

        self.conv6 = nn.Sequential(
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

model = GroupNormalization().to(device)
summary(model, input_size=(1, 28, 28))

"""## Layer Normalization"""

class LayerNormalization(nn.Module):
    def __init__(self):
        super(LayerNormalization, self).__init__()

        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.LayerNorm((8,26,26))
        )
        
        # CONVOLUTION BLOCK 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm((16, 24, 24))
        )

        # TRANSITION BLOCK 1
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2)
        )
        
        # CONVOLUTION BLOCK 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.LayerNorm((12, 10, 10)),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.LayerNorm((16, 8, 8)),
            nn.Dropout(DROPOUT_VALUE)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.LayerNorm((16, 6, 6)),
            nn.Dropout(DROPOUT_VALUE)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.LayerNorm((16, 6, 6)),
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )

        self.conv6 = nn.Sequential(
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

"""# Model Summary"""

model = LayerNormalization().to(device)
summary(model, input_size=(1, 28, 28))

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

"""# Train Loop"""

def train(model, device, train_loader, optimizer, scheduler, epoch, L1, l1_factor):
    model.train()
    epoch_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        if L1:
          L1_loss = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
          reg_loss = 0 
          for param in model.parameters():
            zero_vector = torch.rand_like(param) * 0
            reg_loss += L1_loss(param,zero_vector)
          loss += l1_factor * reg_loss

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'Train set: Average loss: {loss.item():.4f}, Accuracy: {100. * correct/len(train_loader.dataset):.2f}')
    train_loss = epoch_loss / len(train_loader)
    train_acc=100.*correct/len(train_loader.dataset)
    return train_loss, train_acc

"""# Test Loop"""

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_pred = torch.LongTensor()
    target_pred = torch.LongTensor()
    target_data = torch.LongTensor()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred_cpu = output.cpu().data.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_pred = torch.cat((test_pred, pred_cpu), dim=0)
            target_pred = torch.cat((target_pred, target.cpu()), dim=0)
            target_data = torch.cat((target_data, data.cpu()), dim=0)


    test_loss /= len(test_loader.dataset)
    test_acc = 100.*correct/len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.3f}, Accuracy: {100. * correct/len(test_loader.dataset):.2f}')
    return test_loss, test_acc, test_pred, target_pred, target_data

"""## Function of plotting misclassified images"""

def misclassification(predictions, targets, data, xtitle):
  pred = predictions.view(-1)
  target = targets.view(-1)

  index = 0
  misclassified_image = []

  for label, predict in zip(target, pred):
    if label != predict:
      misclassified_image.append(index)
    index += 1

  plt.figure(figsize=(10,5))
  plt.suptitle(xtitle + ' - Misclassified Images');

  for plot_index, bad_index in enumerate(misclassified_image[0:10]):
    p = plt.subplot(2, 5, plot_index+1)
    p.imshow(data[bad_index].reshape(28,28), cmap='binary')
    p.axis('off')
    p.set_title(f'Pred:{pred[bad_index]}, Actual:{target[bad_index]}')

"""# Running the model"""

def main(EPOCH, model, device, train_loader, test_loader, optimizer, scheduler, L1, l1_factor):
  train_loss_values = []
  test_loss_values = []
  train_acc_values = []
  test_acc_values = []

  for epoch in range(1, EPOCH + 1):
      print('\nEpoch {} : '.format(epoch))
      # train the model
      train_loss, train_acc = train(model, device, train_loader, optimizer, scheduler, epoch, L1, l1_factor)
      test_loss, test_acc, test_pred, target_pred, target_data = test(model, device, test_loader)

      train_loss_values.append(train_loss)
      test_loss_values.append(test_loss)

      train_acc_values.append(train_acc)
      test_acc_values.append(test_acc)

  return train_loss_values, test_loss_values, train_acc_values, test_acc_values, test_pred, target_pred, target_data

if __name__=='__main__':

  EPOCH = 20
  batch_size = 64
  l1_factor = 0.0001

  cuda = torch.cuda.is_available()
  device = torch.device("cuda" if cuda else "cpu")

  seed_everything(1)

  kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)
  
  model = BatchNormalization().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.8)
  scheduler = OneCycleLR(optimizer, max_lr=0.015,epochs=20,steps_per_epoch=len(train_loader))
  
  print("------------------------------------------")
  print("Batch Normalization with L1 regularization")
  print("------------------------------------------")
  BN_train_loss, BN_test_loss, BN_train_acc, BN_test_acc, BN_test_pred, BN_target_pred, BN_target_data = main(EPOCH, model, device, train_loader, test_loader, optimizer, scheduler, True, l1_factor)

  model = GroupNormalization().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.8)
  scheduler = OneCycleLR(optimizer, max_lr=0.015,epochs=20,steps_per_epoch=len(train_loader))

  print("------------------------------------------")
  print("Group Normalization")
  print("------------------------------------------")
  GN_train_loss, GN_test_loss, GN_train_acc, GN_test_acc, GN_test_pred, GN_target_pred, GN_target_data = main(EPOCH, model, device, train_loader, test_loader, optimizer, scheduler, False, l1_factor)

  model = LayerNormalization().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.8)
  scheduler = OneCycleLR(optimizer, max_lr=0.015,epochs=20,steps_per_epoch=len(train_loader))

  print("------------------------------------------")
  print("Layer Normalization")
  print("------------------------------------------")
  LN_train_loss, LN_test_loss, LN_train_acc, LN_test_acc, LN_test_pred, LN_target_pred, LN_target_data = main(EPOCH, model, device, train_loader, test_loader, optimizer, scheduler, False, l1_factor)

"""# Plotting the train and test loss across each epoch"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='whitegrid')

# Increase the plot size and font size.
sns.set(font_scale=1)
plt.rcParams["figure.figsize"] = (25,6)

# Plot the learning curve.
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(np.array(BN_train_loss), 'red', label="BN + L1 Training Loss")
ax1.plot(np.array(GN_train_loss), 'green', label="GN Training Loss")
ax1.plot(np.array(LN_train_loss), 'yellow', label="LN Training Loss")
ax1.plot(np.array(BN_test_loss), 'blue', label="BN + L1 Test Loss")
ax1.plot(np.array(GN_test_loss), 'black', label="GN Test Loss")
ax1.plot(np.array(LN_test_loss), 'purple', label="LN Test Loss")

# Label the plot.
ax1.set_title("Training & Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_ylim(0,0.2)
ax1.legend()

ax2.plot(np.array(BN_train_acc), 'red', label="BN + L1 Training Accuracy")
ax2.plot(np.array(GN_train_acc), 'green', label="GN Training Accuracy")
ax2.plot(np.array(LN_train_acc), 'yellow', label="LN Training Accuracy")
ax2.plot(np.array(BN_test_acc), 'blue', label="BN + L1 Test Accuracy")
ax2.plot(np.array(GN_test_acc), 'black', label="GN Test Accuracy")
ax2.plot(np.array(LN_test_acc), 'purple', label="LN Test Accuracy")

# Label the plot.
ax2.set_title("Training & Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_ylim(95,100)
ax2.legend()

plt.show()

"""## Batch Normalization + L1 Norm Misclassified Images"""

misclassification(BN_test_pred, BN_target_pred, BN_target_data, 'Batch Normalization + L1')

"""## Group Normalization Misclassified Images"""

misclassification(GN_test_pred, GN_target_pred, GN_target_data, 'Group Normalization')

"""## Layer Normalization Misclassified Images"""

misclassification(LN_test_pred, LN_target_pred, LN_target_data, 'Layer Normalization')

