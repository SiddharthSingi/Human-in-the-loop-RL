import torch
import torch.nn as nn
import torch.nn.functional as F

# If patch size is 3
# class NeuralNet(nn.Module):
    
#   def __init__(self):
#     super().__init__()

#     self.tconv1 = nn.Sequential(
#       nn.ConvTranspose2d(4, 4, 4, 2),
#       nn.BatchNorm2d(4),
#       nn.ReLU()
#     )
#     self.conv2 = nn.Sequential(
#       nn.Conv2d(4, 1, 1),
#       nn.BatchNorm2d(1),
#       nn.ReLU()
#     )
#     self.conv3 = nn.Conv2d(1, 1, 5)
#     self.linear = nn.Sequential(
#       nn.Flatten(),
#       nn.Linear(16, 4)
#     )

#     self.posn_linear = nn.Sequential(
#       nn.Linear(2, 8),
#       nn.ReLU(),
#       nn.Linear(8, 8),
#       nn.ReLU(),
#       nn.Linear(8,4)
#     )

#     self.mixed_linear = nn.Sequential(
#       nn.Linear(4, 4),
#       nn.ReLU(),
#       nn.Linear(4, 4)
#     )

#   def forward(self, state):
#     '''
#     params:

#       patch: Shape Nx4x3x3
#       posns: Shape: Nx2
#         where N is batch size
#     '''
#     patches = state[0]
#     posns = state[1]
    
#     x1 = self.tconv1(patches) # Shape: Nx4x8x8
#     x1 = self.conv2(x1) # Shape: Nx1x8x8
#     x1 = self.conv3(x1) # Shape: Nx1x4x4
#     x1 = self.linear(x1) # Shape: Nx1x4

#     x2 = self.posn_linear(posns) # Shape: Nx1x4
#     x = x1+x2
    
#     return x

# If patch size is 5
class NeuralNet(nn.Module):
    
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(4, 1, 1),
      nn.BatchNorm2d(1),
      nn.ReLU()
    )

    self.linear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(25, 12),
      nn.ReLU(),
      nn.Linear(12, 4)
    )

    self.posn_linear = nn.Sequential(
      nn.Linear(2, 8),
      nn.ReLU(),
      nn.Linear(8,4)
    )

    self.mixed_linear = nn.Sequential(
      nn.Linear(4, 4),
      nn.ReLU(),
      nn.Linear(4, 4)
    )

  def forward(self, state):
    '''
    params:
      patch: Shape Nx4x5x5
      posns: Shape: Nx2
        where N is batch size
    '''
    patches = state[0]
    posns = state[1]
    
    x1 = self.conv1(patches) # Shape: Nx1x5x5
    x1 = self.linear(x1) # Shape: Nx4
    x2 = self.posn_linear(posns) # Shape: Nx4
    x = x1+x2
    x = self.mixed_linear(x) # Shape: Nx4
    return x