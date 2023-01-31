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

    self.linear1 = nn.Sequential(
      nn.Flatten(),
      nn.Linear(25, 64),
      nn.BatchNorm1d(64),
      nn.ELU()
    )

    self.linear2 = nn.Sequential(
      nn.Linear(64, 64),
      nn.BatchNorm1d(64),
      nn.ELU()
    )

    self.linear3 = nn.Sequential(
      nn.Linear(64, 64),
      nn.BatchNorm1d(64),
      nn.Tanh()
    )

    self.linear4 = nn.Sequential(
      nn.Linear(64, 64),
      nn.BatchNorm1d(64),
      nn.Tanh()
    )

    self.linear5 = nn.Sequential(
      nn.Linear(64, 64),
      nn.BatchNorm1d(64),
      nn.Tanh()
    )

    self.linear6 = nn.Sequential(
      nn.Linear(64, 64),
    )

    # self.linear7 = nn.Sequential(
    #   nn.Linear(64, 64),
    #   nn.BatchNorm1d(64),
    #   nn.ReLU()
    # )

    self.linear8 = nn.Sequential(
      nn.Linear(64, 4)
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

    x = self.linear1(patches)
    x = self.linear2(x)
    x = self.linear3(x)
    x = self.linear4(x)
    x = self.linear5(x)
    x = self.linear6(x)
    # x = self.linear7(x)
    x = self.linear8(x)


    return x