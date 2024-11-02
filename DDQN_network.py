import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQN_network(nn.Module):
   def __init__(self, frames, actions):
       super(DDQN_network, self).__init__()

       # Arhitecture of my NN
       self.conv1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
       self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
       self.fc = nn.Linear(20736, 512)
       self.fc1 = nn.Linear(512, actions)
       self.fc2 = nn.Linear(512, 1)

       # Initializing weights and biases
       torch.nn.init.xavier_uniform_(self.conv1.weight)
       self.conv1.bias.data.fill_(0.01)
       torch.nn.init.xavier_uniform_(self.conv2.weight)
       self.conv2.bias.data.fill_(0.01)
       torch.nn.init.xavier_uniform_(self.fc.weight)
       self.fc.bias.data.fill_(0.01)
       torch.nn.init.xavier_uniform_(self.fc1.weight)
       self.fc1.bias.data.fill_(0.01)
       torch.nn.init.xavier_uniform_(self.fc2.weight)
       self.fc2.bias.data.fill_(0.01)

       # Defining sequential layers
       self.layers = nn.Sequential(
           self.conv1,
           nn.ReLU(),
           self.conv2,
           nn.ReLU(),
           nn.Flatten(),
           self.fc,
           nn.ReLU(),
           self.fc1,
           self.fc2
       )

   def forward(self, x):
       if not isinstance(x, torch.Tensor):
           x = torch.tensor(x, dtype=torch.float32)
       x = F.relu(self.conv1(x))
       x = F.relu(self.conv2(x))
       x = x.view(x.size(0), -1)
       x = F.relu(self.fc(x))
       Q_values = self.fc2(x) + (self.fc1(x) - self.fc1(x).mean(dim=1, keepdim=True))
       return Q_values
