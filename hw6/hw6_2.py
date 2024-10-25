# ========= Import Useful Libraries ========= #

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import tqdm
from itertools import product # to test different hyperparameters

class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5, stride=2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.activation = nn.ReLU()
        
  
        self.batchnorm_1 = nn.BatchNorm2d(20)
        self.batchnorm_2 = nn.BatchNorm2d(20)
        
        self.linear1 = nn.Linear(20 * 5 * 5, 250)
        self.linear2 = nn.Linear(250, 10)
        self.dropout = nn.Dropout(0.2) # set p = 0 to remove dropout
        self.batchnorm_fc = nn.BatchNorm1d(250)

    def forward(self, x):
        x = self.activation(self.batchnorm_1(self.conv1(x)))  # (20, 25, 25)
        x = self.activation(self.batchnorm_2(self.conv2(x)))  # (20, 11, 11)
        x = self.maxpool(x)  # (20, 5, 5)
        x = self.flatten(x)  # (20 * 5 * 5)
        x = self.batchnorm_fc(self.linear1(x))  # (250, )
        x = self.activation(x)
        x = self.dropout(x)
        
        # Final linear layer
        x = self.linear2(x)  # (10, )
        return x




torch.manual_seed(17) # setting seed for reproducibility
device = "cuda" if torch.cuda.is_available() else "cpu"

# parameters to test to try to improve results
param_grid = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [16, 32, 64],
    'epochs': [5, 10, 20]
}


def train(params):
  model = ConvNeuralNetwork()
  model.to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
  transform = transforms.Compose([transforms.ToTensor()])
  train_dataset = MNIST('./data', train=True, download=True, transform = transform)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
  size = len(train_dataset)
  loss_fn = nn.CrossEntropyLoss()

  # ========= Now Train the first Model ========= #
  # Training pipeline
  for epoch in range(params['epochs']):
    print(f"Epoch {epoch+1}\n-------------------------------")
    epoch_loss = 0.0
    accuracy = 0.0
    for batch, (X, y) in enumerate(train_loader):
      model.train()
      X, y = X.to(device), y.to(device)
      y_pred = model(X)
      loss = loss_fn(y_pred, y)
      epoch_loss += loss.item()
      _, predicted = torch.max(y_pred.data, 1)
      accuracy += (predicted == y).sum().item()
      
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if batch % 100 == 0:
              loss, current = loss.item(), (batch + 1) * len(X)
              #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    epoch_loss /= len(train_loader)
    accuracy = accuracy / len(train_dataset)
    print(f"Epoch loss: {epoch_loss}")
    print(f"Accuracy: {accuracy}")

  return accuracy


# train the model and find the best hyperparameters
best_accuracy = 0
best_params = None
for params in product(*param_grid.values()): # test all possible combinations of hyperparameters
    params = dict(zip(param_grid.keys(), params))
    print("Training with hyperparameters:", params)
    accuracy = train(params)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
    
    
print("Best hyperparameters:", best_params)
print("Best accuracy:", best_accuracy)
