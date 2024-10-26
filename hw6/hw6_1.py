# ========= Import Useful Libraries ========= #

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import tqdm

# define transformations to apply to the dataset
# the dataset must be transformed into a Tensor
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST('./data', train=True, download=True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# ========= Define Models ========= #
# Define the Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
      super().__init__()
      self.layer1 = nn.Linear(28*28, 200)
      self.layer2 = nn.Linear(200, 200)
      self.layer3 = nn.Linear(200, 10)
      self.activation = nn.ReLU()

    def forward(self, x):
      vz = self.layer1(x)
      z = self.activation(vz)
      vf = self.layer2(z)
      f = self.activation(vf)
      vy = self.layer3(f)
      y = self.activation(vy)

      return y

torch.manual_seed(17) # setting seed for reproducibility
# define model hyperparameters
model = NeuralNetwork()
size = len(train_dataset)
loss_fn = nn.CrossEntropyLoss()
batch_size = 64
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
n_epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# ========= Now Train the first Model ========= #
# Training pipeline
for epoch in range(n_epochs):
  print(f"Epoch {epoch+1}\n-------------------------------")
  epoch_loss = 0.0
  accuracy = 0.0
  for batch, (X, y) in enumerate(train_loader):
    model.train()
    X, y = X.to(device), y.to(device)
    y_pred = model(X.view(X.shape[0], -1))
    loss = loss_fn(y_pred, y)
    epoch_loss += loss.item()
    _, predicted = torch.max(y_pred.data, 1)
    accuracy += (predicted == y).sum().item()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

  epoch_loss /= len(train_loader)
  accuracy = accuracy / len(train_dataset)
  print(f"Epoch loss: {epoch_loss}")
  print(f"Accuracy: {accuracy}")
