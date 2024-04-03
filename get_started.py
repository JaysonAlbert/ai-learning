import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST("data", True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST("data", False, download=True, transform=ToTensor())

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)

for X, y in test_dataloader:
    print(f"Shape of x: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


class ForwardModel(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = ForwardModel().to(device)

print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * (len(X))
            print(f"loss: {loss}, [{current}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        test_loss = test_loss + loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test error: \n Accuracy: {100*correct:>0.1f}%, average loss: {test_loss}\n")


epochs = 5

for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------- ")
    train(train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
    test(test_dataloader, model=model, loss_fn=loss_fn)

print("Done!")
