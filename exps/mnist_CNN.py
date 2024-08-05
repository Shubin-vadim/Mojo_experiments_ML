import time
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset

def plot_loss(losses):
    plt.figure()
    plt.plot(range(len(losses)), losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss over MNIST epochs")
    plt.legend()
    plt.show()


class MNIST(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.int64)
        self.images = torch.tensor(
            data.iloc[:, 1:].values, dtype=torch.float32
        ).reshape(-1, 1, 28, 28)

        # Normalize data
        self.images = self.images / 255.0

        self.dataset = TensorDataset(self.images, self.labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=5, 
                stride=1, 
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # fully connected layer, output 10 classes
        
        self.fc1 = nn.Linear(32 * 7 * 7, 120)
        
        self.act_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        self.act_fc2 = nn.ReLU()

        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.block1(x)
        print(x.shape)
        x = self.block2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        x = self.act_fc1(self.fc1(x))
        x = self.act_fc2(self.fc2(x))
        output = self.out(x)
        return output


if __name__ == "__main__":
    num_epochs = 20
    batch_size = 8
    learning_rate = 2e-3

    # Load data
    train_data = MNIST("./examples/data/mnist_test_small.csv")

    # Batchwise data loader
    loaders = {
        "train": DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=1
        )
    }

    device = torch.device("cpu")
    cnn = CNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    
    # Train the model
    cnn.train()
    total_step = len(loaders["train"])
    start = time.time()

    all_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(loaders["train"]):
            b_x = Variable(images)
            b_y = Variable(labels)

            output = cnn(b_x)
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(
                "Epoch [{}/{}],\t Step [{}/{}],\t Loss: {:.6f}".format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()
                )
            )

        average_epoch_loss = epoch_loss / total_step
        all_losses.append(average_epoch_loss)


    plot_loss(all_losses)
    print(f"Training time: {time.time() - start:.2f} seconds")