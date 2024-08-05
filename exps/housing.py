import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from matplotlib import pyplot as plt
import time

def plot_loss(losses):
    plt.figure()
    plt.plot(range(len(losses)), losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss over housing prediction epochs")
    plt.legend()
    plt.show()

class BostonHousing(Dataset):
    def __init__(self, data: pd.DataFrame):

        self.data = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
        self.target = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).view(
            -1, 1
        )

        # Normalize data
        self.data = (self.data - self.data.mean(dim=0)) / self.data.std(dim=0)

        # Create dataset
        self.dataset = TensorDataset(self.data, self.target)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("./examples/data/housing.csv")

    train_data = BostonHousing(df)

    # Train Parameters
    batch_size = 32
    num_epochs = 500
    learning_rate = 0.01

    # Batchwise data loader
    loaders = {
        "train": DataLoader(
            train_data, batch_size=batch_size, shuffle=False, num_workers=1
        )
    }

    device = torch.device("cpu")
    model = LinearRegression(train_data.data.shape[1])
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    total_step = len(loaders["train"])
    start = time.time()

    all_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        for batch_data, batch_labels in loaders["train"]:
            start_batch = time.time()

            # Forward pass
            outputs = model(batch_data)
            loss = loss_func(outputs, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        print(
            f"Epoch [{epoch + 1}/{num_epochs}],\t Avg loss per epoch:"
            f" {epoch_loss / num_batches}"
        )

        average_epoch_loss = epoch_loss / total_step
        all_losses.append(average_epoch_loss)

    plot_loss(all_losses)
    print(all_losses)
    print(f"Training time: {time.time() - start:.2f} seconds")