import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

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

class LNN(nn.Module):
    def __init__(self):
        super(LNN, self).__init__()
        
        # fully connected layer, output 10 classes
        self.linear_block = nn.Sequential(
            nn.Linear(28 * 28 * 1, 400),
            nn.ReLU(),

            nn.Linear(400, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 10),

        )

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        output = self.linear_block(x)
        return output

def plot_loss(losses):
    plt.figure()
    plt.plot(range(len(losses)), losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    num_epochs = 20
    batch_size = 8
    learning_rate = 2e-3

    # Load data
    train_data = MNIST("./examples/data/mnist_test_small.csv")

    # Visualize data
    num = 0
    plt.imshow(np.array(train_data[num][0]).squeeze())
    plt.title("%i" % train_data[num][1])
    plt.show()

    # Batchwise data loader
    loaders = {
        "train": DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=1
        ),
    }

    device = torch.device("cpu")
    model = LNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    total_step = len(loaders["train"])
    start = time.time()
    
    all_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(loaders["train"]):
            b_x = images.to(device)
            b_y = labels.to(device)
            output = model(b_x)
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_epoch_loss = epoch_loss / total_step
        all_losses.append(average_epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {average_epoch_loss:.6f}")

    print(f"Training time: {time.time() - start:.2f} seconds")

    # Plot the loss graph
    plot_loss(all_losses)

    # Export to ONNX
    export_onnx = os.environ.get("export_onnx", 0)
    if export_onnx == "1":
        dummy_input = torch.randn(1, 1, 28, 28)
        torch.onnx.export(model, dummy_input, "./examples/data/mnist_torch.onnx", verbose=True)
