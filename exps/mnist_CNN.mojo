from time.time import now

import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP, dtype
from basalt.utils.datasets import MNIST
from basalt.utils.dataloader import DataLoader
from basalt.autograd.attributes import AttributeVector, Attribute
from python import Python
from collections import List

def plot_loss(losses: List[Float32], num_epochs: Int) -> NoneType:

    var np = Python.import_module("numpy")
    var plt = Python.import_module("matplotlib.pyplot")
    var x = np.zeros(num_epochs)
    var y_mojo = np.zeros(num_epochs)

    for i in range(len(losses)):
        x[i] = i + 1
        y_mojo[i] = losses[i]

    plt.figure()
    plt.plot(x, y_mojo, label="Training Loss Mojo")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss over MNIST epochs")
    plt.xticks(x)  
    plt.legend()
    plt.show()


fn create_CNN(batch_size: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 1, 28, 28))

    var conv1 = nn.Conv2d(g, x, out_channels=16, kernel_size=5, padding=2)
    var act_conv1 = nn.ReLU(g, conv1)
    var max_pool1 = nn.MaxPool2d(g, act_conv1, kernel_size=2)
    
    var conv2 = nn.Conv2d(g, max_pool1, out_channels=32, kernel_size=5, padding=2)
    var act_conv2 = nn.ReLU(g, conv2)
    var max_pool2 = nn.MaxPool2d(g, act_conv2, kernel_size=2)
    
    var x_reshape = g.op(
        OP.RESHAPE,
        max_pool2,
        attributes=AttributeVector(
            Attribute(
                "shape",
                TensorShape(max_pool2.shape[0], max_pool2.shape[1] * max_pool2.shape[2] * max_pool2.shape[3]),
            )
        ),
    )

    var fc1 = nn.Linear(g, x_reshape, n_outputs=120)

    var act_fc1 = nn.ReLU(g, fc1)

    var fc2 = nn.Linear(g, act_fc1, n_outputs=84)

    var act_fc2 = nn.ReLU(g, fc2)

    var out = nn.Linear(g, act_fc2, n_outputs=10)
    
    g.out(out)

    var y_true = g.input(TensorShape(batch_size, 10))
    var loss = nn.CrossEntropyLoss(g, out, y_true)

    g.loss(loss)

    return g ^


def main():
    
    alias num_epochs = 20
    alias batch_size = 8
    alias learning_rate = 2e-3

    alias graph = create_CNN(batch_size)


    var model = nn.Model[graph]()
    var optim = nn.optim.Adam[graph](Reference(model.parameters), lr=learning_rate)

    print("Loading data ...")
    var train_data: MNIST
    try:
        train_data = MNIST(file_path="./examples/data/mnist_test_small.csv")
    except e:
        print("Could not load data")
        print(e)
        return

    var training_loader = DataLoader(
        data=train_data.data, labels=train_data.labels, batch_size=batch_size
    )

    print("Training started/")
    var start = now()

    var losses = List[Float32]()

    for epoch in range(num_epochs):
        var num_batches: Int = 0
        var epoch_loss: Float32 = 0.0
        var epoch_start = now()
        for batch in training_loader:
            # [ONE HOT ENCODING!]
            var labels_one_hot = Tensor[dtype](batch.labels.dim(0), 10)
            for bb in range(batch.labels.dim(0)):
                labels_one_hot[int((bb * 10 + batch.labels[bb]))] = 1.0
            # Forward pass
            var loss = model.forward(batch.data, labels_one_hot)

            # Backward pass
            optim.zero_grad()
            model.backward()
            optim.step()

            epoch_loss += loss[0]
            num_batches += 1

            print(
                "Epoch [",
                epoch + 1,
                "/",
                num_epochs,
                "],\t Step [",
                num_batches,
                "/",
                train_data.data.dim(0) // batch_size,
                "],\t Loss:",
                epoch_loss / num_batches,
            )

        print("Epoch time: ", (now() - epoch_start) / 1e9, "seconds")

        average_epoch = epoch_loss / num_batches
        losses.append(average_epoch)

    print("Training finished: ", (now() - start) / 1e9, "seconds")

    model.print_perf_metrics("ms", True)

    plot_loss(losses, num_epochs)