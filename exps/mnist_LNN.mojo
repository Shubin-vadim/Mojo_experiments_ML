from time.time import now

import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP, dtype
from basalt.utils.datasets import MNIST
from basalt.utils.dataloader import DataLoader
from basalt.autograd.attributes import AttributeVector, Attribute


def plot_image(data: Tensor, num: Int):
    from python.python import Python, PythonObject

    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")

    var pyimage: PythonObject = np.empty((28, 28), np.float64)
    for m in range(28):
        for n in range(28):
            pyimage.itemset((m, n), data[num * 28 * 28 + m * 28 + n])

    plt.imshow(pyimage)
    plt.show()


fn create_CNN(batch_size: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 1, 28, 28))

    var x_reshape = g.op(
        OP.RESHAPE,
        x,
        attributes=AttributeVector(
            Attribute(
                "shape",
                TensorShape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]),
            )
        ),
    )

    var fc1 = nn.Linear(g, x, n_outputs=400)
    var act_fc1 = nn.ReLU(g, fc1)

    var fc2 = nn.Linear(g, act_fc1, n_outputs=256)
    var act_fc2 = nn.ReLU(g, fc2)

    var fc3 = nn.Linear(g, act_fc2, n_outputs=128)
    var act_fc3 = nn.ReLU(g, fc3)

    var fc4 = nn.Linear(g, act_fc3, n_outputs=64)
    var act_fc4 = nn.ReLU(g, fc4)

    var fc5 = nn.Linear(g, act_fc4, n_outputs=32)
    var act_fc5 = nn.ReLU(g, fc5)

    var out = nn.Linear(g, act_fc5, n_outputs=10)
    
    g.out(out)

    var y_true = g.input(TensorShape(batch_size, 10))
    var loss = nn.CrossEntropyLoss(g, out, y_true)
    # var loss = nn.MSELoss(g, out, y_true)
    g.loss(loss)

    return g ^


fn main():
    alias num_epochs = 20
    alias batch_size = 8
    alias learning_rate = 2e-3

    alias graph = create_CNN(batch_size)

    # try: graph.render("operator")
    # except: print("Could not render graph")

    var model = nn.Model[graph]()
    var optim = nn.optim.Adam[graph](Reference(model.parameters), lr=learning_rate)

    print("Loading data ...")
    var train_data: MNIST
    try:
        train_data = MNIST(file_path="./examples/data/mnist_test_small.csv")
        plot_image(train_data.data, 1)
    except e:
        print("Could not load data")
        print(e)
        return

    var training_loader = DataLoader(
        data=train_data.data, labels=train_data.labels, batch_size=batch_size
    )

    print("Training started/")
    var start = now()

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

    print("Training finished: ", (now() - start) / 1e9, "seconds")

    model.print_perf_metrics("ms", True)
