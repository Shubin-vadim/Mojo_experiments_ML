# Mojo_experiments_ML

👋🏻 Welcome to the Mojo and Python ML experiments Project repository!
This project compares the performance of Mojo and Python on MNIST digit classification and house price prediction tasks.

## Quick Links

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Results](#results)

---

## Project Overview
The main goal of this project is to compare the performance between Python and Mojo.

## Quick-start
1. Installing dependencies for Python
```bash
pip3 install -r python-requirements.txt
```
2. Be sure that you have Mojo installed
```bash
mojo --version
```
3. Run
  - Mojo
```bash
mojo -I . exps/housing.mojo
mojo -I . exps/mnist_CNN.mojo
mojo -I . exps/mnist_LNN.mojo
```
  - Python
```bash
python3 exps/housing.py
python3 exps/mnist.py
python3 exps/mnist_LNN.mojo
```
## Results
<img src="https://github.com/Shubin-vadim/Mojo_experiments_ML/blob/master/results/HousingPrediction.png" width="400" alt="HousingPrediction" />
<img src="https://github.com/Shubin-vadim/Mojo_experiments_ML/blob/master/results/MNIST.png" width="400" alt="MNIST" />
