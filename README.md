# ML-MNIST

This repository contains two Jupyter Notebooks, `NN_Scratch.ipynb` and `NN_libraries.ipynb`, implementing neural networks for digit recognition on the MNIST dataset. The notebooks are located in the `src` directory.

## Data

Data is taken from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

## NN_Scratch.ipynb
This notebook implements a simple neural network from scratch using NumPy. The neural network has an input layer with 784 nodes, two hidden layers with 512 and 64 nodes, and an output layer with 10 nodes corresponding to digits 0-9. The training is done using the MNIST dataset, and the accuracy is reported during training.

### Requirements
- NumPy
- Pandas
- Matplotlib
- SciPy

### Usage
1. Open the notebook using Jupyter Notebook or Jupyter Lab.
2. Run the cells sequentially to train the neural network and see the training progress.

## NN_libraries.ipynb
This notebook utilizes TensorFlow and Keras to build a neural network for digit recognition on the MNIST dataset. The neural network consists of a flatten layer, followed by three dense layers with ReLU activation, and an output layer with softmax activation. The notebook demonstrates training and evaluating the model using TensorFlow and Keras.

### Requirements
- TensorFlow
- Pandas
- NumPy

### Usage
1. Open the notebook using Jupyter Notebook or Jupyter Lab.
2. Run the cells sequentially to train the neural network and evaluate its performance on the MNIST test set.

Feel free to explore and modify the code to experiment with different architectures, hyperparameters, and training strategies.