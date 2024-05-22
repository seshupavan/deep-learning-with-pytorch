"""
The whole idea of this section is to implement different deep learning optimization techniques in Pytorch and
to make things simple i'll write most used functions here and then use them in the main notebooks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.datasets


def create_dataset():
    """This creates a dataset of inputs, outputs and converts it into torch tensors"""
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=600, noise=.2)
    # visualize the data
    plt.figure(figsize=(6, 4))
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = torch.from_numpy(train_X.T).float().T
    train_Y = torch.from_numpy(train_Y.reshape((1, train_Y.shape[0]))).float().T
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=999)

    # Print details about the dataset
    print(f"Number of training examples: {X_train.shape[0]}")
    print(f"Number of testing examples: {X_test.shape[0]}")
    print(f"Shape of train set: {X_train.shape}")
    print(f"Shape of test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def calculate_accuracy(preds, actuals):
    """This function used to calculate accuracy"""
    with torch.no_grad():
        rounded_preds = torch.round(preds)
        num_correct = torch.sum(rounded_preds == actuals)
        accuracy = num_correct / len(preds)
    return accuracy


def create_classifier(num_inputs, num_outputs):
    """Creates a sequential model with 2 hidden layers"""

    class SimpleClassifier(nn.Module):
        def __init__(self, num_inputs, num_outputs):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(num_inputs, 10),
                nn.ReLU(),
                nn.Linear(10, 30),
                nn.ReLU(),
                nn.Linear(30, 10),
                nn.ReLU(),
                nn.Linear(10, num_outputs),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.layers(x)

    model = SimpleClassifier(num_inputs, num_outputs)
    return model


def plot_decision_boundary(model, X, y):
    """Creates decision boundary to help visualize the dots after training the model"""
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Convert to torch tensors
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    # Predict the function value for the whole grid
    Z = model(grid)
    # Convert back to numpy
    Z = Z.detach().numpy()
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def plot_metrics(train_losses, test_losses, train_accs, test_accs):
    """Creates plots to visualize losses and accuracies for both training and test sets"""
    plt.figure(figsize=(12, 6))

    # Subplot for losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['train_losses', 'test_losses'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and Testing Losses')

    # Subplot for accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.legend(['train_accs', 'test_accs'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training and Testing Accuracies')

    plt.tight_layout()
    plt.show()
