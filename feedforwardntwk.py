# Written by Stephen Blystone
# Based on https://iamtrask.github.io/2015/07/12/basic-python-network/
# Part of my "100 Days of ML" 2018

import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def relu(x, deriv=False):
    if deriv:
        # Make dx the same shape and type as x.
        # Fill dx with ones.
        dx = np.ones_like(x)
        dx[x <= 0] = 0
        return dx
    return np.max(x, 0)


def leakyRelu(x, deriv=False):
    alpha = 0.01

    if deriv:
        # Make dx the same shape and type as x.
        # Fill dx with ones.
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx
    x[x < 0] *= alpha
    return x

# Neural Network Shape.
# 4 training examples.
# 3 inputs.
# L1 = 4 neurons.
# 1 output.


def runNetwork(X, y, iterations=60000, nonlin=sigmoid):
    # initialize weights with mean 0
    l0_weights = 2 * np.random.random((3, 4)) - 1
    l1_weights = 2 * np.random.random((4, 1)) - 1

    # Go through iterations.
    for j in range(iterations):
        l0 = X
        l1 = nonlin(np.dot(l0, l0_weights))
        l2 = nonlin(np.dot(l1, l1_weights))

        l2_error = y - l2

        if (j % 10000) == 0:
            print("Error: ", str(np.mean(np.abs(l2_error))))

        # Update weights.
        l2_delta = l2_error * nonlin(l2, deriv=True)
        l1_error = l2_delta.dot(l1_weights.T)
        l1_delta = l1_error * nonlin(l1, deriv=True)

        l1_weights += l1.T.dot(l2_delta)
        l0_weights += l0.T.dot(l1_delta)


if __name__ == "__main__":
        # Seed random numbers to make calculations deterministic.
    np.random.seed(1)

    # Input dataset
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    # Output dataset
    y = np.array([[0, 0, 1, 1]]).T

    runNetwork(X, y, nonlin=leakyRelu)
