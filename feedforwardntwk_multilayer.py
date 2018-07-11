# Written by Stephen Blystone
# Originally based on https://iamtrask.github.io/2015/07/12/basic-python-network/
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


def initializeWeights(numberNodes):
    weights = []

    # initialize weights with mean 0
    for l in range(len(numberNodes) - 1):
        weights.append(2 * np.random.random((numberNodes[l], numberNodes[l + 1])) - 1)

    return weights


def runNetwork(X, y, hiddenLayerSizes, iterations=60000, nonlin=sigmoid):
    # numberNodes is number inputs, hiddenLayerSizes, number outputs.
    numberNodes = [X.shape[1]]
    numberNodes.extend(hiddenLayerSizes)
    numberNodes.append(y.shape[1])

    # Initialize weights
    weights = initializeWeights(numberNodes)

    # # Initialize bias
    # biases = initializeBiases(numberNodes)

    # Go through iterations.
    for j in range(iterations):
        # layers keeps track of the forward propagation values at each layer.
        layers = [0] * len(numberNodes)

        # Forward propagate.
        for l in range(len(numberNodes)):
            if l == 0:
                layers[l] = X  # Input layer
                continue
            layers[l] = nonlin(np.dot(layers[l - 1], weights[l - 1]))

        # Calculate output error.
        outputError = y - layers[-1]

        if (j % 10000) == 0:
            print("Error: ", str(np.mean(np.abs(outputError))))

        # Backward propagate.
        # Deltas keeps track of the derivative values.
        # Errors keeps track of the errors at each layer.
        deltas = [0] * (len(numberNodes))
        errors = [0] * (len(numberNodes))

        # Start at last layer and move toward first layer.
        for l in range(len(numberNodes) - 1, 0, -1):
            if l == (len(numberNodes) - 1):
                errors[l] = outputError

            deltas[l] = errors[l] * nonlin(layers[l], deriv=True)
            if l != 0:
                # Only need to calculate the errors if this is not the first layer.
                errors[l - 1] = deltas[l].dot(weights[l - 1].T)

        # Update weights.
        for l in range(len(numberNodes) - 1):
            weights[l] += layers[l].T.dot(deltas[l + 1])


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

    hiddenLayerSizes = [3, 10, 3]

    runNetwork(X, y, hiddenLayerSizes, nonlin=leakyRelu)
