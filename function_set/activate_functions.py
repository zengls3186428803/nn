import numpy as np


def sigmoid(Z):
    positive_mask = (Z >= 0)
    negative_mask = Z < 0
    result_positive = 1 / (1 + np.exp(-Z * positive_mask))
    result_positive[~positive_mask] = 0
    result_negative = np.exp(Z * negative_mask) / (np.exp(Z * negative_mask) + 1)
    result_negative[~negative_mask] = 0
    result = result_negative + result_positive
    return result


def relu(Z):
    return np.maximum(0, Z)


def tanh(Z):
    return np.tanh(Z)


def softmax(Z):
    max_Z = np.max(Z, axis=0)
    return np.exp(Z - max_Z) / np.sum(np.exp(Z - max_Z), axis=0)
