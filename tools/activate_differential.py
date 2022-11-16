import numpy as np


def sigmoid_d(A, Z):
    return A * (1 - A)


def relu_d(A ,Z):
    result = (Z > 0) * 1
    return result


def tanh_d(A, Z):
    return 1 - A * A


def softmax_d(A, Z):
    pass