from typing import List
import tensor
import numpy as np


class Operator:
    def __init__(self):
        pass

    def forward(self, src_list: List[tensor.Tensor]) -> np.ndarray:
        pass

    def backward(self, result: tensor.Tensor, src_list: List[tensor.Tensor]):
        pass


class MatMul(Operator):
    def __init__(self):
        super().__init__()

    def forward(self, src_list: List[tensor.Tensor]) -> np.ndarray:
        t1 = src_list[0]
        t2 = src_list[1]
        data = np.matmul(t1.data, t2.data)
        return data

    def backward(self, result: tensor.Tensor, src_list: List[tensor.Tensor]):
        r_grad = result.grad
        for i in range(len(src_list)):
            src_grad = None
            if i == 0:
                axes = tuple(range(src_list[1].data.ndim - 2)) + (-1, -2)
                src_grad = np.matmul(r_grad, src_list[1].data.transpose(axes))
            elif i == 1:
                axes = tuple(range(src_list[0].data.ndim - 2)) + (-1, -2)
                src_grad = np.matmul(src_list[0].data.transpose(axes), r_grad)
            src_list[i].accumulate_grad(delta_grad=src_grad)


class Add(Operator):
    def __init__(self):
        super().__init__()

    def forward(self, src_list: List[tensor.Tensor]) -> np.ndarray:
        return src_list[0].data + src_list[1].data

    def backward(self, result: tensor.Tensor, src_list: List[tensor.Tensor]) -> None:
        r_grad = result.grad
        for i in range(len(src_list)):
            src_list[i].accumulate_grad(delta_grad=r_grad)


class Relu(Operator):
    def __init__(self):
        super().__init__()

    def forward(self, src_list: List[tensor.Tensor]) -> np.ndarray:
        return np.maximum(np.zeros_like(src_list[0].data), src_list[0].data)

    def backward(self, result: tensor.Tensor, src_list: List[tensor.Tensor]):
        r_grad = result.grad
        for i in range(len(src_list)):
            src_grad = None
            if i == 0:
                src_grad = r_grad * (src_list[i].data >= 0)
            src_list[i].accumulate_grad(delta_grad=src_grad)


class Softmax(Operator):
    def __init__(self):
        super().__init__()

    def forward(self, src_list: List[tensor.Tensor]) -> np.ndarray:
        z = src_list[0].data
        p = z / np.sum(np.exp(z), axis=-1, keepdims=True)
        return p

    def backward(self, result: tensor.Tensor, src_list: List[tensor.Tensor]):
        r_grad = result.grad
        r_grad: np.ndarray
        for i in range(len(src_list)):
            src = src_list[i]
            src_grad = None
            if i == 0:
                P = src.data
                axes = tuple(range(r_grad.ndim - 2)) + (-1, -2)
                dim = P.shape[-1]
                pre_shape = P.shape[:-1]
                I = np.identity(dim)
                I = np.tile(I, list(pre_shape) + list((1, 1)))
                one = np.ones(dim)
                one = np.tile(one, list(pre_shape) + list((1,)))
                # print("P.shape=", P.shape)
                # print("I.shape=", I.shape)
                # print("one.shape=", one.shape)
                # print("einsum.shape=", np.einsum("...i,...j->...ij", P, one).shape)
                tmp = I - np.einsum("...i,...j->...ij", P, one)
                r_grad = np.expand_dims(r_grad, -2)
                tmp = np.matmul(r_grad, tmp)
                tmp: np.ndarray
                tmp = tmp.squeeze(axis=-2)
                # print("tmp.shape=", tmp.shape)
                src_grad = tmp * P
                src_list[i].accumulate_grad(delta_grad=src_grad)


class BroadcastPre(Operator):
    def __init__(self, pre_shape: tuple):
        super().__init__()
        self.pre_shape = pre_shape

    def forward(self, src_list: List[tensor.Tensor]) -> np.ndarray:
        src = src_list[0]
        data = src.data
        data: np.ndarray
        ndim = data.ndim
        ones = np.ones(ndim).astype(int)
        rep_tuple = tuple(list(self.pre_shape) + list(ones))
        return np.tile(data, rep_tuple)

    def backward(self, result: tensor.Tensor, src_list: List[tensor.Tensor]):
        r_grad = result.grad
        r_grad: np.ndarray
        for i in range(len(src_list)):
            src = src_list[i]
            src_grad = None
            if i == 0:
                src_grad = r_grad.sum(axis=tuple(range(len(self.pre_shape))))
            src_list[i].accumulate_grad(delta_grad=src_grad)


class YNegLogSoftmax(Operator):
    def __init__(self):
        super().__init__()
        self.p = None

    def forward(self, src_list: List[tensor.Tensor]) -> np.ndarray:
        z = src_list[0].data
        y = src_list[1].data
        z: np.ndarray
        y: np.ndarray
        z_tmp = z - z.max(axis=-1, keepdims=True)
        z_tmp: np.ndarray
        s_tmp = np.exp(z_tmp).sum(axis=-1, keepdims=True)
        self.p = z_tmp / s_tmp
        result = - (z_tmp - np.log(s_tmp))
        return result

    def backward(self, result: tensor.Tensor, src_list: List[tensor.Tensor]):
        r_grad = result.grad
        for i in range(len(src_list)):
            if i == 0:
                y = src_list[1].data
                src_grad = self.p - y
                print("self.p.shape=", self.p.shape)
                print("y.shape", y.shape)
                print("src_grad.shape", src_grad.shape)
                src_list[i].accumulate_grad(delta_grad=src_grad)


class SumAll(Operator):
    def __init__(self):
        super().__init__()

    def forward(self, src_list: List[tensor.Tensor]) -> np.ndarray:
        data = src_list[0].data
        return data.sum() / data.size

    def backward(self, result: tensor.Tensor, src_list: List[tensor.Tensor]):
        r_grad = result.grad
        for i in range(len(src_list)):
            if i == 0:
                data = src_list[i].data
                data: np.ndarray
                ne = data.size
                src_grad = r_grad * (1.0 / ne)
                src_list[i].accumulate_grad(delta_grad=src_grad)
