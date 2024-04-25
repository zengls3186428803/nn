import tensor
import my_operator


def matmul(t1: tensor.Tensor, t2: tensor.Tensor):
    result = tensor.Tensor()
    result.op = my_operator.MatMul()
    result.src_list = [t1, t2]
    return result


def add(t1: tensor.Tensor, t2: tensor.Tensor):
    result = tensor.Tensor()
    result.op = my_operator.Add()
    result.src_list = [t1, t2]
    return result


def relu(t1: tensor.Tensor):
    result = tensor.Tensor()
    result.op = my_operator.Relu()
    result.src_list = [t1]
    return result


def softmax(t1: tensor.Tensor):
    result = tensor.Tensor()
    result.op = my_operator.Softmax()
    result.src_list = [t1]
    return result


def broadcast_pre(t1: tensor.Tensor, pre_shape):
    result = tensor.Tensor()
    result.op = my_operator.BroadcastPre(pre_shape)
    result.src_list = [t1]
    return result


def sum_all(t1: tensor.Tensor):
    result = tensor.Tensor()
    result.op = my_operator.SumAll()
    result.src_list = [t1]
    return result


def cross_entropy(t1: tensor.Tensor, t2: tensor.Tensor):
    result = tensor.Tensor()
    result.op = my_operator.YNegLogSoftmax()
    result.src_list = [t1, t2]
    loss = sum_all(result)
    return loss
