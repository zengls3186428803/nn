from typing import List
import numpy as np


class Tensor:
    def __init__(self, data=None, op=None, src_list=None):
        data: np.ndarray
        src_list: List[Tensor]

        self.data = None
        self.op = op
        self.src_list = src_list
        self.grad = None
        self.cnt_for_backward = 0
        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        self.data = data
        if self.grad is None:
            self.grad = np.zeros(data.shape)

    def forward(self):
        if self.src_list is not None and len(self.src_list) > 0:
            for src in self.src_list:
                src.forward()
            data = self.op.forward(src_list=self.src_list)
            self.set_data(data)

    def backward(self):
        if self.op is not None:
            self.op.backward(self, self.src_list)

    def down_cnt_for_backward(self):
        self.cnt_for_backward -= 1
        if self.cnt_for_backward <= 0:
            self.backward()

    def accumulate_grad(self, delta_grad):
        delta_grad: np.ndarray
        self.grad += delta_grad
        self.down_cnt_for_backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.grad)
        if self.src_list is not None and len(self.src_list) > 0:
            for src in self.src_list:
                src.zero_grad()

    def update_tensor(self, lr):
        self.data = self.data - lr * self.grad
        if self.src_list is not None and len(self.src_list) > 0:
            for src in self.src_list:
                src.update_tensor(lr)
