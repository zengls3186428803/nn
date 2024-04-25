import numpy as np
import tensor
import functs_lib

A = tensor.Tensor(np.random.randn(1, 2, 3))
B = tensor.Tensor(np.random.randn(1, 3, 2))
D = tensor.Tensor(np.random.randn(1, 2, 2))
C = functs_lib.matmul(A, B)
E = functs_lib.add(C, D)
F = functs_lib.relu(E)
G = functs_lib.softmax(F)
H = functs_lib.broadcast_pre(G, (9, 8))
I = np.random.randn(9, 8, 1, 2, 2)
I = tensor.Tensor(I)
J = functs_lib.cross_entropy(H, I)
Z = J
Z.forward()
Z.grad = np.ones(1)
Z.backward()
print(Z.data)
print(A.grad)
