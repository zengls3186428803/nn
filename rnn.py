import numpy as np
from rnn_utils import softmax, sigmoid


class Rnn(object):
    def __init__(self, n_a=5):
        self.unit = "lstm"
        self.paras = dict()
        self.cache = dict()
        self.n_a = n_a
        self.n_x = 0
        self.n_y = 0
        self.T_x = 0
        self.T_y = 0
        self.m = 0
        pass

    def init_paras(self):
        if self.unit == "normal":
            self.paras["Wa"] = np.random.randn(self.n_a, self.n_a + self.n_x)
            self.paras["ba"] = np.random.randn(self.n_a).reshape((-1, 1))
        elif self.unit == "lstm":
            self.paras["Wf"] = np.random.randn(self.n_a, self.n_a + self.n_x)
            self.paras["Wu"] = np.random.randn(self.n_a, self.n_a + self.n_x)
            self.paras["Wc"] = np.random.randn(self.n_a, self.n_a + self.n_x)
            self.paras["Wo"] = np.random.randn(self.n_a, self.n_a + self.n_x)
            self.paras["bf"] = np.random.randn(self.n_a).reshape((-1, 1))
            self.paras["bu"] = np.random.randn(self.n_a).reshape((-1, 1))
            self.paras["bc"] = np.random.randn(self.n_a).reshape((-1, 1))
            self.paras["bo"] = np.random.randn(self.n_a).reshape((-1, 1))
        self.paras["Wy"] = np.random.randn(self.n_y, self.n_a)
        self.paras["by"] = np.random.randn(self.n_y).reshape((-1, 1))

    def init(self, x, y):
        self.n_x, self.m, self.T_x = x.shape
        self.n_y, self.m, self.T_y = y.shape
        self.init_paras()
        self.cache["a-1"] = np.random.randn(self.n_a, self.m)
        self.cache["a" + str(self.T_x)] = np.zeros((self.n_a, self.m))
        if self.unit == "lstm":
            self.cache["c-1"] = np.zeros((self.n_a, self.m))
            self.cache["c" + str(self.T_x)] = np.zeros((self.n_a, self.m))
        pass

    def init_gradient(self):
        self.cache["dWy"] = np.zeros((self.n_y, self.n_a))
        self.cache["dby"] = np.zeros((self.n_y, 1))
        self.cache["dAa" + str(self.T_x)] = np.zeros((self.n_a, self.m))
        if self.unit == "normal":
            self.cache["dWa"] = np.zeros((self.n_a, self.n_a + self.n_x))
            self.cache["dba"] = np.zeros((self.n_a, 1))
        elif self.unit == "lstm":
            self.cache["dWo"] = np.zeros((self.n_a, self.n_a + self.n_x))
            self.cache["dWc"] = np.zeros((self.n_a, self.n_a + self.n_x))
            self.cache["dWu"] = np.zeros((self.n_a, self.n_a + self.n_x))
            self.cache["dWf"] = np.zeros((self.n_a, self.n_a + self.n_x))
            self.cache["dbo"] = np.zeros((self.n_a, 1))
            self.cache["dbc"] = np.zeros((self.n_a, 1))
            self.cache["dbu"] = np.zeros((self.n_a, 1))
            self.cache["dbf"] = np.zeros((self.n_a, 1))
            self.cache["dc_pre" + str(self.T_x)] = np.zeros((self.n_a, self.m))
            pass

    def fit(self, x, y, epochs=1000, learning_rate=0.01):
        self.init(x, y)
        for epoch in range(epochs):
            self.f_p(x)
            cost = self.get_cost(y)
            print(cost)
            self.b_p(y)
            self.u_p(learning_rate)

    def f_p(self, x):
        if self.unit == "lstm":
            for t in range(0, self.T_x):
                self.c_p(t, a_pre=self.cache["a" + str(t - 1)], x_t=x[:, :, t], c_pre=self.cache["c" + str(t - 1)])
        elif self.unit == "normal":
            for t in range(0, self.T_x):
                self.c_p(t, a_pre=self.cache["a" + str(t - 1)], x_t=x[:, :, t])

    def get_cost(self, y):
        cost1 = 0  # cost1为交叉熵
        for t in range(0, self.T_x):
            Z = self.cache["Zy" + str(t)]
            max_Z = np.max(Z, axis=0)
            loss_matrix = y[:, :, t] * (max_Z - Z + np.log(np.sum(np.exp(Z - max_Z), axis=0)))
            cost1 += (1 / self.m) * np.sum(np.sum(loss_matrix, axis=0), axis=0)
        return cost1

    def b_p(self, y):
        self.init_gradient()
        if self.unit == "normal":
            for t in range(self.T_x - 1, -1, -1):
                self.c_b(t, y)
        elif self.unit == "lstm":
            for t in range(self.T_x - 1, -1, -1):
                self.c_b(t, y)
            pass

    def u_p(self, learning_rate):
        if self.unit == "normal":
            self.paras["Wy"] -= learning_rate * self.cache["dWy"]
            self.paras["by"] -= learning_rate * self.cache["dby"]
            self.paras["Wa"] -= learning_rate * self.cache["dWa"]
            self.paras["ba"] -= learning_rate * self.cache["dba"]
        elif self.unit == "lstm":
            self.paras["Wy"] -= learning_rate * self.cache["dWy"]
            self.paras["Wo"] -= learning_rate * self.cache["dWo"]
            self.paras["Wc"] -= learning_rate * self.cache["dWc"]
            self.paras["Wu"] -= learning_rate * self.cache["dWu"]
            self.paras["Wf"] -= learning_rate * self.cache["dWf"]

            self.paras["by"] -= learning_rate * self.cache["dby"]
            self.paras["bo"] -= learning_rate * self.cache["dbo"]
            self.paras["bc"] -= learning_rate * self.cache["dbc"]
            self.paras["bu"] -= learning_rate * self.cache["dbu"]
            self.paras["bf"] -= learning_rate * self.cache["dbf"]
            pass

    def predict(self):
        pass

    def c_p(self, t, a_pre, x_t, c_pre=None):
        a__x = np.concatenate((a_pre, x_t), axis=0)
        self.cache["a__x" + str(t)] = a__x
        if self.unit == "normal":
            self.cache["a" + str(t)] = np.tanh(np.dot(self.paras["Wa"], a__x) + self.paras["ba"])
        elif self.unit == "lstm":
            self.cache["gate_f" + str(t)] = sigmoid(np.dot(self.paras["Wf"], a__x) + self.paras["bf"])
            self.cache["gate_u" + str(t)] = sigmoid(np.dot(self.paras["Wu"], a__x) + self.paras["bu"])
            self.cache["c~" + str(t)] = np.tanh(np.dot(self.paras["Wc"], a__x) + self.paras["bc"])
            self.cache["gate_o" + str(t)] = sigmoid(np.dot(self.paras["Wo"], a__x) + self.paras["bo"])
            self.cache["c" + str(t)] = self.cache["gate_f" + str(t)] * a_pre + self.cache["gate_u" + str(t)] * self.cache["c~" + str(t)]
            self.cache["a" + str(t)] = self.cache["gate_o" + str(t)] * np.tan(self.cache["c" + str(t)])
        elif self.unit == "gru":
            pass
        self.cache["Zy" + str(t)] = np.dot(self.paras["Wy"], self.cache["a" + str(t)]) + self.paras["by"]  # 计算损失函数的时候用
        self.cache["y" + str(t)] = softmax(self.cache["Zy" + str(t)])

    def c_b(self, t, y):
        if self.unit == "normal":
            self.cache["dZy" + str(t)] = (1 / self.m) * (self.cache["y" + str(t)] - y[:, :, t])  # (1/m)(A-Y)
            self.cache["dWy"] += np.dot(self.cache["dZy" + str(t)], self.cache["a" + str(t)].T)  # dZ . A.T
            self.cache["dby"] += np.sum(self.cache["dZy" + str(t)], axis=1).reshape((-1, 1))  # d对dZ按列求和
            self.cache["dAy" + str(t)] = np.dot(self.paras["Wy"].T, self.cache["dZy" + str(t)])  # W.T . dZ

            self.cache["dA" + str(t)] = self.cache["dAy" + str(t)] + self.cache["dAa" + str(t + 1)]

            self.cache["dZ" + str(t)] = self.cache["dA" + str(t)] * (
                    1 - self.cache["a" + str(t)] * self.cache["a" + str(t)])
            self.cache["dWa"] += np.dot(self.cache["dZ" + str(t)], self.cache["a__x" + str(t)].T)
            self.cache["dba"] += np.sum(self.cache["dZ" + str(t)], axis=1).reshape((-1, 1))
            self.cache["da__x" + str(t)] = np.dot(self.paras["Wa"].T, self.cache["dZ" + str(t)])

            self.cache["dAa" + str(t)] = self.cache["da__x" + str(t)][range(self.n_a), :]

        if self.unit == "lstm":
            self.cache["dZy" + str(t)] = (1 / self.m) * (self.cache["y" + str(t)] - y[:, :, t])  # (1/m)(A-Y)
            self.cache["dWy"] += np.dot(self.cache["dZy" + str(t)], self.cache["a" + str(t)].T)  # dZ . A.T
            self.cache["dby"] += np.sum(self.cache["dZy" + str(t)], axis=1).reshape((-1, 1))  # d对dZ按列求和
            self.cache["dAy" + str(t)] = np.dot(self.paras["Wy"].T, self.cache["dZy" + str(t)])  # W.T . dZ

            self.cache["dA" + str(t)] = self.cache["dAy" + str(t)] + self.cache["dAa" + str(t + 1)]
            self.cache["dc" + str(t)] = self.cache["dc_pre" + str(t + 1)]
            self.cache["dc" + str(t)] += 1 - np.square(self.cache["dA" + str(t)] * self.cache["gate_o" + str(t)])
            self.cache["da__x" + str(t)] = np.zeros((self.n_a + self.n_x, self.m))

            #输出门
            self.cache["dgate_o" + str(t)] = self.cache["dA" + str(t)] * np.tanh(self.cache["c" + str(t)])
            self.cache["dgate_oZ" + str(t)] = self.cache["dgate_o" + str(t)] * self.cache["gate_o" + str(t)] * (1 - self.cache["gate_o" + str(t)])
            self.cache["dWo"] += np.dot(self.cache["dgate_oZ" + str(t)], self.cache["a__x" + str(t)].T)
            self.cache["dbo"] += np.sum(self.cache["dgate_oZ" + str(t)], axis=1).reshape((-1, 1))
            self.cache["da__x" + str(t)] += np.dot(self.paras["Wo"].T, self.cache["dgate_oZ" + str(t)])

            #tanh门
            self.cache["dc~" + str(t)] = self.cache["dc" + str(t)] * self.cache["gate_u" + str(t)]
            self.cache["dc~Z" + str(t)] = self.cache["dc~" + str(t)] * (1 - np.square(self.cache["c~" + str(t)]))
            self.cache["dWc"] += np.dot(self.cache["dc~Z" + str(t)], self.cache["a__x" + str(t)].T)
            self.cache["dbc"] += np.sum(self.cache["dc~Z" + str(t)], axis=1).reshape((-1, 1))
            self.cache["da__x" + str(t)] += np.dot(self.paras["Wc"].T, self.cache["dc~Z" + str(t)])

            #更新门
            self.cache["dgate_u" + str(t)] = self.cache["dc" + str(t)] * self.cache["c~" + str(t)]
            self.cache["dgate_uZ" + str(t)] = self.cache["dgate_u" + str(t)] * self.cache["gate_u" + str(t)] * (1 - self.cache["gate_u" + str(t)])
            self.cache["dWu"] += np.dot(self.cache["dgate_uZ" + str(t)], self.cache["a__x" + str(t)].T)
            self.cache["dbu"] += np.sum(self.cache["dgate_uZ" + str(t)], axis=1).reshape((-1, 1))
            self.cache["da__x" + str(t)] += np.dot(self.paras["Wu"].T, self.cache["dgate_uZ" + str(t)])

            #遗忘门
            self.cache["dgate_f" + str(t)] = self.cache["dc" + str(t)] * self.cache["c" + str(t - 1)]
            self.cache["dgate_fZ" + str(t)] = self.cache["dgate_f" + str(t)] * self.cache["gate_f" + str(t)] * (1 - self.cache["gate_f" + str(t)])
            self.cache["dWf"] += np.dot(self.cache["dgate_fZ" + str(t)], self.cache["a__x" + str(t)].T)
            self.cache["dbf"] += np.sum(self.cache["dgate_fZ" + str(t)], axis=1).reshape((-1, 1))
            self.cache["da__x" + str(t)] += np.dot(self.paras["Wf"].T, self.cache["dgate_fZ" + str(t)])

            #c
            self.cache["dc_pre" + str(t)] = self.cache["dc" + str(t)] * self.cache["gate_f" + str(t)]

            #A
            self.cache["dAa" + str(t)] = self.cache["da__x" + str(t)][range(self.n_a), :]
            pass


rnn = Rnn(n_a=6)
x = np.random.randn(3, 10, 7)
y = np.random.randn(2, 10, 7)
y = (y > 0.5) * 1
rnn.fit(x, y, learning_rate=0.001)
