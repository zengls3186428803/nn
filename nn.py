import numpy as np
from function_set.activate_functions import sigmoid, relu, tanh, softmax  # eval()会用到，不删
from function_set.activate_differential import sigmoid_d, relu_d, tanh_d  # eval（）会用到，不删


class neural_network:
    def __init__(self,
                 layer_num=3,
                 dim_num_list=None,
                 funct_list=None,
                 a_functs=("tanh", "sigmoid"),
                 reg="L2",
                 lambda_reg=0.0,
                 beta_momentum=0.9,
                 batch_size=-1,
                 keep_prob=1.0
                 ):
        self.funct_list = funct_list  # 激活函数列表，记录每一层的激活函数
        self.layer_num = layer_num  # 层数
        self.dim_num_list = dim_num_list  # 每一层的结点数
        self.a_functs = a_functs  # 激活函数名
        self.paras = dict()  # 保存参数 W,b
        self.cache = dict()  # 存储前向传播的数据，供反向传播使用
        self.reg = reg.upper()
        self.epsilon = 1e-8  # 精度，防止除0或log
        self.lambda_reg = lambda_reg  # L2正则化的lambda
        self.beta_momentum = beta_momentum  # 动量梯度下降法的超参数
        self.iter_cnt = 0  # 迭代计数, 用于momentum等优化
        self.batch_size = batch_size  # mini-batch的大小
        self.keep_prob = keep_prob  # dropout中的保持概率

    def init(self, X, Y):
        if not self.dim_num_list:
            self.dim_num_list = np.random.randint(30, 31, self.layer_num + 1)  # 默认维度为30
        else:
            self.layer_num = len(self.dim_num_list) - 1
        if not self.funct_list:
            self.funct_list = list()
            for i in range(0, self.layer_num):
                self.funct_list.append(self.a_functs[0])
            self.funct_list.append(self.a_functs[1])
        self.dim_num_list[0] = X.shape[0]
        self.dim_num_list[self.layer_num] = Y.shape[0]
        for i in range(1, self.layer_num + 1):
            self.cache["v_dW" + str(i)] = np.zeros((self.dim_num_list[i], self.dim_num_list[i - 1]))
            self.cache["v_db" + str(i)] = np.zeros((self.dim_num_list[i], 1))
        self.init_paras()

    def init_paras(self):
        for i in range(1, self.layer_num + 1):
            self.paras["W" + str(i)] = np.random.randn(self.dim_num_list[i], self.dim_num_list[i - 1])
            self.paras["b" + str(i)] = np.zeros(self.dim_num_list[i], float).reshape(-1, 1)

    def forward_propagate(self, X):
        self.cache["A0"] = X
        for i in range(1, self.layer_num + 1):
            self.cache["Z" + str(i)] = np.dot(self.paras["W" + str(i)], self.cache["A" + str(i - 1)])
            self.cache["A" + str(i)] = eval(self.funct_list[i])(self.cache["Z" + str(i)])
            if self.keep_prob < 1:  # dropout
                D = np.random.rand(self.cache["A" + str(i)].shape[0], self.cache["A" + str(i)].shape[1])
                D = (D < self.keep_prob) * 1
                self.cache["D" + str(i)] = D
                self.cache["A" + str(i)] = self.cache["A" + str(i)] * D / self.keep_prob

    def get_cost(self, Y):
        m = self.cache["A0"].shape[1]
        A = self.cache["A" + str(self.layer_num)]
        Z = self.cache["A" + str(self.layer_num)]
        cost1 = 0  # cost1为交叉熵
        loss_matrix = None
        if self.funct_list[self.layer_num] == "sigmoid":
            loss_matrix = np.maximum(Z, 0) - Z * Y + np.log(1 + np.exp(-np.abs(Z)))
        elif self.funct_list[self.layer_num] == "softmax":
            max_Z = np.max(Z, axis=0)
            loss_matrix = Y * (max_Z - Z + np.log(np.sum(np.exp(Z - max_Z), axis=0)))
        cost1 += (1 / m) * np.sum(np.sum(loss_matrix, axis=0), axis=0)
        cost2 = 0  # cost2 为正则项
        if self.reg == "L2":
            for i in range(1, self.layer_num + 1):
                cost2 += self.lambda_reg * np.sum(np.sum(self.paras["W" + str(i)] * self.paras["W" + str(i)], axis=0),
                                                  axis=0)
        else:
            pass
        cost = cost1 + cost2
        return cost

    def backward_propagate(self, Y):
        A = self.cache["A" + str(self.layer_num)]
        m = A.shape[1]
        dA = (-1 / m) * Y * (1 / (A + self.epsilon))
        i = self.layer_num
        while (i > 0):
            function_name = self.funct_list[i]
            if (i == self.layer_num) and (function_name == "softmax" or function_name == "sigmoid"):
                dZ = (1 / m) * (A - Y)
            else:
                if self.keep_prob < 1:
                    dA = dA * self.cache["D" + str(i)] / self.keep_prob
                dZ = dA * eval(function_name + "_d")(self.cache["A" + str(i)], self.cache["Z" + str(i)])
            dW_reg = 2 * self.lambda_reg * self.paras["W" + str(i)]
            db_reg = 2 * self.lambda_reg * self.paras["b" + str(i)]
            self.cache["dW" + str(i)] = np.dot(dZ, self.cache["A" + str(i - 1)].T) + dW_reg
            self.cache["db" + str(i)] = np.sum(dZ, axis=1).reshape(-1, 1) + db_reg
            dA = np.dot(self.paras["W" + str(i)].T, dZ)
            i -= 1

    def update_parameters(self, learning_rate):
        for i in range(1, self.layer_num + 1):
            self.cache["v_dW" + str(i)] = self.beta_momentum * self.cache["v_dW" + str(i)] + (1 - self.beta_momentum) * \
                                          self.cache["dW" + str(i)]
            self.cache["v_db" + str(i)] = self.beta_momentum * self.cache["v_db" + str(i)] + (1 - self.beta_momentum) * \
                                          self.cache["db" + str(i)]
            self.cache["v_c_dW" + str(i)] = self.cache["v_dW" + str(i)] / (1 - self.beta_momentum ** self.iter_cnt)
            self.cache["v_c_db" + str(i)] = self.cache["v_db" + str(i)] / (1 - self.beta_momentum ** self.iter_cnt)
            self.paras["W" + str(i)] = self.paras["W" + str(i)] - learning_rate * self.cache["v_c_dW" + str(i)]
            self.paras["b" + str(i)] = self.paras["b" + str(i)] - learning_rate * self.cache["v_c_db" + str(i)]

    def fit(self, X, Y, learning_rate=0.5, iter_num=1000, batch_size=-1, a_functs=("tanh", "softmax")):
        self.batch_size = batch_size
        self.a_functs = a_functs
        self.init(X, Y)
        batch_generator = self.get_batch(X, Y)
        for X, Y in batch_generator:
            self.iter_cnt = 0
            for epoch in range(iter_num):
                self.iter_cnt += 1
                self.forward_propagate(X)
                cost = self.get_cost(Y)
                print(cost)
                self.backward_propagate(Y)
                self.update_parameters(learning_rate)

    def predict_probability(self, X):
        self.forward_propagate(X)
        return self.cache["A" + str(self.layer_num)]

    def predict(self, X):
        probabilty = self.predict_probability(X)
        return self.map_to_int(probabilty)

    def get_batch(self, X, Y):  # mini-batch生成器
        m = X.shape[1]
        if self.batch_size == -1:
            self.batch_size = m
        batch_num = (m + self.batch_size - 1) // self.batch_size
        index_list = np.random.permutation(m)
        start = 0
        for i in range(batch_num):
            end = min(start + self.batch_size, m)
            yield X[:, index_list[start:end]], Y[:, index_list[start:end]]
            start = end

    def map_to_int(self, A):  # 概率转预测结果
        result = None  #
        if self.funct_list[self.layer_num] == "softmax":
            result = np.argmax(A, axis=0)
        elif self.funct_list[self.layer_num] == "sigmoid":
            result = (A >= 0.5) * 1
            result = result[0]
        return result

    def get_one_hot(self, Y, class_num):  # 将整数类别转换为独热编码
        m = Y.shape[1]
        n = class_num
        y = Y[0]
        result = np.zeros((n, m))
        for i in range(m):
            result[y[i]][i] = 1
        return result

