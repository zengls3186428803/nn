from nn import neural_network
from deep_learning_course.course_1_3.planar_utils import load_planar_dataset


def test(X_train, Y_train, X_test, Y_test, learning_rate=0.1, iter_num=1000, batch_size=200,
         a_functs=("tanh", "softmax"), L=None, class_num=2):
    if L is None:
        L = [2, 5, 5, 5, 5, 1]

    if a_functs[1] == "softmax":
        Y_train = neural_network().get_one_hot(Y_train, class_num=class_num)
    elif a_functs[1] == "sigmoid":
        pass
    nn = neural_network(dim_num_list=L)
    nn.fit(X_train, Y_train, learning_rate=learning_rate, iter_num=iter_num, batch_size=batch_size, a_functs=a_functs)
    result = nn.predict(X_test)
    print("result = ", result)
    y = Y_test[0]
    cnt = 0
    for e in range(len(y)):
        if result[e] == y[e]:
            cnt += 1
    print(cnt / len(y))


X_train, Y_train = load_planar_dataset()
X_train /= 4
# plt.scatter(X_train[0, :], X_train[1, :], c=Y_train, s=40, cmap=plt.cm.Spectral) #绘制散点图
# plt.show()
X_test, Y_test = load_planar_dataset()
X_test /= 4

test(X_train, Y_train, X_test, Y_test, a_functs=("tanh", "sigmoid"))
