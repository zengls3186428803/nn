import cnn_utils
import tf_utils
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T
print(type(X_test))
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


class Cnn(object):
    def __init__(self, hyper_paras=None, paras=None, layer=None):
        if hyper_paras is None:
            hyper_paras = None
        if paras is None:
            paras = dict()
        if layer is None:
            layer = dict()
            layer["name"] = [""] + ["conv"] + ["pool"] + ["conv"] + ["pool"] + ["full"] + ["softmax"]
            layer["num"] = [1] + [1] + [1] + [1] + [1] + [3]
        self.hyper_paras = hyper_paras
        self.paras = paras
        self.layer = layer
        self.cache = dict()
        self.learning_rate = 0.1
        self.x_ = None
        self.y_ = None

    def init_paras(self):
        self.paras["w" + str(1)] = tf.get_variable("w1", [4, 4, 3, 8], dtype=float,
                                                   initializer=tf.keras.initializers.glorot_normal())
        self.paras["w" + str(3)] = tf.get_variable("w3", [2, 2, 8, 16], dtype=float,
                                                   initializer=tf.keras.initializers.glorot_normal())

    def fit(self, x, y, learning_rate=0.01, iter_num=1000):
        (m, n_h, n_w, n_c) = x.shape
        (m, n_y) = y.shape
        self.learning_rate = learning_rate
        self.x_ = tf.placeholder(tf.float32, [None, n_h, n_w, n_c])
        self.y_ = tf.placeholder(tf.float32, [None, n_y])
        self.init_paras()
        self.f_p(self.x_, self.y_)
        self.b_p()
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for cnt in range(iter_num):
                _, cost = session.run([self.cache["optimizer"], self.cache["cost"]], feed_dict={self.x_: x, self.y_: y})
                if cnt % 5 == 0:
                    print("第{}次代价为：{}".format(cnt, cost))
            saver = tf.train.Saver()
            saver.save(session, "./checkpoint_dir/model")

    def predict(self, x, y):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "./checkpoint_dir/model")
            a = self.cache["output"]
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(a, axis=1), tf.argmax(self.y_, axis=1)), tf.float32))
            accuracy = sess.run(accuracy, feed_dict={self.x_: x, self.y_: y})
            print(accuracy)

    def f_p(self, x, y):
        self.cache["a0"] = x
        m, n_y = y.shape
        for i in range(1, len(self.layer["name"])):
            print("i = ", i)
            name = self.layer["name"][i]
            if name == "conv":
                self.cache["z" + str(i)] = tf.nn.conv2d(tf.cast(self.cache["a" + str(i - 1)], tf.float32),
                                                        self.paras["w" + str(i)],
                                                        strides=[1, 1, 1, 1], padding="SAME")
                self.cache["a" + str(i)] = tf.nn.relu(self.cache["z" + str(i)])
            elif name == "pool":
                self.cache["a" + str(i)] = tf.nn.max_pool(self.cache["a" + str(i - 1)], ksize=[1, 8, 8, 1],
                                                          strides=[1, 8, 8, 1], padding="SAME")
            elif name == "full":
                a = tf.layers.flatten(self.cache["a" + str(i - 1)])
                for _ in range(self.layer["num"][i]):
                    self.cache["a" + str(i)] = tf.layers.dense(a, n_y, activation=tf.nn.leaky_relu)
                    a = self.cache["a" + str(i)]

            elif name == "softmax" or name == "sigmoid":
                self.cache["z" + str(i)] = tf.layers.dense(self.cache["a" + str(i - 1)], n_y, activation=None)
                self.cache["logit"] = self.cache["z" + str(i)]
                self.cache["output"] = tf.nn.softmax(self.cache["logit"])
                self.cache["cross_entropy"] = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.cache["logit"],
                    labels=y)
                self.cache["cost"] = tf.reduce_mean(self.cache["cross_entropy"])

    def b_p(self):
        self.cache["optimizer"] = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.cache["cost"])


cnn = Cnn()
cnn.fit(X_train, Y_train, iter_num=700, learning_rate=0.005)
cnn.predict(X_test, Y_test)
