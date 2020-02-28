
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt  # 加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA  # 加载PCA算法包
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import getLabels
from keras.utils import to_categorical

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
"""
train_images = np.empty(shape=[10000, 784])
for k in range(6):
    file = open('G:\python\MNIST_data\cc_outnew%s.txt' % str(k + 1))
    train_images_tmp = []
    while 1:
        read_data = file.readline()
        if not read_data:
            break
        pass
        a = read_data.split(' ')
        tmp = []
        for each in a[:-1]:
            cc = float(each)
            tmp.append(cc)
        train_images_tmp.append(tmp)
    file.close()
    train_images_tmp = np.array(train_images_tmp)
    print(train_images_tmp.shape)
    train_images = np.concatenate((train_images, train_images_tmp), axis=0)

train_images = np.array(train_images[10000:])

train_labels = getLabels.load_train_labels('MNIST_data/train-labels.idx1-ubyte')
train_labels = np.array(train_labels)
one_hot_train = to_categorical(train_labels)

test_images = []
test_train_images = getLabels.load_train_images('MNIST_data/t10k-images.idx3-ubyte')
# 读取输入文件
file = open('G:\python\MNIST_data\cc_outnew_test.txt')
while 1:
    read_data = file.readline()
    if not read_data:
        break
    pass
    a = read_data.split(' ')
    tmp = []
    for each in a[:-1]:
        cc = float(each)
        tmp.append(cc)
    test_images.append(np.array(tmp))
file.close()

test_images = np.array(test_images)
test_labels = getLabels.load_train_labels('MNIST_data/t10k-labels.idx1-ubyte')
# test_labels=getLabels.load_train_labels('MNIST_data/t10k-labels.idx1-ubyte')
test_labels = np.array(test_labels)

one_hot_test = to_categorical(test_labels)
"""
train_images=mnist.train.images
test_images=mnist.test.images
# 3.1.创建pca对象
pca = PCA(n_components=60)
pca.fit(train_images)
x_train_reduction = pca.transform(train_images)
x_test_reduction = pca.transform(test_images)


class Dataset:
    def __init__(self, data, label):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._label = label
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes

            np.random.shuffle(idx)  # shuffle indexe 打乱顺序

            self._data = self.data[idx]  # get list of `num` random samples
            self._label = self.label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            label_rest_part = self.label[start:self._num_examples]

            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            self._label = self.label[idx0]  # get list of `num` random samples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            label_new_part = self._label[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate(
                (label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._label[start:end]


train = Dataset(x_train_reduction, mnist.train.labels)

test = Dataset(x_test_reduction, mnist.test.labels)

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

if __name__ == "__main__":
    # 获取所有的数据集

    # 创建一个交互式的会话
    sess = tf.InteractiveSession()
    # 定义神经网络的参数
    in_units = 60
    h1_units = 1000
    w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units], dtype=tf.float32))
    w2 = tf.Variable(tf.truncated_normal([h1_units, 10], stddev=0.1), dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([10], dtype=tf.float32))
    # 定义输入变量

    global_step = tf.Variable(0, name='global_step', trainable=False)
    add_global = global_step.assign_add(1)
    boundaries = [2, 4, 6, 8, 10]
    learing_rates = [0.1, 0.0952, 0.0856, .0664, 0.0568, 0.052]
    lr = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learing_rates)
    x = tf.placeholder(dtype=tf.float32, shape=[None, in_units])
    # 定义dropout保留的节点数量
    keep_prob = tf.placeholder(dtype=tf.float32)
    # 定义学习率变量

    # 定义前向传播过程
    h1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
    # 使用dropout
    h1_drop = tf.nn.dropout(h1, keep_prob)
    # 定义输出y
    y = tf.nn.softmax(tf.matmul(h1_drop, w2) + b2)
    # 定义输出变量
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    # 定义损失函数
    loss_func = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdagradOptimizer(lr).minimize(loss_func)

    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    # 初始化变量
    epoch = 200
    batch = 100
    saver = tf.train.Saver()


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, 'G:\\python\\log_files\\train\\save_net.ckpt')
        for step in range(epoch):
            ra = sess.run(lr)
            for i in range(600):
                batch_xs, batch_ys = train.next_batch(100)
                train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                if i % 100 == 0:
                    print("train accuracy :", accuracy.eval({x: batch_xs, y_: batch_ys, keep_prob: 1.0}),
                          "当前学习率为%f:" % ra)
            # 计算准确率
            test_acc = accuracy.eval({x: x_test_reduction, y_: mnist.test.labels, keep_prob: 1.0}, session=sess)
            g = sess.run(add_global)
            print("第%d个epoch训练完毕 测试精确度未%.4f ，马上进入第%d个epoch的训练" % (step, test_acc, g))
            save_path = saver.save(sess, "G:\\python\\log_files\\train\\save_net.ckpt")

        print("Save to path: ", save_path)



    # 训练集上的准确率

    print("test accuracy:", accuracy.eval({x: x_test_reduction, y_: mnist.test.labels, keep_prob: 1.0}))
