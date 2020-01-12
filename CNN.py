import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import matplotlib.pyplot as plt
import time
import getLabels
from keras.utils import to_categorical
train_images= np.empty(shape=[10000, 784])
for k in range(6):
    file = open('cc_out%s.txt' %str(k+1))
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

"""
train_images2 = []
file = open('cc_out2.txt')
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
    train_images2.append(tmp)
file.close()
train_images2 = np.array(train_images2)
train_images=np.concatenate((train_images1,train_images2),axis=0)

train_images3 = []
file = open('cc_out3.txt')
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
    train_images3.append(tmp)
file.close()
train_images3 = np.array(train_images3)
train_images=np.concatenate((train_images,train_images3),axis=0)

train_images4 = []
file = open('cc_out4.txt')
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
    train_images4.append(tmp)
file.close()
train_images4 = np.array(train_images4)
train_images=np.concatenate((train_images,train_images4),axis=0)

train_images5 = []
file = open('cc_out5.txt')
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
    train_images5.append(tmp)
file.close()
train_images5 = np.array(train_images5)
train_images=np.concatenate((train_images,train_images5),axis=0)


train_images6 = []
file = open('cc_out6.txt')
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
    train_images6.append(tmp)
file.close()
train_images6 = np.array(train_images6)
train_images=np.concatenate((train_images,train_images6),axis=0)
"""

train_images = np.array(train_images[10000:])


train_labels=getLabels.load_train_labels('MNIST_data/train-labels.idx1-ubyte')
train_labels=np.array(train_labels)
one_hot_train=to_categorical(train_labels)

"""
train_labels=getLabels.load_train_labels('MNIST_data/train-labels.idx1-ubyte')[0:10000]

train_labels=np.array(train_labels)
one_hot_train=to_categorical(train_labels)
"""

test_images = []
test_train_images=getLabels.load_train_images('MNIST_data/t10k-images.idx3-ubyte')
# 读取输入文件
file = open('ss.txt')
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
test_labels=getLabels.load_train_labels('MNIST_data/t10k-labels.idx1-ubyte')
#test_labels=getLabels.load_train_labels('MNIST_data/t10k-labels.idx1-ubyte')
test_labels=np.array(test_labels)

one_hot_test=to_categorical(test_labels)



class Dataset:
    def __init__(self,data,label):
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

    def next_batch(self,batch_size,shuffle = True):
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
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch
            data_new_part =  self._data[start:end]
            label_new_part =  self._label[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._label[start:end]







train = Dataset(train_images[:-1], one_hot_train)
test = Dataset(test_images, one_hot_test)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(type(train_images),train_images,len(train_images),len(train_labels))
# 设置批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size


# 定义初始化权值函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义初始化偏置函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(value):
    return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 输入层
# 定义两个placeholder
sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')



x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()
for i in range(1000):
    batch = train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x:test_images, y_: one_hot_test, keep_prob: 1.0}))





"""
if __name__ == '__main__':


    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

   # print(test_images[0:100])

    print(train_labels[39900:])

    #print(test_labels[0])
    print(len(train_images))
    print(len(train_labels))
    fig, ax = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(100):
        img =np.array(train_images[i+39900]).reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
"""
