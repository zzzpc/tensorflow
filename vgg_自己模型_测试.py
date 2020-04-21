import pickle
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

# ========鏁版嵁澶勭悊=========

data_dir = '/input/cifar10/cifar-10-batches-py/'
class_num = 10
image_size = 32
img_channels = 3


"""
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch['data']
    labels = batch['labels']
    print("Loading %s : img num %d." % (file, len(data)))
    return data, labels



def load_data(files, data_dir, label_count):
    global image_size, img_channels

    data, labels = load_data_one(data_dir + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)

    
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    
    #data = data.reshape([-1, img_channels, image_size, image_size])
    
    #data = data.transpose([0, 2, 3, 1])

   

    return data, labels


def prepare_data():
    print("======Loading data======")
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + 'batches.meta')
    print(meta)

    label_names = meta['label_names']

    
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, class_num)
    test_data, test_labels = load_data(['test_batch'], data_dir, class_num)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    
    print("======预处理准备就绪======")

    return train_data, train_labels, test_data, test_labels



class CifarDate:
    def __init__(self,data,labels,need_shuffle):


        self._data = data
        self._labels = labels

        self.start = 0
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        if self._need_shuffle:
            self._shuffle_data()
    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]
    def next_batch(self,batch_size):
        end = self.start + batch_size
        if end > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self.start = 0
                end = batch_size
            else:
                raise Exception('have no more examples')
        if end > self._num_examples:
            raise Exception('batch size is larger than all examplts')
        batch_data = self._data[self.start:end]
        batch_labels = self._labels[self.start:end]
        self.start = end
        return batch_data,batch_labels

"""

test_labels = []
with open('/input/cifar10/cifar-10-batches-py/test_batch', mode='rb') as file:
    data_dict = pickle.load(file, encoding='bytes')
    test_labels += list(data_dict[b'labels'])
test_labels = np.array(test_labels)
one_hot_test = to_categorical(test_labels)
train_labels = []
for each in range(5):
    with open('/input/cifar10/cifar-10-batches-py/data_batch_%s' % str(each + 1), mode='rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        train_labels += list(data_dict[b'labels'])
train_labels = np.array(train_labels)
one_hot_train = to_categorical(train_labels)

train_filename = ['/input/cifar10/qk_03_1.txt', '/input/cifar10/qk_03_2.txt', '/input/cifar10/qk_03_3.txt',
                  '/input/cifar10/qk_03_4.txt', '/input/cifar10/qk_03_5.txt']

images_T = (np.empty([50000, 3072])).astype(np.float32)
train_images = []
for each_file in range(5):
    f = open(train_filename[each_file])
    for each_img in range(10000):
        for i in range(3):
            data = f.readline()
            count = 0
            for each in data.split(' ')[:-1]:
                each = float(each)
                images_T[each_img + 10000 * each_file][count + 1024 * i] = each
                count += 1
        tmp = np.reshape(images_T[each_img + 10000 * each_file], (3072))
        train_images.append((tmp))
    f.close()
train_images = np.array(train_images)

test_filename = ['/input/cifar10/qk_03_test.txt']
images_Te = (np.empty([10000, 3072])).astype(np.float32)  # 瀵筺umpy float64杩涜鏁版嵁杞崲  python32
test_images = []
for each_file in range(1):
    f = open(test_filename[each_file])
    for each_img in range(10000):
        for i in range(3):
            data = f.readline()
            count = 0
            for each in data.split(' ')[:-1]:
                each = float(each)
                images_Te[each_img + 10000 * each_file][count + 1024 * i] = each
                count += 1
        tmp = np.reshape(images_Te[each_img + 10000 * each_file], (3072))
        test_images.append((tmp))
    f.close()
test_images = np.array(test_images)


class CifarDate:
    def __init__(self, xy, labels, need_shuffle):

        self._data = xy
        self._labels = labels
        self.start = 0
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        end = self.start + batch_size
        if end > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self.start = 0
                end = batch_size
            else:
                raise Exception('have no more examples')
        if end > self._num_examples:
            raise Exception('batch size is larger than all examplts')
        batch_data = self._data[self.start:end]
        batch_labels = self._labels[self.start:end]
        self.start = end
        return batch_data, batch_labels


train = CifarDate(train_images, one_hot_train, True)
test_data = CifarDate(test_images, one_hot_test, False)


def weight_variable(name, shape):
    initial = tf.keras.initializers.he_normal()  # 閲囩敤he_normal() 鍒濆鍖�
    return tf.get_variable(name=name, shape=shape, initializer=initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_norm(input):  # 鎵瑰綊涓€鍖栧鐞�
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)


def conv(name, x, w, b):
    return tf.nn.relu(batch_norm(tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME'), b)),
                      name=name)


def max_pool(name, x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def fc(name, x, w, b):
    return tf.nn.relu(batch_norm(tf.matmul(x, w) + b), name=name)


def vgg_net(_X, _weights, _biases, keep_prob):
    conv1_1 = conv('conv1_1', _X, _weights['wc1_1'], _biases['bc1_1'])
    conv1_2 = conv('conv1_2', conv1_1, _weights['wc1_2'], _biases['bc1_2'])
    pool1 = max_pool('pool1', conv1_2, k=2)

    conv2_1 = conv('conv2_1', pool1, _weights['wc2_1'], _biases['bc2_1'])
    conv2_2 = conv('conv2_2', conv2_1, _weights['wc2_2'], _biases['bc2_2'])
    pool2 = max_pool('pool2', conv2_2, k=2)

    conv3_1 = conv('conv3_1', pool2, _weights['wc3_1'], _biases['bc3_1'])
    conv3_2 = conv('conv3_2', conv3_1, _weights['wc3_2'], _biases['bc3_2'])
    conv3_3 = conv('conv3_3', conv3_2, _weights['wc3_3'], _biases['bc3_3'])
    pool3 = max_pool('pool3', conv3_3, k=2)

    _shape = pool3.get_shape()
    flatten = _shape[1].value * _shape[2].value * _shape[3].value
    pool3 = tf.reshape(pool3, shape=[-1, flatten])

    fc1 = fc('fc1', pool3, _weights['fc1'], _biases['fb1'])
    fc1 = tf.nn.dropout(fc1, keep_prob)

    fc2 = fc('fc2', fc1, _weights['fc2'], _biases['fb2'])
    fc2 = tf.nn.dropout(fc2, keep_prob)

    output = fc('fc3', fc2, _weights['fc3'], _biases['fb3'])

    return output


weights = {  # 瀛楀吀绫诲瀷
    'wc1_1': weight_variable('wc1_1', [3, 3, 3, 64]),
    'wc1_2': weight_variable('wc1_2', [3, 3, 64, 64]),
    'wc2_1': weight_variable('wc2_1', [3, 3, 64, 128]),
    'wc2_2': weight_variable('wc2_2', [3, 3, 128, 128]),
    'wc3_1': weight_variable('wc3_1', [3, 3, 128, 256]),
    'wc3_2': weight_variable('wc3_2', [3, 3, 256, 256]),
    'wc3_3': weight_variable('wc3_3', [3, 3, 256, 256]),
    'fc1': weight_variable('fc1', [4 * 4 * 256, 1024]),
    'fc2': weight_variable('fc2', [1024, 1024]),
    'fc3': weight_variable('fc3', [1024, 10])
}

biases = {
    'bc1_1': bias_variable([64]),
    'bc1_2': bias_variable([64]),
    'bc2_1': bias_variable([128]),
    'bc2_2': bias_variable([128]),
    'bc3_1': bias_variable([256]),
    'bc3_2': bias_variable([256]),
    'bc3_3': bias_variable([256]),

    'fb1': bias_variable([1024]),
    'fb2': bias_variable([1024]),
    'fb3': bias_variable([10]),
}


def data_aug_train():
    print('===============training==================')
    x_image = tf.reshape(x, [-1, 3, 32, 32])
    x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

    x_image_arr = tf.split(x_image, num_or_size_splits=64, axis=0)

    result_x_image_arr = []

    # 数据优化
    for x_single_image in x_image_arr:
        x_single_image = tf.reshape(x_single_image, [32, 32, 3])

        data_aug_1 = tf.image.random_flip_left_right(x_single_image)
        # 调整光照
        data_aug_2 = tf.image.random_brightness(data_aug_1, max_delta=63)
        # 改变对比度
        data_aug_3 = tf.image.random_contrast(data_aug_2, lower=0.2, upper=1.8)
        # 白化
        data_aug_4 = tf.image.per_image_standardization(data_aug_3)
        x_single_image = tf.reshape(data_aug_4, [1, 32, 32, 3])
        result_x_image_arr.append(x_single_image)

    result_x_images = tf.concat(result_x_image_arr, axis=0)
    return result_x_images


def data_aug_test():
    print('----------------------test--------------')
    x_image = tf.reshape(x, [-1, 3, 32, 32])
    x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

    x_image_arr = tf.split(x_image, num_or_size_splits=64, axis=0)

    result_x_image_arr = []

    # 数据优化
    for x_single_image in x_image_arr:
        x_single_image = tf.reshape(x_single_image, [32, 32, 3])

        x_single_image = tf.reshape(x_single_image, [1, 32, 32, 3])
        result_x_image_arr.append(x_single_image)

    result_x_images = tf.concat(result_x_image_arr, axis=0)
    return result_x_images

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 3072])
    y_ = tf.placeholder(tf.float32, [None, 10])



    global_step = tf.Variable(0, name='global_step', trainable=False)
    add_global = global_step.assign_add(1)
    boundaries = [100]
    learing_rates = [0.0001, 0.0001]
    lr = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learing_rates)

    keep_prob = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32)

    result_x_images_test = data_aug_test()

    # output=vgg_net(x_image,weights,biases,keep_prob)
    output = vgg_net(result_x_images_test, weights, biases, keep_prob)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=output))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])  # 瀵规墍鏈夎缁冨弬鏁板姞鍏2姝ｅ垯鍖�
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        with tf.name_scope('train_op'):
            train_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(
                cross_entropy + l2 * 5e-10)

    #train_x, train_y, test_x, test_y = prepare_data()
    batch_size = 64
    steps = int(50000 / 64)
    epochs = 100
    # train = CifarDate(train_x, train_y, need_shuffle=True)

    train_acc = []
    train_loss = []
    test_acc = []

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "/data/model_o.ckpt")
        # sess.run(tf.global_variables_initializer())
        for ep in range(epochs):
            acc = 0
            # test_data = CifarDate(test_x, test_y, need_shuffle=False)
            test_data = CifarDate(test_filename, one_hot_test, False)
            for i in range(156):
                batch_x, batch_y = test_data.next_batch(64)

                acc_test = sess.run([accuracy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: False})

                acc += acc_test[0]
            test_acc.append(acc / 156)
            flag = 0
            print("epcho: %d,   acc: %.4f" % (ep, acc / 156))
            print("========================epcho: %d=============================" % ep)
            loss = 0

