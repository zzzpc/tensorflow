import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.contrib.layers import xavier_initializer

test_labels = []
with open('/input/cifar10/cifar-10-batches-py/test_batch', mode='rb') as file:
    data_dict = pickle.load(file, encoding='bytes')
    test_labels += list(data_dict[b'labels'])

test_labels = np.array(test_labels)



train_labels = []
for each in range(5):
    with open('/input/cifar10/cifar-10-batches-py/data_batch_%s' % str(each + 1), mode='rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        train_labels += list(data_dict[b'labels'])
train_labels = np.array(train_labels)
class CifarDate:
    def __init__(self,filename,labels,need_shuffle):
        data_length=int(len(labels))
        file_num=int(data_length/10000)
        images_T = (np.empty([data_length, 3072])).astype(np.float32)  # 对numpy float64进行数据转换  python32
        xy = []
        for  each_file in range(file_num):
            f = open(filename[each_file])
            for each_img in range(10000):
                for i in range(3):
                    data = f.readline()
                    count = 0
                    for each in data.split(' ')[:-1]:
                        each = float(each)/256
                        images_T[each_img+10000*each_file][count + 1024 * i] = each
                        count += 1
                tmp = np.reshape(images_T[each_img+10000*each_file], (3072))
                xy.append((tmp))
            f.close()
        xy = np.array(xy)

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
train_filename=['/input/cifar10/qk9503_1.txt', '/input/cifar10/qk9503_2.txt', '/input/cifar10/qk9503_3.txt',
                      '/input/cifar10/qk9503_4.txt','/input/cifar10/qk9503_5.txt']

test_filename=['/input/cifar10/qk9503_test.txt']

train_data = CifarDate(train_filename,train_labels, True)
test_data = CifarDate(test_filename,test_labels, False)

"""
def load_data(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f,encoding='bytes')
        return data[b'data'],data[b'labels']

class CifarDate:
    def __init__(self,filenames,need_shuffle):
        all_data = []
        all_label = []
        for filename in filenames:
            data,labels = load_data(filename)
            all_data.append(data)
            all_label.append(labels)
        self._data = np.vstack(all_data)
        self._labels = np.hstack(all_label)

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
train_filenames=['/input/cifar10/cifar-10-batches-py/data_batch_1','/input/cifar10/cifar-10-batches-py/data_batch_2','/input/cifar10/cifar-10-batches-py/data_batch_3','/input/cifar10/cifar-10-batches-py/data_batch_4','/input/cifar10/cifar-10-batches-py/data_batch_5']

test_filenames=['/input/cifar10/cifar-10-batches-py/test_batch']

train_data = CifarDate(train_filenames, True)
test_data = CifarDate(test_filenames, False)
"""
batch_size = 20
x = tf.placeholder(tf.float32, [None, 3072])

y = tf.placeholder(tf.int64, [None])
x_image = tf.reshape(x,[-1,3,32,32])
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

x_image_arr = tf.split(x_image,num_or_size_splits=batch_size,axis=0)

result_x_image_arr = []

# 数据优化
for x_single_image in x_image_arr:
    x_single_image = tf.reshape(x_single_image,[32,32,3])
    #随机翻转
    data_aug_1 = tf.image.random_flip_left_right(x_single_image)
    #调整光照
    data_aug_2 = tf.image.random_brightness(data_aug_1,max_delta=63)
    #改变对比度
    data_aug_3 = tf.image.random_contrast(data_aug_2,lower=0.2,upper=1.8)
    #白化
    data_aug_4 = tf.image.per_image_standardization(data_aug_3)
    x_single_image = tf.reshape(data_aug_4,[1,32,32,3])
    result_x_image_arr.append(x_single_image)

# 拼接
result_x_images = tf.concat(result_x_image_arr,axis=0)

# 归一化
normal_result_x_images = result_x_images /255

# 卷积方法
def conv_wrapper(inputs,name,is_training,output_channel,kernel_size=(3,3),
                 activation=tf.nn.relu,padding='same'):
    with tf.name_scope(name):
        conv2d = tf.layers.conv2d(inputs,output_channel,kernel_size,padding=padding,
                                  activation=None,name=name + '/conv2d',
                                  kernel_initializer=tf.initializers.he_normal())
                                  #kernel_initializer=xavier_initializer())
        # 从截断的正态分布中输出随机值。
        # 生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
        bn = tf.layers.batch_normalization(conv2d,training=is_training)
        # 表示该网络当前是否正在训练，告知Batch
        # Normalization层是否应该更新或者使用均值或方差的分布信息
        return activation(bn)

# 池化方法
def pooling_wrapper(inputs,name):
    return tf.layers.max_pooling2d(inputs,(2,2),(2,2),name=name)

conv1_1 = conv_wrapper(normal_result_x_images,'conv1_1',True,64)
conv1_2 = conv_wrapper(conv1_1,'conv1_2',True,64)
conv1_3 = conv_wrapper(conv1_2,'conv1_3',True,64)
pooling1 = pooling_wrapper(conv1_3,'pool1')

conv2_1 = conv_wrapper(pooling1, 'conv2_1',True,128)
conv2_2 = conv_wrapper(conv2_1, 'conv2_2',True,128)
conv2_3 = conv_wrapper(conv2_2, 'conv2_3',True,128)
pooling2 = pooling_wrapper(conv2_3, 'pool2')

conv3_1 = conv_wrapper(pooling2, 'conv3_1',True,256)
conv3_2 = conv_wrapper(conv3_1, 'conv3_2',True,256)
conv3_3 = conv_wrapper(conv3_2, 'conv3_3',True,256)
pooling3 = pooling_wrapper(conv3_3, 'pool3')

# 展平
flatten = tf.layers.flatten(pooling3)

# 全连接
y_ = tf.layers.dense(flatten, 10)

# 代价
loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)
#loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_)
tf.add_to_collection(loss,tf.contrib.layers.l2_regularizer(loss))
pred = tf.argmax(y_,1)
corr = tf.equal(pred,y)
accur = tf.reduce_mean(tf.cast(corr,tf.float64))
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
# 5e-40



saver = tf.train.Saver()    #保存模型快照  参数


output_model_every_steps = 500
train_steps = 80000
test_steps = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    """
    if os.path.exists(model_path+'.index'):
        #saver.restore(sess,model_path)  #model_path存储的参数初始化sess
        print('model restored from %s' % model_path)
    else:
        print('model %s does not exist' % model_path)

    """
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val, _ = sess.run([loss, accur, train_op],
                                        feed_dict={x: batch_data,y: batch_labels})
        if (i + 1) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f'% (i + 1, loss_val, acc_val))

        if (i + 1) % 5000 == 0:
            test_data = CifarDate(test_filenames,test_labels, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accur],feed_dict={x: test_batch_data, y: test_batch_labels})
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %4.5f' % (i + 1, test_acc))
        
