import numpy as np
import  tensorflow as tf
import  random
# ========鏁版嵁澶勭悊=========


# 璇绘枃浠�

import pickle  # 用于序列化和反序列化
import os
import cv2
import matplotlib.pyplot as plt

# data_dir = "/content/drive/My Drive/cifar-100-python"
data_dir = "/input/cifar100/cifar-100-python"

'''
字典形式的数据：
cifar100 data 文件的内容结构: 
    { 
    "data" : [(R,G,B, R,G,B ,....),(R,G,B, R,G,B, ...),...]    # 50000张图片，每张： 32 * 32 * 3
    "coarse_labels":[0,...,19],                         # 0~19 super category 
    "filenames":["volcano_s_000012.png",...],   # 文件名
    "batch_label":"", 
    "fine_labels":[0,1...99]          # 0~99 category 
    } 
'''


class Cifar100DataReader():
    def __init__(self, cifar_folder, batch_size, onehot=True):
        self.batch_size = batch_size
        self.cifar_folder = cifar_folder
        self.onehot = onehot
        self.data_label_train = None  # 训练集
        self.data_label_test = None  # 测试集
        self.batch_index = 0  # 训练数据的batch块索引
        self.test_batch_index = 0  # 测试数据的batch块索引
        f = os.path.join(self.cifar_folder, "train")  # 训练集有50000张图片，100个类，每个类500张
        # print ('read: %s'%f  )
        fo = open(f, 'rb')
        self.dic_train = pickle.load(fo, encoding='bytes')
        fo.close()
        self.data_label_train = list(zip(self.dic_train[b'data'], self.dic_train[b'fine_labels']))  # label 0~99
        np.random.shuffle(self.data_label_train)

    def dataInfo(self):
        print(self.data_label_train[0:2])  # 每个元素为二元组，第一个是numpy数组大小为32*32*3，第二是label
        print(self.dic_train.keys())
        print(b"coarse_labels:", len(self.dic_train[b"coarse_labels"]))
        print(b"filenames:", len(self.dic_train[b"filenames"]))
        print(b"batch_label:", len(self.dic_train[b"batch_label"]))
        print(b"fine_labels:", len(self.dic_train[b"fine_labels"]))
        print(b"data_shape:", np.shape((self.dic_train[b"data"])))
        print(b"data0:", type(self.dic_train[b"data"][0]))

    # 得到下一个batch训练集
    def next_train_data(self):
        """
        return list of numpy arrays [na,...,na] with specific batch_size
                na: N dimensional numpy array
        """
        if self.batch_index < len(self.data_label_train) / self.batch_size:
            # print ("batch_index:",self.batch_index  )
            datum = self.data_label_train[self.batch_index * self.batch_size:(self.batch_index + 1) * self.batch_size]
            self.batch_index += 1
            return self._decode(datum, self.onehot)
        else:
            self.batch_index = 0
            np.random.shuffle(self.data_label_train)
            datum = self.data_label_train[self.batch_index * self.batch_size:(self.batch_index + 1) * self.batch_size]
            self.batch_index += 1
            return self._decode(datum, self.onehot)

            # 把一个batch的训练数据转换为可以放入神经网络训练的数据

    def _decode(self, datum, onehot):
        rdata = list()  # batch训练数据
        rlabel = list()
        if onehot:
            for d, l in datum:
                print(l)
                rdata.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))  # 转变形状为：32*32*3
                hot = np.zeros(100)
                hot[int(l)] = 1  # label设为100维的one-hot向量
                rlabel.append(hot)
        else:
            for d, l in datum:
                rdata.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
                rlabel.append(int(l))
        return rdata, rlabel

        # 得到下一个测试数据 ，供神经网络计算模型误差用

    def next_test_data(self):
        '''''
        return list of numpy arrays [na,...,na] with specific batch_size
                na: N dimensional numpy array
        '''
        if self.data_label_test is None:
            f = os.path.join(self.cifar_folder, "test")
            # print ('read: %s'%f  )
            fo = open(f, 'rb')
            dic_test = pickle.load(fo, encoding='bytes')
            fo.close()
            data = dic_test[b'data']
            labels = dic_test[b'fine_labels']  # 0 ~ 99
            self.data_label_test = list(zip(data, labels))
            self.batch_index = 0

        if self.test_batch_index < len(self.data_label_test) / self.batch_size:
            # print ("test_batch_index:",self.test_batch_index )
            datum = self.data_label_test[
                    self.test_batch_index * self.batch_size:(self.test_batch_index + 1) * self.batch_size]
            self.test_batch_index += 1
            return self._decode(datum, self.onehot)
        else:
            self.test_batch_index = 0
            np.random.shuffle(self.data_label_test)
            datum = self.data_label_test[
                    self.test_batch_index * self.batch_size:(self.test_batch_index + 1) * self.batch_size]
            self.test_batch_index += 1
            return self._decode(datum, self.onehot)
    def next_test_data_ndb(self):
        '''''
        return list of numpy arrays [na,...,na] with specific batch_size
                na: N dimensional numpy array
        '''
        if self.data_label_test is None:
            f = os.path.join(self.cifar_folder, "test")
            # print ('read: %s'%f  )
            fo = open(f, 'rb')
            dic_test = pickle.load(fo, encoding='bytes')
            fo.close()
            images_T = (np.empty([10000, 3072])).astype(np.float32)  # 对numpy float64进行数据转换  python32
            xy = []
            f = open('/input/cifar100/cifar.txt')
            for each_img in range(10000):
                    data = f.readline()
                    count = 0
                    for each in data.split(' ')[:-1]:
                        each = float(each)
                        images_T[each_img ][count] = each
                        count += 1
                    tmp = np.reshape(images_T[each_img ], (32,32,3))
                    xy.append((tmp))
            f.close()



            data = xy
            labels = dic_test[b'fine_labels']  # 0 ~ 99
            self.data_label_test = list(zip(data, labels))
            self.batch_index = 0

        if self.test_batch_index < len(self.data_label_test) / self.batch_size:
            # print ("test_batch_index:",self.test_batch_index )
            datum = self.data_label_test[
                    self.test_batch_index * self.batch_size:(self.test_batch_index + 1) * self.batch_size]
            self.test_batch_index += 1
            return self._decode(datum, self.onehot)
        else:
            self.test_batch_index = 0
            np.random.shuffle(self.data_label_test)
            datum = self.data_label_test[
                    self.test_batch_index * self.batch_size:(self.test_batch_index + 1) * self.batch_size]
            self.test_batch_index += 1
            return self._decode(datum, self.onehot)


def resize_and_batch(images, labels):
    image_batch = list()
    label_batch = list()

    for i in range(len(images)):
        image = cv2.resize(images[i], (224, 224), cv2.INTER_NEAREST)
        label = np.argmax(labels[i])
        image_batch.append(image)
        label_batch.append(label)
    image_batch = np.array(image_batch)
    label_batch = np.array(label_batch)


    return image_batch, label_batch

def weight_variable(name,shape):
    initial=tf.keras.initializers.he_normal()   #采用he_normal() 初始化
    return tf.get_variable(name=name,shape=shape, initializer=initial) #当 reuse为TRUE是 tf.get_variable可以共享变量

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def batch_norm(input): #批归一化处理
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)

def conv(name,x,w,b):  #先卷积加偏置向量  再进行批归一化处理 最后经过激活函数
    return tf.nn.relu(batch_norm(tf.nn.bias_add(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME'),b)),name=name)

def max_pool(name,x,k):
    return tf.nn.max_pool(x,ksize=[1,k,k,1], strides=[1,k,k,1],padding='SAME',name=name)

def fc(name, x,w,b):
    return tf.nn.relu(batch_norm(tf.matmul(x,w)+b),name=name)

def vgg_net(_X, _weights,_biases,keep_prob):
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

    conv4_1 = conv('conv4_1', pool3, _weights['wc4_1'], _biases['bc4_1'])
    conv4_2 = conv('conv4_2', conv4_1, _weights['wc4_2'], _biases['bc4_2'])
    conv4_3 = conv('conv4_3', conv4_2, _weights['wc4_3'], _biases['bc4_3'])
    pool4 = max_pool('pool4', conv4_3, k=2)

    conv5_1 = conv('conv5_1', pool4, _weights['wc5_1'], _biases['bc5_1'])
    conv5_2 = conv('conv5_2', conv5_1, _weights['wc5_2'], _biases['bc5_2'])
    conv5_3 = conv('conv5_3', conv5_2, _weights['wc5_3'], _biases['bc5_3'])
    pool5 = max_pool('pool5', conv5_3, k=1)

    _shape=pool5.get_shape()
    flatten=_shape[1].value*_shape[2].value*_shape[3].value
    pool5=tf.reshape(pool5,shape=[-1,flatten])

    fc1=fc('fc1',pool5,_weights['fc1'],_biases['fb1'])
    fc1=tf.nn.dropout(fc1,keep_prob)

    fc2 = fc('fc2', fc1, _weights['fc2'], _biases['fb2'])
    fc2 = tf.nn.dropout(fc2, keep_prob)

    output= fc('fc3', fc2, _weights['fc3'], _biases['fb3'])

    return output

weights={  #字典类型
    'wc1_1' : weight_variable('wc1_1', [3,3,3,64]),
    'wc1_2' : weight_variable('wc1_2', [3,3,64,64]),
    'wc2_1' : weight_variable('wc2_1', [3,3,64,128]),
    'wc2_2' : weight_variable('wc2_2', [3,3,128,128]),
    'wc3_1' : weight_variable('wc3_1', [3,3,128,256]),
    'wc3_2' : weight_variable('wc3_2', [3,3,256,256]),
    'wc3_3' : weight_variable('wc3_3', [3,3,256,256]),
    'wc4_1' : weight_variable('wc4_1', [3,3,256,512]),
    'wc4_2' : weight_variable('wc4_2', [3,3,512,512]),
    'wc4_3' : weight_variable('wc4_3', [3,3,512,512]),
    'wc5_1' : weight_variable('wc5_1', [3,3,512,512]),
    'wc5_2' : weight_variable('wc5_2', [3,3,512,512]),
    'wc5_3' : weight_variable('wc5_3', [3,3,512,512]),
    'fc1' : weight_variable('fc1', [2*2*512,4096]),
    'fc2' : weight_variable('fc2', [4096,4096]),
    'fc3' : weight_variable('fc3', [4096,100])
}

biases={
    'bc1_1' : bias_variable([64]),
    'bc1_2' : bias_variable([64]),
    'bc2_1' : bias_variable([128]),
    'bc2_2' : bias_variable([128]),
    'bc3_1' : bias_variable([256]),
    'bc3_2' : bias_variable([256]),
    'bc3_3' : bias_variable([256]),
    'bc4_1' : bias_variable([512]),
    'bc4_2' : bias_variable([512]),
    'bc4_3' : bias_variable([512]),
    'bc5_1' : bias_variable([512]),
    'bc5_2' : bias_variable([512]),
    'bc5_3' : bias_variable([512]),
    'fb1' : bias_variable([4096]),
    'fb2' : bias_variable([4096]),
    'fb3' : bias_variable([100]),
}




if __name__ == '__main__':
    x=tf.placeholder(tf.float32,[None,32,32,3])
    y_input = tf.placeholder(tf.int64, [None, ])
    y_input_one_hot =tf.one_hot(y_input, 100)
    keep_prob=tf.placeholder(tf.float32)
    train_flag=tf.placeholder(tf.bool)
    learning_rate=tf.placeholder(tf.float32)

    output=vgg_net(x,weights,biases,keep_prob)

    logit_softmax = tf.nn.softmax(output)
    correct_prediction=tf.equal(tf.argmax(logit_softmax,1),y_input)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.name_scope('loss') :
        cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_input_one_hot,logits=output))
        l2=tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])                           # 瀵规墍鏈夎缁冨弬鏁板姞鍏2姝ｅ垯鍖�

    with tf.name_scope('train_op'):
        train_step=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999 ,epsilon=1e-8).minimize(cross_entropy+l2*5e-10)

    batch_size = 64
    steps = int(50000 / 64)
    epochs = 40
    data=Cifar100DataReader(data_dir ,64,onehot=False)

    train_acc = []
    train_loss = []
    test_acc = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        alpha = 0.001

        for ep in range(epochs):
            print("========================epcho: %d=============================" % ep)
            loss = 0
            acc = 0
            for i in range(steps):
                batch_x, batch_y = data.next_train_data()
                #print(batch_y)
               
                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_x, y_input: batch_y, keep_prob: 0.5, train_flag: True,
                                                    learning_rate: alpha})
                # print(batch_x[0])
                batch_acc = accuracy.eval(
                    feed_dict={x: batch_x, y_input: batch_y, keep_prob: 0.5, train_flag: True, learning_rate: alpha})
                train_acc.append(batch_acc)
                train_loss.append(batch_loss)
                # loss+=batch_loss
                # acc+=batch_acc
                if i % 100 == 0:
                    print("epcho: %d, iterations: %d, loss: %.4f, acc: %.4f" % (ep, i, batch_loss, batch_acc))

            # test_data=CifarDate(test_x,test_y,need_shuffle=False)
            #test_data = CifarDate(test_filename, one_hot_test, False)
            for i in range(500):
                batch_x, batch_y = data. next_test_data()
                loss_, acc_ = sess.run([cross_entropy, accuracy],
                                       feed_dict={x: batch_x, y_input: batch_y, keep_prob: 1.0, train_flag: False})
                loss += loss_
                acc += acc_
            test_acc.append(acc / 156)
            print("epcho: %d,  loss: %.4f, acc: %.4f" % (ep, loss / 500, acc / 500))

        print(train_acc, train_loss, test_acc)


