import pickle
import numpy as np
import  tensorflow as tf
from keras.utils import to_categorical
# ========数据处理=========

data_dir='/input/cifar10/cifar-10-batches-py/'
class_num=10
image_size=32
img_channels=3
# 读文件

"""
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


# 从读入的文件中获取图片数据(data)和标签信息(labels)
def load_data_one(file):
    batch = unpickle(file)
    data = batch['data']
    labels = batch['labels']
    print("Loading %s : img num %d." % (file, len(data)))
    return data, labels


# 将从文件中获取的信息进行处理，得到可以输入到神经网络中的数据。
def load_data(files, data_dir, label_count):
    global image_size, img_channels

    data, labels = load_data_one(data_dir + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)

    # 标签labels从0-9的数字转化为float类型(-1,10)的标签矩阵
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    # 将图片数据从(-1,3072)转化为(-1,3,32,32)
    #data = data.reshape([-1, img_channels, image_size, image_size])
    # 将(-1,3,32,32)转化为(-1,32,32,3)的图片标准输入
    #data = data.transpose([0, 2, 3, 1])

    # data数据归一化


    return data, labels


def prepare_data():
    print("======Loading data======")
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + 'batches.meta')
    print(meta)

    label_names = meta['label_names']

    # 依次读取data_batch_1-5的内容
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, class_num)
    test_data, test_labels = load_data(['test_batch'], data_dir, class_num)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    # 重新打乱训练集的顺序

    print("======数据准备结束======")

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
one_hot_test=to_categorical(test_labels)
train_labels = []
for each in range(5):
    with open('/input/cifar10/cifar-10-batches-py/data_batch_%s' % str(each + 1), mode='rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        train_labels += list(data_dict[b'labels'])
train_labels = np.array(train_labels)
one_hot_train=to_categorical(train_labels)
class CifarDate:
    def __init__(self, filename, labels, need_shuffle):
        data_length = int(len(labels))
        file_num = int(data_length / 10000)
        images_T = (np.empty([data_length, 3072])).astype(np.float32)  # 对numpy float64进行数据转换  python32
        xy = []
        for each_file in range(file_num):
            f = open(filename[each_file])
            for each_img in range(10000):
                for i in range(3):
                    data = f.readline()
                    count = 0
                    for each in data.split(' ')[:-1]:
                        each = float(each)
                        images_T[each_img + 10000 * each_file][count + 1024 * i] = each
                        count += 1
                tmp = np.reshape(images_T[each_img + 10000 * each_file], (3072))
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
train_filename = ['/input/cifar102/qk_1.txt', '/input/cifar102/qk_2.txt', '/input/cifar102/qk_3.txt',
                  '/input/cifar102/qk_4.txt', '/input/cifar102/qk_5.txt']
test_filename = ['/input/cifar102/qk_test.txt']
train = CifarDate(train_filename,one_hot_train, True)
test_data = CifarDate(test_filename, one_hot_test, False)


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
    'fc3' : weight_variable('fc3', [4096,10])
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
    'fb3' : bias_variable([10]),
}




if __name__ == '__main__':
    x=tf.placeholder(tf.float32,[None,3072])
    y_=tf.placeholder(tf.float32,[None,10])
    
    #x_image = tf.reshape(x, [-1, 3, 32, 32])
    #x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])
    x_image = tf.reshape(x,[-1,3,32,32])
    x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

    x_image_arr = tf.split(x_image,num_or_size_splits=64,axis=0)

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
    
    keep_prob=tf.placeholder(tf.float32)
    train_flag=tf.placeholder(tf.bool)
    learning_rate=tf.placeholder(tf.float32)

    #output=vgg_net(x_image,weights,biases,keep_prob)
    output=vgg_net(result_x_images,weights,biases,keep_prob)

    correct_prediction=tf.equal(tf.argmax(output,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    
      
    with tf.name_scope('loss') :
        cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=output))
        l2=tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])                           # 对所有训练参数加入l2正则化
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
       
        with tf.name_scope('train_op'):
            train_step=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999 ,epsilon=1e-8).minimize(cross_entropy+l2*5e-10)

    #train_x, train_y, test_x, test_y = prepare_data()
    batch_size=64
    steps=int(50000/64)
    epochs=100
    #train=CifarDate(train_x,train_y,need_shuffle=True)
    #test_data=CifarDate(test_x,test_y,need_shuffle=False)
    train_acc=[]
    train_loss=[]
    test_acc=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        alpha=0.001


        for ep in range(epochs):
            print("========================epcho: %d=============================" % ep)
            loss=0
            acc=0
            for i in range(steps):
                batch_x,batch_y=train.next_batch(batch_size)
                _, batch_loss=sess.run([train_step,cross_entropy],feed_dict={x:batch_x,y_:batch_y,keep_prob:0.5,train_flag:True,learning_rate:alpha})
                #print(batch_x[0])
                batch_acc=accuracy.eval(feed_dict={x:batch_x,y_:batch_y,keep_prob:0.5,train_flag:True,learning_rate:alpha})

                #loss+=batch_loss
                #acc+=batch_acc
                if i%100==0:
                    train_acc.append(batch_acc)
                    train_loss.append(batch_loss)
                    print("epcho: %d, iterations: %d, loss: %.4f, acc: %.4f" % (ep, i, batch_loss, batch_acc))

            #test_data=CifarDate(test_x,test_y,need_shuffle=False)
            test_data = CifarDate(test_filename, one_hot_test, False)
            for i in range(156):
                batch_x,batch_y=test_data.next_batch(64)
                loss_, acc_ = sess.run([cross_entropy, accuracy],
                               feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: False})
                loss += loss_
                acc += acc_
            test_acc.append(acc/156)
            print("epcho: %d,  loss: %.4f, acc: %.4f" % (ep,  loss/156, acc/156))
            
            
        print(test_acc)
        with open("acc2.txt", "w") as f:
            for i in range(len(train_loss)):
                f.write(str(train_loss[i]))
                f.write(" ")
                f.write(str(train_acc[i]))
                f.write(" ")
                f.write('\n')

