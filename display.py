import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import time
import getLabels
from keras.utils import to_categorical
train_images = []

# 读取输入文件
for each in range(1,4):
    #file = open('CC_out%s.txt' % str(each + 1))
    file = open('CC_out%s.txt' % str(each + 1))
    while 1:
        read_data = file.readline()
        if not read_data:
            break
        pass
        a = read_data.split(' ')
        tmp=[]
        for each in a[:-1]:
            cc=float(each)
            tmp.append(cc)
        train_images.append(tmp)
    file.close()
train_images = np.array(train_images)

train_labels=getLabels.load_train_labels('MNIST_data/train-labels.idx1-ubyte')[10000:49999]
train_labels=np.array(train_labels)
one_hot_train=to_categorical(train_labels)
print(one_hot_train[:10])

test_images = []

# 读取输入文件
file = open('CC_out6.txt')
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
        test_images.append(tmp)
file.close()

test_images = np.array(test_images)
test_labels=train_labels=getLabels.load_train_labels('MNIST_data/train-labels.idx1-ubyte')[50000:]
#test_labels=getLabels.load_train_labels('MNIST_data/t10k-labels.idx1-ubyte')
test_labels=np.array(test_labels)
one_hot_test=to_categorical(test_labels)
print(one_hot_test[:10])

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

 import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

    print(train_images[0:100])
    print(test_labels[:100])
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(4):
        img = train_images[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
