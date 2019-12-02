import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#download mnist datasets
#55000 * 28 * 28 55000image
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('mnist_data',one_hot=True)#参数一：文件目录。参数二：是否为one_hot向量

#one_hot is encoding format
#None means tensor 的第一维度可以是任意维度
#/255. 做均一化
input_x=tf.placeholder(tf.float32,[None,28*28])/255.
#输出是一个one hot的向量
output_y=tf.placeholder(tf.int32,[None,10])

#输入层 [28*28*1]
input_x_images=tf.reshape(input_x,[-1,28,28,1])
#从(Test)数据集中选取3000个手写数字的图片和对应标签

test_x=mnist.test.images[:3000] #image
test_y=mnist.test.labels[:3000] #label

fig, ax = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(100):
    img = test_x[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
