# -*- coding: utf-8 -*-
"""
@author: ispurs
cifar10 数据可视化
label:
      0 airplane
      1 automobile
      2 bird
      3 cat
      4 deer
      5 dog
      6 frog
      7 horse
      8 ship
      9 truck
"""
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import numpy as np
import matplotlib.pyplot as plt
import pickle
filename = 'G:\\python\\cifar_data\\cifar-10-batches-py\\data_batch_4'  # cifar10二进制文件路径

def unpickle(file):
	with open(file,'rb')as fo:
		dict=pickle.load(fo, encoding='iso-8859-1')
	return dict
cifar10_data=unpickle(filename)

image = cifar10_data['data']
labels=cifar10_data['labels']
fig, ax = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(100):
    image0 = image[i+9900]
    pic = image0.reshape(3, 32, 32)
    pic = pic.transpose(1, 2, 0)
    ax[i].imshow(pic, cmap='Greys', interpolation='nearest')
    print(classes[labels[i]], i)
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()




"""
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import pickle


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

images_T1 = (np.empty([10000, 3072])).astype(np.float32)  # 对numpy float64进行数据转换  python32
xy1 = []
f = open('cifar10_1out.txt' )
for each_img in range(10000):
    for i in range(3):
        data = f.readline()
        count = 0
        for each in data.split(' ')[:-1]:
            each = float(each)
            images_T1[each_img][count + 1024 * i] = each
            count += 1
    tmp = np.reshape(images_T1[each_img], (3, 32, 32))
    xy1.append((tmp))
f.close()
xy1 = np.array(xy1)


images_T2 = (np.empty([10000, 3072])).astype(np.float32)  # 对numpy float64进行数据转换  python32
xy2 = []
f = open('cifar10_2out.txt' )
for each_img in range(10000):
    for i in range(3):
        data = f.readline()
        count = 0
        for each in data.split(' ')[:-1]:
            each = float(each)
            images_T2[each_img][count + 1024 * i] = each
            count += 1
    tmp = np.reshape(images_T2[each_img], (3, 32, 32))
    xy2.append((tmp))
f.close()
xy2 = np.array(xy2)

images_T3 = (np.empty([10000, 3072])).astype(np.float32)  # 对numpy float64进行数据转换  python32
xy3 = []
f = open('cifar10_3out.txt' )
for each_img in range(10000):
    for i in range(3):
        data = f.readline()
        count = 0
        for each in data.split(' ')[:-1]:
            each = float(each)
            images_T3[each_img][count + 1024 * i] = each
            count += 1
    tmp = np.reshape(images_T3[each_img], (3, 32, 32))
    xy3.append((tmp))
f.close()
xy3 = np.array(xy3)

images_T4 = (np.empty([10000, 3072])).astype(np.float32)  # 对numpy float64进行数据转换  python32
xy4 = []
f = open('cifar10_4out.txt' )
for each_img in range(10000):
    for i in range(3):
        data = f.readline()
        count = 0
        for each in data.split(' ')[:-1]:
            each = float(each)
            images_T4[each_img][count + 1024 * i] = each
            count += 1
    tmp = np.reshape(images_T4[each_img], (3, 32, 32))
    xy4.append((tmp))
f.close()
xy4 = np.array(xy4)

images_T5 = (np.empty([10000, 3072])).astype(np.float32)  # 对numpy float64进行数据转换  python32
xy5 = []
f = open('cifar10_5out.txt' )
for each_img in range(10000):
    for i in range(3):
        data = f.readline()
        count = 0
        for each in data.split(' ')[:-1]:
            each = float(each)
            images_T5[each_img][count + 1024 * i] = each
            count += 1
    tmp = np.reshape(images_T5[each_img], (3, 32, 32))
    xy5.append((tmp))
f.close()
xy5 = np.array(xy5)

train=np.concatenate((xy1,xy2),axis=0)
train1=np.concatenate((train,xy3),axis=0)
train2=np.concatenate((train1,xy4),axis=0)
train3=np.concatenate((train2,xy5),axis=0)

print(len(train3))
fig, ax = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all')
ax = ax.flatten()

train_labels=[]
for each in range(5):
    with open('G:\\python\\cifar_data\\cifar-10-batches-py\\data_batch_%s' %str(each+1), mode='rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        train_labels += list(data_dict[b'labels'])
train_labels=np.array(train_labels)


for i in range(100):
        img =train3[i+49900].transpose(1, 2, 0)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        print(classes[train_labels[i]],i)
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
"""
