import numpy as np
from PIL import Image
import pickle
import os
import matplotlib.image as plimg
import time

def dtb(num):
    # 取整数部分
    integer = int(num)
    # 整数部分进制转换
    integercom = str(bin(integer))
    integercom = integercom[2:]
    tmpflo = []
    flocom = tmpflo
    return (integercom).zfill(8)

CHANNEL = 3
WIDTH = 32
HEIGHT = 32

data = []
labels = []
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

with open('G:\\python\\cifar_data\\cifar-10-batches-py\\test_batch', mode='rb') as file:
    # 数据集在当脚本前文件夹下
    data_dict = pickle.load(file, encoding='bytes')
    data += list(data_dict[b'data'])
    labels += list(data_dict[b'labels'])
img = np.reshape(data, [-1, CHANNEL, WIDTH, HEIGHT])

print(labels)

"""
images=(np.empty([10000,3072])).astype(np.float32)   #对numpy float64进行数据转换  python32
images_train=[]
f=open('cifar10_1out.txt')
for  each_img in range(10000):
    for i in range(3):
        data = f.readline()
        count=0
        for each in data.split(' ')[:-1]:
            each = float(each)
            images[each_img][count+1024*i]=each
            count+=1
    tmp=np.reshape(images[each_img], (3,32,32))
    images_train.append((tmp))
images_train=np.array(images_train)
    #images[each_img]=np.reshape( images[each_img],(1,3072))

f.close()



# 代码创建文件夹，也可以自行创建

f=open('cifar10.txt')
while 1:
    read_data=f.readline()
    if not read_data:
        break
    pass
    print(len(read_data),read_data[2:])



time1=time.time()
"""
f=open('cifar10_test.txt','r+', encoding='utf-8')

for i in range(10000):
    for j in range(3):
        for k in range(32):
            tmp=img[i][j][k]
            for r in range(32):
                bin_text = dtb(int(tmp[r]))
                f.write(bin_text)
        f.write("\n\t")

f.close()
time2=time.time()





