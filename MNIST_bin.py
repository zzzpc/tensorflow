from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
import struct
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x=mnist.train.images[:1000]


accuracy = 16 #  小数部分精度
def dtb(num):
        #取整数部分
        integer = int(num)
        #取小数部分
        flo = num - integer
        #整数部分进制转换
        integercom = str(bin(integer))
        integercom = integercom[2:]
        #小数部分进制转换
        tem = flo
        tmpflo = []
        for i in range(accuracy):
            tem *= 2
            tmpflo += str(int(tem))
            tem -= int(tem)
        flocom = tmpflo
        return integercom + '.' + ''.join(flocom)


time1=time.time()
f=open('cc.txt','r+', encoding='utf-8')
for i in range(len(x)):
    tt=x[i].tolist()
    for j in range(len(x[i])):
        bin_text=dtb(x[i][j])
        f.write(bin_text[2:])
    f.write("\n")
f.close() 
time2=time.time()
print("加密时间为:%.2f"%(time2-time1) ,"s")
