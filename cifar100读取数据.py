import os
import numpy as np
import  matplotlib.pyplot as plt
# 给我个路径我要创建文件呢


# 解压数据呢
def unpickle(file):
    import _pickle
    with open(file, 'rb') as fo:
        dict = _pickle.load(fo, encoding='bytes')
    return dict

# 给定路径添加数据呢
def Dealdata(meta, photo):
    
    data=[]
    labels=[]
    for aa, bb, c, d in zip( photo[b'filenames'],  photo[b'fine_labels'],  photo[b'coarse_labels'],  photo[b'data']):
        labels.append(meta[b'fine_label_names'][bb])
        data.append(d)
    return data ,labels

if __name__ == '__main__':
    metapath = 'H:\cifar100\cifar-100-python\meta'
    # 解压后test的路径
    testpath = 'H:\cifar100\cifar-100-python\\test'
    # 解压后train的路径
    trainpath = 'H:\cifar100\cifar-100-python\\train'
    a = unpickle(metapath)
    b = unpickle(testpath)
    c = unpickle(trainpath)
    train, train_labels=Dealdata(a, b)
    test,  test_labels=Dealdata(a, c)

    test_img=np.array(train)
    fig, ax = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(100):
        img = np.reshape(test_img[i],(3,32,32)).transpose(1, 2, 0)
        ax[i].imshow(img.astype(np.uint8), cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()




