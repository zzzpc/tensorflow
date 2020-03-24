import pickle
import numpy as np

# ========数据处理=========

data_dir='G:\python\cifar_data\cifar-10-batches-py/'
class_num=10
image_size=32
img_channels=3
# 读文件
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
    #labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    # 将图片数据从(-1,3072)转化为(-1,3,32,32)
    data = data.reshape([-1, img_channels, image_size, image_size])
    # 将(-1,3,32,32)转化为(-1,32,32,3)的图片标准输入
    data = data.transpose([0, 2, 3, 1])

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


train_x, train_y, test_x, test_y = prepare_data()
print(train_x[0],train_y[0],test_x[0],test_y[0])
