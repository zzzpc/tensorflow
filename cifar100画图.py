import matplotlib.pyplot as plt
images_T2 = (np.empty([10000, 3072])).astype(np.float32)  # 对numpy float64进行数据转换  python32
tmp_test = []
for each_file in range(1):
    f = open('H:\cifar.txt' )
    for each_img in range(100):
        data = f.readline()
        count = 0
        for each in data.split(' ')[:-1]:
            each = float(each)
            images_T2[each_img + 10000 * each_file][count] = each
            count += 1
        tmp = np.reshape(images_T2[each_img+10000*each_file], (3, 32, 32))
        tmp_test.append((tmp))
    f.close()
test_img = np.array(tmp_test)


# print(test_labels[0])
print(len(test_img))

fig, ax = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(100):
    img = test_img[i].transpose(1, 2, 0)
    ax[i].imshow(img.astype(np.uint8), cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

