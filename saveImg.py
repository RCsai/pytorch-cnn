import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
from tqdm import tqdm
import cv2
from PIL import Image
from torchvision.datasets.mnist import read_image_file
from  torchvision.datasets import mnist
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as CM


def saveImg():
    trainSet = datasets.MNIST(root='pymnist',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    testSet = datasets.MNIST(root='pymnist',
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)

    train_loader = DataLoader(dataset=trainSet)

    test_loader = DataLoader(dataset=testSet)

    for i,(images,labels) in enumerate(tqdm(train_loader,ncols=50)):
        if i==0:
            images=images.numpy()
            images=images.reshape(28,28)
            print(type(images))
            print(images.shape)
            img=cv2.imread(images)
            cv2.imshow("img",img)
            cv2.waitKey()


def load_data(data_path):
    '''
        函数功能：导出MNIST数据
        输入: data_path   传入数据所在路径（解压后的数据）
        输出: train_data  输出data
             train_label  输出label
        '''

    f_data = open(os.path.join(data_path, 'train-images-idx3-ubyte'))
    loaded_data = np.fromfile(file=f_data, dtype=np.uint8)
    # 前16个字符为说明符，需要跳过
    train_data = loaded_data[16:].reshape((-1, 784)).astype(np.float)

    f_label = open(os.path.join(data_path, 'train-labels-idx1-ubyte'))
    loaded_label = np.fromfile(file=f_label, dtype=np.uint8)
    # 前8个字符为说明符，需要跳过
    train_label = loaded_label[8:].reshape((-1)).astype(np.float)

    return train_data, train_label

def showImg():
    root="./mnist"
    train_data,train_label=load_data(root)
    print(train_data.shape)
    print(train_label.shape,'\n')

    for i in range(1,2):
        img=train_data[i].reshape(28,28)
        label=int(train_label[i])

        # 多行显示
        plt.subplot(5,10,i)
        plt.imshow(img,cmap=CM.gray)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("label:"+str(label),fontsize=15)
    plt.show()



if __name__ == "__main__":
    showImg()
