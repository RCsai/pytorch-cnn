import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader

class dataPreparation():
    def __init__(self):
        super(dataPreparation,self).__init__()
        self.batch_size=100
        self.trainSet=datasets.MNIST(root='./pymnist',                   # 选择数据集目录
                                      train=True,                       # 选择训练集
                                      transform=transforms.ToTensor(),  # 转化为Tensor变量
                                      download=True)                    # 从网络上进行下载

        self.testSet=datasets.MNIST(root='./pymnist',
                                     train=False,                       # 选择测试集
                                     transform=transforms.ToTensor(),
                                     download=True)

        self.train_loader=DataLoader(dataset=self.trainSet,             # 设置转载数据
                                     batch_size=self.batch_size,        # 使用批次数据
                                     shuffle=True)                      # 将数据打乱

        self.test_loader=DataLoader(dataset=self.testSet,
                                    batch_size=10000,
                                    shuffle=True)

        self.test_loader_final=DataLoader(dataset=self.testSet,shuffle=True)


if __name__=="__main__":
    data=dataPreparation()

    # print("trainData:",data.trainSet.train_data.size())
    # print("trainDataLabel:",data.trainSet.train_labels.size())
    # print("testData",data.testSet.test_data.size())
    # print("testDataLabel",data.testSet.test_labels.size())

    # print("批次尺寸：",data.batch_size)
    # print("train_data_loader:",data.train_loader.dataset.train_data.shape)
    # print("train_labels_loader:",data.train_loader.dataset.train_labels.shape)


    # 现在将train_data,test_data名称改为data，train_labels,test_labels名称改为targets
    print("trainData:",data.trainSet.data.size())
    print("trainDataLabel:",data.trainSet.targets.size())
    print("testData",data.testSet.data.size())
    print("testDataLabel",data.testSet.targets.size())

    print("批次尺寸：",data.batch_size)
    print("train_data_loader:",data.train_loader.dataset.data.shape)
    print("train_labels_loader:",data.train_loader.dataset.targets.shape)




