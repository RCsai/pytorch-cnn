from torch import nn
import torch

# 全连接网络
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.input = 28 * 28                # 输入层大小
        self.hidden1 = 300                  # 隐藏层1的大小
        self.hidden2 = 100                  # 隐藏层2的大小
        self.hidden3 = 50                   # 隐藏层3的大小
        self.output = 10                    # 输出层的大小
        self.layer1 = nn.Sequential(nn.Linear(self.input, self.hidden1), nn.ReLU(True))    # 设置每个全连接层的激活函数为ReLU
        self.layer2 = nn.Sequential(nn.Linear(self.hidden1, self.hidden2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(self.hidden2, self.hidden3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(self.hidden3, self.output))

    def forward(self, x):                   # 设置模型的数据传递顺序
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Neural_net(nn.Module):
    def __init__(self):
        super(Neural_net, self).__init__()
        self.input = 28 * 28
        self.hidden1 = 500
        self.output = 10
        self.layer1 = nn.Sequential(nn.Linear(self.input, self.hidden1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(self.hidden1, self.output))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


# 卷积神经网络
class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,25,kernel_size=3),      # 3*3的卷积层
            nn.BatchNorm2d(25),                 # 批规范化层
            nn.ReLU(inplace=True)               # 激活函数
        )

        self.layer2=nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)    # 2*2的池化层
        )

        self.layer3=nn.Sequential(
            nn.Conv2d(25,50,kernel_size=3),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True)
        )

        self.layer4=nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.fc=nn.Sequential(
            nn.Linear(50*5*5,1024),         # 连接层
            nn.ReLU(inplace=True),          # 激活函数
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x


# if __name__ == "__main__":
#     net = Neural_net()
#     print(net)

if __name__ == "__main__":
    model = CNN_net()
    print(model)
    input=torch.ones((10,1,28,28))
    print(input.shape)
    output=model(input)
    print("output_shape:",output.shape)
