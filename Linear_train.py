from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from net import net
from net import Neural_net
from tqdm import tqdm
from dataset import dataPreparation
import os
from tensorboardX import SummaryWriter


def train_Neural_net():
    # 参数定义
    lr = 1e-1
    epoches = 5
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Neural_net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # 使用随机梯度下降

    data = dataPreparation()
    train_loader = data.train_loader
    test_loader = data.test_loader

    checkpoints = './checkpoints/Neural_net/'
    writer = SummaryWriter()

    accuracy_best = 0
    for epoch in range(epoches):
        print("第%d轮训练" % (epoch + 1))
        for i, (images, labels) in enumerate(tqdm(train_loader, ncols=50)):
            images = images.to(device)
            labels = labels.to(device)
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            outputs = model(images)  # 计算模型输出结果
            loss = criterion(outputs, labels)  # 计算误差loss
            optimizer.zero_grad()  # 在反向传播之前清除网络状态
            loss.backward()  # loss反向传播
            optimizer.step()  # 更新参数

        print("epoch_" + str(epoch) + "_loss:", loss.item())
        writer.add_scalar('Train_loss', loss, epoch)

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for test_images, test_labels in test_loader:
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                test_images = Variable(test_images.view(-1, 28 * 28))
                test_outputs = model(test_images)
                _, predicts = torch.max(test_outputs.data, 1)
                total += test_labels.size(0)
                correct+=(predicts==test_labels).sum().numpy()
                accuracy = correct / total
                if accuracy > accuracy_best:
                    accuracy_best = accuracy
                    torch.save(model.state_dict(), os.path.join(checkpoints, 'best.pth'))
            print("accuracy_best:", accuracy_best)


def train_net():
    # 参数定义
    lr = 1e-1                   # 学习率
    epoches = 55                # 迭代次数
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = net().to(device)
    criterion = nn.CrossEntropyLoss()   # 目标函数
    optimizer = optim.SGD(model.parameters(), lr=lr)  # 定义优化器种类，使用随机梯度下降
    data = dataPreparation()
    train_loader = data.train_loader    # 设置训练数据
    test_loader = data.test_loader      # 设置测试数据

    checkpoints = './checkpoints/net/'
    writer = SummaryWriter()

    accuracy_best = 0

    for epoch in range(epoches):
        print("第%d轮训练" % (epoch + 1))
        for i, (images, labels) in enumerate(tqdm(train_loader, ncols=50)):
            images = images.to(device)
            labels = labels.to(device)
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            outputs = model(images)  # 计算模型输出结果
            loss = criterion(outputs, labels)  # 计算误差loss
            optimizer.zero_grad()  # 在反向传播之前清除网络状态
            loss.backward()  # loss反向传播
            optimizer.step()  # 更新参数

        print("epoch_" + str(epoch) + "_loss:", loss.item())
        writer.add_scalar('Train_loss', loss, epoch)            #第一个参数为图形名称，第二个参数为y值，第三个为x值

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for test_images, test_labels in test_loader:
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                test_images = Variable(test_images.view(-1, 28 * 28))
                test_outputs = model(test_images)
                _, predicts = torch.max(test_outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicts == test_labels).sum().numpy()
                accuracy = correct / total
                if accuracy > accuracy_best:
                    accuracy_best = accuracy
                    torch.save(model.state_dict(), os.path.join(checkpoints, 'best.pth'))
            writer.add_scalar("accuracy",accuracy,epoch)
            print("accuracy_best:", accuracy_best, '\n')


if __name__ == "__main__":
    train_net()
