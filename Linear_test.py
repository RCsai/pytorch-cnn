from dataset import dataPreparation
from net import Neural_net
from net import net
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.cm as CM


def test():
    weight_root = './checkpoints/net/best.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = net().to(device)
    model.load_state_dict(torch.load(weight_root))
    data = dataPreparation()
    test_loader = data.test_loader_final
    # test_loader=data.test_loader
    num = 0
    error=0
    right=0
    # print(test_loader.dataset.data.shape)
    for i, (test_images, test_labels) in enumerate(test_loader):
        # test_images = test_images[i].to(device)
        # test_labels = test_labels[i].to(device)
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_images = Variable(test_images.view(-1, 28 * 28))
        test_outputs = model(test_images)
        _, predicts = torch.max(test_outputs.data, 1)

        # print("truth:", test_labels, '\t', "predict:", predicts, '\n')
        if test_labels==predicts:
            num+=1

        # 显示错误50张图片
        if error<50 and test_labels!=predicts:
            plt.figure("error")
            plt.subplot(5, 10, error+1)
            img = test_images.numpy().reshape(28, 28)
            plt.imshow(img, cmap=CM.gray)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("truth:" + str(test_labels.numpy()) + "  predict:" + str(predicts.numpy()))
            error+=1

        # 显示正确50张图片
        if right<50 and test_labels==predicts:
            plt.figure("right")
            plt.subplot(5, 10, right + 1)
            img = test_images.numpy().reshape(28, 28)
            plt.imshow(img, cmap=CM.gray)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("truth:" + str(test_labels.numpy()) + "  predict:" + str(predicts.numpy()))
            right += 1

    accuracy=num/10000
    print("accuracy:",accuracy)
    plt.show()


if __name__ == "__main__":
    test()
