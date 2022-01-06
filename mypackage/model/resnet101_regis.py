import torch.nn as nn
import torchvision
import torch
class resnet101_regis(object):
    def __init__(self):
        self.netG = torchvision.models.resnet101(pretrained=False, num_classes=2)
        self.netG.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
        self.netG.fc = nn.Sequential(self.netG.fc, nn.Tanh())
# netG = torchvision.models.resnet101(pretrained=False, num_classes=2)
# netG.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
# netG.fc = nn.Sequential(netG.fc, nn.Tanh())
class resnet18_regis(object):
    def __init__(self):
        self.netG = torchvision.models.resnet18(pretrained=False, num_classes=2)
        self.netG.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
        self.netG.fc = nn.Sequential(self.netG.fc, nn.Tanh())

class small_regis(nn.Module):
    def __init__(self):
        super(small_regis, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = nn.Linear(512*8*8, 2)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.tanh(x)
        return x

class small_regis_avg(nn.Module):
    def __init__(self):
        super(small_regis_avg, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = nn.Linear(512*8*8, 2)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.tanh(x)
        return x

class middle_regis_avg(nn.Module):
    def __init__(self):
        super(middle_regis_avg, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = nn.Linear(512*8*8, 2)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.tanh(x)
        return x
# from torch import nn
#
#
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 25, kernel_size=3),
#             nn.BatchNorm2d(25),
#             nn.ReLU(inplace=True)
#         )
#
#         self.layer2 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(25, 50, kernel_size=3),
#             nn.BatchNorm2d(50),
#             nn.ReLU(inplace=True)
#         )
#
#         self.layer4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.fc = nn.Sequential(
#             nn.Linear(50 * 5 * 5, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 10)
#         )
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
# if __name__ == '__main__':
    # netG = resnet101_regis().netG
    # a = torch.ones(2, 1, 128, 128)
    # b = netG(a)
    # print(b.shape)
    # size = count_parameters(netG)
    # print(size)

    # net = small_regis()
    # a = torch.ones(4, 1, 256, 256)
    # b = net(a)
    # print(b.shape)