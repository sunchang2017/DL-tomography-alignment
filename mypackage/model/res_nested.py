# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F
import torch
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image
from dataset import gamma_correction

def plot_img(img_np, title='output', cmap='gray'):
    im_max = img_np.max()
    im_min = img_np.min()
    print(im_max, im_min)
    img = (img_np - im_min) / (im_max - im_min)
    print(img.max(), img.min())
    # img = img/1.5
    # img = gamma_correction(img)
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU()):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


class ResNestedUNet(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512), scale_output=True):
        super().__init__()
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_pre = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        # self.conv_mask = VGGBlock(in_channels, nb_filter[0], nb_filter[0])

        self.conv0_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.conv_post = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
    def forward(self, input, mask):
        pre = self.conv_pre(input)
        mask_pre = pre * mask

        x0_0 = self.conv0_0(mask_pre)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.scale_output == True:
            post = self.conv_post(pre-x0_4)
            output = self.final(post)

            # k = pre.clone().cpu().detach()
            # print(k.shape)
            # k = k.numpy()[1]
            # k = k.mean(0)
            # print(k.shape)
            # plt.imshow(gamma_correction(k), cmap='gray')
            # plt.show()
            # plt.imshow(gamma_correction(k))
            # plt.show()
            # print('finish')

            return nn.Tanh()(output)

class ResNestedUNetVer2(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512), scale_output=True):
        super().__init__()
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_pre = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        # self.conv_mask = VGGBlock(in_channels, nb_filter[0], nb_filter[0])

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.conv_post = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
    def forward(self, input, mask):
        pre = self.conv_pre(input)
        mask_pre = input * mask

        # k = mask_pre.clone().cpu().detach()
        # print(k.shape)
        # k = k.numpy()[1]
        # k = k.mean(0)
        # print(k.shape)
        # plt.imshow(gamma_correction(k), cmap='gray')
        # plt.show()
        # plt.imshow(gamma_correction(k))
        # plt.show()
        # print('finish')

        x0_0 = self.conv0_0(mask_pre)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.scale_output == True:
            post = self.conv_post(pre-x0_4)
            output = self.final(post)
            return nn.Tanh()(output)


class ResNestedUNetDeep(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512), scale_output=True):
        super().__init__()
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_pre = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        # self.conv_mask = VGGBlock(in_channels, nb_filter[0], nb_filter[0])

        self.conv0_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.conv_post_1 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
    def forward(self, input, mask):
        pre = self.conv_pre(input)
        mask_pre = pre * mask

        x0_0 = self.conv0_0(mask_pre)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.scale_output == True:
            post = self.conv_post_1(pre-x0_4)
            post = self.conv_post_2(post)
            post = self.conv_post_3(post)
            post = self.conv_post_4(post)
            output = self.final(post)

            k = x0_4.clone().cpu().detach()
            k = k.numpy()[0]
            k = k.mean(0)
            print(k.max(), k.min())
            plot_img(k, cmap='gray')

            k = pre.clone().cpu().detach()
            k = k.numpy()[0]
            k = k.mean(0)
            print(k.max(), k.min())
            plot_img(k, cmap='gray')
            print('finish')

            k = abs(pre-x0_4).clone().cpu().detach()
            k = k.numpy()[0]
            k = k.mean(0)
            print(k.max(), k.min())
            plot_img(k, cmap='gray')

            return nn.Tanh()(output)

class ResNestedUNetDeep_contrast1(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512), scale_output=True):
        super().__init__()
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_pre = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        # self.conv_mask = VGGBlock(in_channels, nb_filter[0], nb_filter[0])

        self.conv0_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.conv_post_1 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
    def forward(self, input):
        pre = self.conv_pre(input)

        x0_0 = self.conv0_0(pre)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.scale_output == True:
            post = self.conv_post_1(pre-x0_4)
            post = self.conv_post_2(post)
            post = self.conv_post_3(post)
            post = self.conv_post_4(post)
            output = self.final(post)
            return nn.Tanh()(output)

class ResNestedUNetDeep_contrast2_3down(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256), scale_output=True):
        super().__init__()
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_pre = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        # self.conv_mask = VGGBlock(in_channels, nb_filter[0], nb_filter[0])

        self.conv0_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])


        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.conv_post_1 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
    def forward(self, input):
        pre = self.conv_pre(input)

        x0_0 = self.conv0_0(pre)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        if self.scale_output == True:
            post = self.conv_post_1(x0_3)
            post = self.conv_post_2(post)
            post = self.conv_post_3(post)
            post = self.conv_post_4(post)
            output = self.final(post)

            # k = pre.clone().cpu().detach()
            # print(k.shape)
            # k = k.numpy()[1]
            # k = k.mean(0)
            # print(k.shape)
            # plt.imshow(gamma_correction(k), cmap='gray')
            # plt.show()
            # plt.imshow(gamma_correction(k))
            # plt.show()
            # print('finish')

            return nn.Tanh()(output)



class ResNestedUNetDeep_contrast2(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512), scale_output=True):
        super().__init__()
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_pre = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        # self.conv_mask = VGGBlock(in_channels, nb_filter[0], nb_filter[0])

        self.conv0_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.conv_post_1 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
    def forward(self, input):
        pre = self.conv_pre(input)

        x0_0 = self.conv0_0(pre)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.scale_output == True:
            post = self.conv_post_1(x0_4)
            post = self.conv_post_2(post)
            post = self.conv_post_3(post)
            post = self.conv_post_4(post)
            output = self.final(post)

            # k = pre.clone().cpu().detach()
            # print(k.shape)
            # k = k.numpy()[1]
            # k = k.mean(0)
            # print(k.shape)
            # plt.imshow(gamma_correction(k), cmap='gray')
            # plt.show()
            # plt.imshow(gamma_correction(k))
            # plt.show()
            # print('finish')

            return nn.Tanh()(output)



class NestEncoderDecoder(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512), scale_output=True):
        super().__init__()
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self.up(x4_0))
        x2_2 = self.conv2_2(self.up(x3_1))
        x1_3 = self.conv1_3(self.up(x2_2))
        x0_4 = self.conv0_4(self.up(x1_3))

        if self.scale_output == True:
            output = self.final(x0_4)
            return nn.Tanh()(output)
        else:
            output = self.final(x0_4)
            return output

class ResNestedUNetDeep_ver3(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512), scale_output=True):
        super().__init__()
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.conv_pre = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        # self.conv_mask = VGGBlock(in_channels, nb_filter[0], nb_filter[0])

        self.conv0_0 = VGGBlock(1, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])


        self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.conv_post_1 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv_post_2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv_post_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
    def forward(self, input, mask):
        mask_pre = input * mask

        x0_0 = self.conv0_0(mask_pre)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x0_4_final = self.final1(x0_4)

        if self.scale_output == True:
            post = self.conv_post_1(input-x0_4_final)
            post = self.conv_post_2(post)
            post = self.conv_post_3(post)
            post = self.conv_post_4(post)
            output = self.final2(post)

            k = (input-x0_4_final).clone().cpu().detach()
            k = k.numpy()[0]
            k = k.mean(0)
            im_max = k.max()
            im_min = k.min()
            img = (k - im_min) / (im_max - im_min)
            plt.imshow(img, cmap='PuRd')
            plt.axis('off')
            plt.savefig('/home/sc/xin/data/dataset/evaluate/high_angle/some_images_for_paper/' +
                           'substrate2.png')
            plt.show()

            # k = x0_4_final.clone().cpu().detach()
            # k = k.numpy()[0]
            # k = k.mean(0)
            # print(k.max(), k.min())
            # plot_img(k, cmap='viridis')
            # #
            # k = input.clone().cpu().detach()
            # k = k.numpy()[0]
            # k = k.mean(0)
            # print(k.max(), k.min())
            # plot_img(k, cmap='gray')
            # print('finish')
            #
            # k = abs(input-x0_4).clone().cpu().detach()
            # k = k.numpy()[0]
            # k = k.mean(0)
            # print(k.max(), k.min())
            # plot_img(k, cmap='gray')

            return {"output": nn.Tanh()(output), "middle":x0_4_final}


class Clean_Lucky_Net(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512), num_vggblocks=4, scale_output=True):
        super().__init__()
        self.num_vggblock = num_vggblocks
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.post_1 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.post_2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_5 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        if self.num_vggblock ==5:
            self.post_6 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, input, mask):
        x0_0 = self.conv0_0(input*mask)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = nn.Tanh()(output)

        # k = output.clone().cpu().detach()
        # k = k.numpy()[0]
        # k = k.mean(0)
        # plot_img(k, cmap='gray')
        #保存为灰色图
        # im_max = k.max()
        # im_min = k.min()
        # img = (k - im_min) / (im_max - im_min)
        # outputImg = Image.fromarray(img * 255.0)
        # outputImg = outputImg.convert('L')
        # outputImg.save('/home/sc/xin/data/dataset/evaluate/high_angle/some_images_for_paper/' +
        #                'substrate.png')
        #保存为彩色图
        # im_max = k.max()
        # im_min = k.min()
        # img = (k - im_min) / (im_max - im_min)
        # plt.imshow(img, cmap='PuRd')
        # plt.axis('off')
        # plt.savefig('/home/sc/xin/data/dataset/evaluate/high_angle/some_images_for_paper/' +
        #                'substrate.png')
        # plt.show()

        # #
        # k = input.clone().cpu().detach()
        # k = k.numpy()[0]
        # k = k.mean(0)
        # # print(k.max(), k.min())
        # plot_img(k, cmap='gray')
        # print('finish')
        #
        # k = abs(input - output).clone().cpu().detach()
        # k = k.numpy()[0]
        # k = k.mean(0)
        # print(k.max(), k.min())
        # plot_img(k, cmap='gray')

        output = self.post_1(input-output)
        if self.num_vggblock == 3:
            output = self.post_2(output)
            output = self.post_3(output)
            output = self.post_4(output)
        elif self.num_vggblock == 4: #(ours)
            output = self.post_2(output)
            output = self.post_3(output)
            output = self.post_4(output)
            output = self.post_5(output)
        elif self.num_vggblock == 5:
            output = self.post_2(output)
            output = self.post_3(output)
            output = self.post_4(output)
            output = self.post_5(output)
            output = self.post_6(output)
        output = self.post_final(output)
        return nn.Tanh()(output)

class Clean_Lucky_Net_lunwen(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512),
                 num_vggblocks=4,
                 scale_output=True,
                 NoInpainting=False,
                 NoSubstract=False):
        super().__init__()
        self.nosubstract = NoSubstract
        self.noinpainting = NoInpainting
        self.num_vggblock = num_vggblocks
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.post_1 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.post_2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_5 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        if self.num_vggblock ==5:
            self.post_6 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, input, mask):
        if self.noinpainting:
            x0_0 = self.conv0_0(input)
        else:
            x0_0 = self.conv0_0(input*mask)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        output = self.final(x0_3)
        output = nn.Tanh()(output)

        # k = output.clone().cpu().detach()
        # k = k.numpy()[0]
        # k = k.mean(0)
        # plot_img(k, cmap='gray')
        #保存为灰色图
        # im_max = k.max()
        # im_min = k.min()
        # img = (k - im_min) / (im_max - im_min)
        # outputImg = Image.fromarray(img * 255.0)
        # outputImg = outputImg.convert('L')
        # outputImg.save('/home/sc/xin/data/dataset/evaluate/high_angle/some_images_for_paper/' +
        #                'substrate.png')
        #保存为彩色图
        # im_max = k.max()
        # im_min = k.min()
        # img = (k - im_min) / (im_max - im_min)
        # plt.imshow(img, cmap='PuRd')
        # plt.axis('off')
        # plt.savefig('/home/sc/xin/data/dataset/evaluate/high_angle/some_images_for_paper/' +
        #                'substrate.png')
        # plt.show()

        # #
        # k = input.clone().cpu().detach()
        # k = k.numpy()[0]
        # k = k.mean(0)
        # # print(k.max(), k.min())
        # plot_img(k, cmap='gray')
        # print('finish')
        #
        # k = abs(input - output).clone().cpu().detach()
        # k = k.numpy()[0]
        # k = k.mean(0)
        # print(k.max(), k.min())
        # plot_img(k, cmap='gray')

        if self.nosubstract:
            output = self.post_1(output)
        else:
            output = self.post_1(input-output)
        if self.num_vggblock == 3:
            output = self.post_2(output)
            output = self.post_3(output)
            output = self.post_4(output)
        elif self.num_vggblock == 4: #(ours)
            output = self.post_2(output)
            output = self.post_3(output)
            output = self.post_4(output)
            output = self.post_5(output)
        elif self.num_vggblock == 5:
            output = self.post_2(output)
            output = self.post_3(output)
            output = self.post_4(output)
            output = self.post_5(output)
            output = self.post_6(output)
        output = self.post_final(output)
        return nn.Tanh()(output)


class Inpainting_simple_net(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512), scale_output=True):
        super().__init__()
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.post_1 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.post_2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_5 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_5 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_5 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = nn.Tanh()(output)

        # k = output.clone().cpu().detach()
        # k = k.numpy()[0]
        # k = k.mean(0)
        # print(k.max(), k.min())
        # plot_img(k, cmap='viridis')
        #
        # k = input.clone().cpu().detach()
        # k = k.numpy()[0]
        # k = k.mean(0)
        # print(k.max(), k.min())
        # plot_img(k, cmap='gray')
        # print('finish')
        #
        # k = abs(input - output).clone().cpu().detach()
        # k = k.numpy()[0]
        # k = k.mean(0)
        # print(k.max(), k.min())
        # plot_img(k, cmap='gray')

        output = self.post_1(input-output)
        output = self.post_2(output)
        output = self.post_3(output)
        output = self.post_4(output)
        output = self.post_5(output)
        output = self.post_final(output)
        return nn.Tanh()(output)

class No_inpainting(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512), scale_output=True):
        super().__init__()
        self.scale_output = scale_output
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.post_1 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.post_2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_5 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_5 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_5 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.post_final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = nn.Tanh()(output)
        output = self.post_1(output)
        output = self.post_2(output)
        output = self.post_3(output)
        output = self.post_4(output)
        output = self.post_5(output)
        output = self.post_final(output)
        return nn.Tanh()(output)


class SubsCroase2Fine(nn.Module):
    def __init__(self, in_channels=1, nb_filter=(64,64)):
        super().__init__()
        self.conv0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv5 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv6 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv7 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv8 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv9 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
    def forward(self, input):
        output = self.conv0(input)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        output = self.conv8(output)
        output = self.conv9(output)
        output = self.final(output)
        return nn.Tanh()(output)

if __name__ == '__main__':
    # from jdit import Model
    #
    # unet = Model(NestedUNet(1))
    # print(unet.num_params)



    # netG = ResNestedUNet()
#     # input = torch.zeros((16,1,256,256))
#     # input_mask = torch.zeros((16,1,256,256))
#     # output = netG(input, input_mask)
#     # print(output.shape)

    netG = ResNestedUNetDeep_ver3()
    input = torch.zeros((16,1,256,256))
    mask = torch.zeros((16,1,256,256))
    output = netG(input, mask)
    print(output["output"].shape)
    print(output["middle"].shape)

    # import PIL.Image as Image
    # from utils import  add_noise
    # import numpy as np
    #
    # img1_path = '/home/sc/xin/data/dataset/evaluate/high_angle/img_proj/00466006.png'
    # img2_path = '/home/sc/xin/data/dataset/evaluate/high_angle/img_proj_par/00466006.png'
    #
    # with Image.open(img1_path) as img1:
    #     img1 = img1.convert('L')
    #     img1_org = np.array(img1, dtype=float)
    #     img1_org = img1_org/255
    #     img1 = add_noise(img1)
    #     # img1 = np.array(img1)
    #     img1_noise_error = abs(img1_org-img1)
    # with Image.open(img2_path) as img2:
    #     img2 = img2.convert('L')
    #     # img2 = guiyi(img2)
    #     img2 = np.array(img2, dtype=float)
    #     img2 = img2/255
    #
    #
    # error = abs(img1-img2)
    # # print(error.max(), error.min())
    # # error_max = error.max()
    # # error_min = error.min()
    # # error = (error-error_min)/(error_max-error_min)
    # plt.imshow(img1, cmap='gray')
    # plt.show()
    # plt.imshow(img2, cmap='gray')
    # plt.show()
    # # plt.imshow(gamma_correction(error), cmap='gray')
    # # plt.show()
    # plt.imshow(img1_org, cmap='gray')
    # plt.show()
    # plt.imshow(img1_noise_error, cmap='gray')
    # plt.show()
    # plt.imshow(error, cmap='gray')
    # plt.show()
    # plt.imshow(abs(img1_org-img2), cmap='gray')
    # plt.show()
    # print(img1_org[500,100])
    # print(img1[500, 100])
    # print(img2[500,100])
    # print(error[500, 100])



