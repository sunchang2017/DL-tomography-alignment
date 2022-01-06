import os
import numpy as np
# import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from utils import guiyi, add_noise
import matplotlib
import math

class _NoisedDataset(Dataset):
    def __init__(self, transform, input_root, groundtruth_root, plus_noise = True,rotate = False, shear=False, random_crop=None, resize=None):
        self.transform = transforms.Compose(transform)
        names = os.listdir(input_root)
        names.sort()
        #每隔5度选一个
        names = np.array(names)
        a = np.arange(0, len(names), 5)
        names = names[a]
        self.paths_lr = [os.path.join(input_root, name) for name in names]
        self.paths_hr = [os.path.join(groundtruth_root, name) for name in names]
        self.total = len(self.paths_lr)
        self.random_crop = random_crop
        self.rotate = rotate
        self.shear = shear
        self.resize = resize
        self.plus_noise = plus_noise

    def __len__(self):
        return self.total

    def __getitem__(self, i):
        path_lr = self.paths_lr[i]
        path_hr = self.paths_hr[i]
        # print(self.paths_lr[i])
        # print(i)
        with Image.open(path_lr) as lr:
            lr = lr.convert("L")
            # lr = np.array(lr)
            # lr = lr / 255.
        with Image.open(path_hr) as hr:
            hr = hr.convert("L")
            # hr = hr.resize((512,512))
            # hr = np.array(hr)
            # hr = hr / 255.
        # print(np.array(lr).max()/255.)
        # print(np.array(lr).min() / 255.)
        # print(np.array(hr).max() / 255.)
        # print(np.array(hr).min() / 255.)
        if self.shear:
            shear = np.random.uniform(-60, 60)
            # print(shear)
            lr = F.affine(lr, angle=0, shear=shear, translate=(1,1), scale=1, resample=Image.BICUBIC)
            hr = F.affine(hr, angle=0, shear=shear, translate=(1,1), scale=1, resample=Image.BICUBIC)
        if self.rotate:
            angle = np.random.uniform(-180, 180)
            # print(angle)
            lr = F.rotate(lr, angle=angle, resample=Image.BICUBIC)
            hr = F.rotate(hr, angle=angle, resample=Image.BICUBIC)

        if self.random_crop is not None:
            lr, hr = self.random_crop(lr, hr)

        if self.resize is not None:
            lr = F.resize(lr, self.resize)
            hr = F.resize(hr, self.resize)


        if self.plus_noise==True:
            lr_noise = add_noise(lr)

        # lr = guiyi(lr)
        # hr = guiyi(hr)
        lr_noise = self.transform(lr_noise)
        hr = self.transform(hr)
        lr = self.transform(lr)
        denoise_res = lr_noise.double()-lr.double()
        subs_res = lr.double() - hr.double()
        noise_subs_res = lr_noise.double() - hr.double()
        # print("lr:", lr.max(), lr.min())
        # print("hr", hr.max(), hr.min())
        # print(path_lr)
        return {"lr_noise":lr_noise, "hr":hr, "lr":lr, "denoise_res":denoise_res, "subs_res":subs_res,\
                "noise_subs_res": noise_subs_res}


class NoisedDatasets():
    def __init__(self, input_root, groundtruth_root,batch_size=32,
                 num_workers=-1, valid_size=100,
                 shuffle=True,
                 rotate=False,
                 shear=False,
                 resize=None,
                 transform=[transforms.ToTensor()],
                 random_crop = None,
                 plus_noise = True):
        self.input_root = input_root
        self.groundtruth_root = groundtruth_root
        self.valid_size = valid_size
        self.batchsize = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transform = transform
        self.trainsetlength = 0
        self.random_crop = random_crop
        self.rotate = rotate
        self.shear = shear
        self.resize = resize
        self.plus_noise = plus_noise
        self.normnoise_dataset = _NoisedDataset(self.transform,
                                                input_root = self.input_root,
                                                groundtruth_root = self.groundtruth_root,
                                                plus_noise = self.plus_noise,
                                                rotate=self.rotate,
                                                shear= self.shear,
                                                resize= self.resize,
                                                random_crop = self.random_crop)

    def build_datasets(self):
        dataset_train, dataset_valid = random_split(self.normnoise_dataset,
                                                    [self.normnoise_dataset.total - self.valid_size, self.valid_size])
        self.trainsetlength = self.normnoise_dataset.total - self.valid_size
        print("train size:%d    valid size:%d" % (self.normnoise_dataset.total - self.valid_size, self.valid_size))
        train_dataloader = DataLoader(dataset_train, batch_size=self.batchsize,
                                      shuffle=self.shuffle, num_workers=self.num_workers)
        if self.valid_size == 0:
            valid_dataloader = DataLoader(dataset_train, batch_size=self.batchsize,
                                      shuffle=self.shuffle, num_workers=self.num_workers)
        else:
            valid_dataloader = DataLoader(dataset_valid, batch_size=self.batchsize,
                                          shuffle=self.shuffle, num_workers=self.num_workers)
        return train_dataloader, valid_dataloader


class random_crop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def __call__(self, img, gt):
        h, w = img.size
        i = j = th = tw = 0
        yes_flag = 0
        while (yes_flag == 0):
            if self.crop_size[0]==self.crop_size[1]:
                th = tw = self.crop_size[0]
            else:
                th = tw = np.random.randint(self.crop_size[0], self.crop_size[1])
            i = np.random.randint(0, h - th)
            j = np.random.randint(0, w - tw)
            no_zero_gt = np.array(gt.crop((i, j, i + th, j + tw)))
            if no_zero_gt.sum() > 0:
                yes_flag = 1
        # print(th, i, j)
        return img.crop((i, j, i + th, j + tw)), gt.crop((i, j, i + th, j + tw))


def save_L_image(input, path):
    # 灰度图像保存
    outputImg = Image.fromarray(input * 255.0)
    # "L"代表将图片转化为灰度图
    outputImg = outputImg.convert('L')
    outputImg.save(path)

def gamma_correction(input, gam=0.3):
    input = np.power(input, gam)
    ma = input.max()
    mi = input.min()
    return (input-mi)/(ma-mi)

if __name__ == '__main__':
    torch.manual_seed(2) # cpu
    torch.cuda.manual_seed(1) #gpu
    np.random.seed(0) #numpy

    transform = [transforms.ToTensor()]
    # transform = [transforms.Resize(512),
    #     transforms.ToTensor()]
    resize = 256
    image_datasets = NoisedDatasets(input_root=r'../data/dataset/image_proj',
                                    groundtruth_root=r'../data/dataset/image_proj_par',
                                    batch_size=1,
                                    valid_size=0,
                                    shuffle=True,
                                    rotate = True,
                                    shear = True,
                                    resize=resize,
                                    transform=transform,
                                    random_crop = random_crop((500, 800)),
                                    num_workers=0)
    print(len(image_datasets.normnoise_dataset))
#
# # image_datasets = NoisedDatasets(input_root=r'C:\Users\sunchang\Desktop\obj_testimage\subs_matlab\dataset\image_proj',
# #                                 groundtruth_root=r'C:\Users\sunchang\Desktop\obj_testimage\subs_matlab\dataset\image_proj_par',
# #                                 batch_size=1,
# #                                 valid_size=0,
# #                                 shuffle=True,
# #                                 rotate = False,
# #                                 shear = False,
# #                                 transform=transform,
# #                                 random_crop = None,
# #                                 num_workers=0)
#
# train_dataloader, valid_dataloader = image_datasets.build_datasets()
#
# i=0
# for data in train_dataloader:
#     i=i+1
#     input, groundtruth = data
#     input = input[0,0,:,:].numpy()
#     groundtruth = groundtruth[0,0,:,:].numpy()
#     print('input:',input.max(), input.min())
#     print('gt:', groundtruth.max(), groundtruth.min())
#
#     #save_L_image(input, './1.png')
#     #save_L_image(groundtruth, './2.png')
#
#     if i==4:
#         plt.subplot(1,2,1)
#         plt.imshow(gamma_correction(input), cmap="gray")
#         plt.subplot(1,2,2)
#         plt.imshow(gamma_correction(groundtruth), cmap='gray')
#         plt.show()
#         print('finish')
#         break

# a = np.arange(0, 49, 1).astype(np.uint8)
# a = a.reshape(7,7)
# b = Image.fromarray(a)
#
# ii = 5
# jj = 1
# th = 2
# tw = 2
# c = b.crop((ii, jj, ii+th, jj+tw))
# d = np.asarray(c)
# print(d)


# a = np.arange(0, 63450, 1)
# b = np.arange(0, 63450, 5)
# c = a[b]
# print(a.shape)
# print(b.shape)
# print(b.shape)