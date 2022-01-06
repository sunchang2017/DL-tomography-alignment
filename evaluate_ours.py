import torch
from mypackage.model.unet_standard import NestedUNet_3down
from mypackage.model.res_nested import Clean_Lucky_Net_lunwen,ResNestedUNetDeep_contrast2
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from dataset import gamma_correction
import matplotlib.pyplot as plt
from mypackage.tricks import get_PSNR
from utils import cal_barycenter, cal_mse, draw_mask_on, compute_ssim
import torch.nn as nn
from math import log10, sqrt
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--testset_name', type= str, default='low_angle')
parser.add_argument('--testset_resolution', type= str, default='300')
parser.add_argument('--train_type', type= str, default='ours')
parser.add_argument('--use_cuda', type=str, default='2')

parser.parse_args()
args = parser.parse_args()
testset_name = args.testset_name
testset_resolution = args.testset_resolution
train_type = args.train_type
os.environ["CUDA_VISIBLE_DEVICES"] = args.use_cuda
use_gpu = True
dev = torch.device('cuda') if use_gpu else torch.device('cpu')

def model_set(train_type, dev):
    if train_type == 'ours':
        net_G = Clean_Lucky_Net_lunwen(num_vggblocks=4)
        net_G = net_G.to(dev)
        net_G_path = r'./save_models/overall.pth'
        net_G.load_state_dict(torch.load(net_G_path, map_location=dev)['model_2'])

        net_D = ResNestedUNetDeep_contrast2()
        net_D = net_D.to(dev)
        net_D.load_state_dict(torch.load(net_G_path, map_location=dev)['model_1'])
        return {'net_G': net_G, 'net_D': net_D}



def forward(net_G,net_D,train_type,input,mask):
    if train_type=='ours':
        net_G.eval()
        net_D.eval()
        with torch.no_grad():
            output_M = mask - 1
            output_M[output_M < 0] = 1
            output = net_D(input)
            output = net_G(output, output_M)
            return output



def preprocess_test_img(img_path, dev,type='mask',transform=transforms.Compose([transforms.ToTensor()])):
    with Image.open(img_path) as lr:
        lr = lr.convert("L")
        lr_size = lr.size
        lr = lr.resize((512,512),Image.BICUBIC) #一般是512*512
        lr = np.array(lr)
        lr = lr / 255.
        if type=='mask':
            lr[lr <= 0.5] = 0
            lr[lr > 0.5] = 1
            # plt.figure()
            # plt.imshow(lr,cmap='gray')
            # plt.show()
            lr = transform(lr).to(dev)
            return lr
        else:
            min_lr = lr.min()
            max_lr = lr.max()
            lr = (lr - min_lr)/(max_lr-min_lr)
            # lr = lr[0:512, 0:512]
            # plt.figure()
            # plt.imshow(lr,cmap='gray')
            # plt.show()
            lr = transform(lr).to(dev)
            return lr, lr_size, min_lr, max_lr


def save_image(tensor, mask, img_path, input_size, min_lr, max_lr, gt_path=None,train_type=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it\
    image = image.squeeze(0)  # remove the fake batch dimension

    image = image*(max_lr-min_lr)+min_lr
    image = image.float()*mask.cpu().clone().squeeze(0).float()

    image = transforms.ToPILImage()(image.float())
    image = image.resize(input_size, Image.BICUBIC)
    image.save(os.path.splitext(img_path)[0]+'.png')

    if gt_path is not None:
        with Image.open(gt_path) as gt:
            gt = gt.convert("L")
            gt = np.array(gt)
            image = image.convert("L")
            image = np.array(image)

            bary_gt = cal_barycenter(gt)
            bary = cal_barycenter(image)
            bary_mse = cal_mse(bary_gt, bary)

            mse = ((gt - image)**2).mean()
            psnr = 20*log10(255/sqrt(mse))
            print(psnr, bary_mse)
            ssim = compute_ssim(gt, image, win_size=gt.shape[0])
            return [psnr,ssim,bary_mse]
    return 0


def read_img(img_path, img_name):
    img_path = os.path.join(img_path, img_name)
    with Image.open(img_path) as img:
        img = img.convert('L')
        img = np.array(img, dtype=float)
        # img = img/255
    return img



test_img_dir = './dataset/evaluate/' + testset_name + '/' + testset_resolution + '/img_proj_noise'
test_img_gt_dir = './dataset/evaluate/'+testset_name+'/'+testset_resolution+'/img_proj_par'
img_mask_path = './dataset/test/evaluate_output/' + testset_name + '_' + testset_resolution + '/' + 'mask'
save_output_dir = './dataset/evaluate_output/'+testset_name+'_'+testset_resolution+'/'+train_type
if not os.path.exists(save_output_dir):
    os.makedirs(save_output_dir)

save_barymse_dir = './dataset/evaluate_output/'+testset_name+'_'+testset_resolution+'/BaryMse_'+train_type+".txt"
bary_dic = {}
bary_dic[train_type] = []


img_name = os.listdir(test_img_dir)
img_name.sort()
with torch.no_grad():
    for i in range(len(img_name)):
        img_path = os.path.join(test_img_dir, img_name[i])
        gt_path = os.path.join(test_img_gt_dir, img_name[i])
        mask_path = os.path.join(img_mask_path, img_name[i])
        save_path = os.path.join(save_output_dir, img_name[i])

        input, input_size, min_lr, max_lr= preprocess_test_img(img_path, dev=dev, type='input')
        mask = preprocess_test_img(mask_path, dev=dev, type='mask')

        input = input.unsqueeze(0)
        mask = mask.unsqueeze(0)

        models = model_set(train_type=train_type, dev=dev)
        output = forward(net_G=models['net_G'], net_D=models['net_D'],input=input.float(), mask=mask.float(),train_type=train_type)
        test_results = save_image(output,mask,save_path,input_size, min_lr, max_lr, gt_path=gt_path, train_type=train_type)
        psnr, ssim, bary_mse = test_results
        bary_dic[train_type].append(bary_mse)
        print('finish%d个图像' % (i + 1))
# print(np.mean(np.array(bary_dic[train_type])))

f = save_barymse_dir
os.mknod(f)
with open(f,"w") as file:   #”w"代表着每次运行都覆盖内容
    for j in range(300):
        file.write(str(bary_dic[train_type][j]) + " ")
        file.write("\n")
