import torch
from mypackage.model.unet_standard import NestedUNet
from mypackage.model.res_nested import ResNestedUNet, ResNestedUNetVer2, ResNestedUNetDeep, NestEncoderDecoder
from mypackage.model.res_nested import ResNestedUNetDeep_contrast1, ResNestedUNetDeep_contrast2,ResNestedUNetDeep_ver3
from mypackage.model.res_nested import Inpainting_simple_net
from mypackage.model.res_nested import Clean_Lucky_Net, No_inpainting
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from dataset import gamma_correction
import matplotlib.pyplot as plt
import torch.nn as nn
from math import log10, sqrt

import tqdm

def model_set(train_type, dev):
    if train_type == 'denoise':
        net_G = ResNestedUNetDeep_contrast2()
        net_G = net_G.to(dev)
        net_G_path = r'./save_models/denoise.pth'
        net_G.load_state_dict(torch.load(net_G_path, map_location=dev))
        return {'net_G': net_G, 'net_M': None, 'net_D': None}



def forward(net_G,train_type,input,net_M=None, net_D=None):
    if train_type=='denoise':
        net_G.eval()
        with torch.no_grad():
            output = net_G(input.float())
            return output



def preprocess_test_img(img_path, dev,transform=transforms.Compose([transforms.ToTensor()])):
    with Image.open(img_path) as lr:
        lr = lr.convert("L")
        lr_size = lr.size
        lr = lr.resize((512,512),Image.BICUBIC) #一般是512*512
        lr = np.array(lr)
        lr = lr / 255.
        min_lr = lr.min()
        max_lr = lr.max()
        lr = (lr - min_lr)/(max_lr-min_lr)
        # lr = lr[0:512, 0:512]
        lr = transform(lr).to(dev)
    return lr, lr_size, min_lr, max_lr


def save_image(tensor, img_path, input_size, min_lr, max_lr, gt_path=None,train_type=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    if train_type != 'mask':
        image = image*(max_lr-min_lr)+min_lr
    # plot_img(image)
    image = transforms.ToPILImage()(image.float())
    image = image.resize(input_size, Image.BICUBIC)
    image.save(os.path.splitext(img_path)[0]+'.png')

    if gt_path is not None:
        with Image.open(gt_path) as gt:
            gt = gt.convert("L")
            gt = np.array(gt)
            image = image.convert("L")
            image = np.array(image)
            mse = ((gt - image)**2).mean()
            psnr = 20*log10(255/sqrt(mse))
            print(psnr)
            return psnr
    return 0



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = True
if use_gpu & torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False
dev = torch.device('cuda') if use_gpu else torch.device('cpu')



testset_name = 'high_angle'
testset_resolution = '900'
train_type = 'denoise'


test_img_dir = './dataset/evaluate/'+testset_name+'/'+testset_resolution+'/img_proj_noise'
test_img_gt_dir = './evaluate/'+testset_name+'/'+testset_resolution+'/img_proj_par'
save_output_dir = './dataset/evaluate_output/'+testset_name+'_'+testset_resolution+'/'+train_type

os.makedirs(save_output_dir)

img_name = os.listdir(test_img_dir)
with torch.no_grad():
    for i in range(len(img_name)):
        img_path = os.path.join(test_img_dir, img_name[i])
        gt_path = os.path.join(test_img_gt_dir, img_name[i])
        save_path = os.path.join(save_output_dir, img_name[i])

        input, input_size, min_lr, max_lr= preprocess_test_img(img_path, dev=dev)
        input = input.unsqueeze(0)
        models = model_set(train_type=train_type, dev=dev)
        output = forward(net_G=models['net_G'], net_M=models['net_M'],net_D=models['net_D'],\
                         input=input.float(), train_type=train_type)
        psnr = save_image(output,save_path,input_size, min_lr, max_lr, gt_path=None, train_type=train_type)
        print('finish%d个图像'%(i+1))

