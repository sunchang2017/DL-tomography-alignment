import torch
from mypackage.model.unet_standard import NestedUNet_3down
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from dataset import gamma_correction
import matplotlib.pyplot as plt
import torch.nn as nn
from math import log10, sqrt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--testset_name', type= str, default='high_angle')
parser.add_argument('--testset_resolution', type= str, default='600')
parser.add_argument('--train_type', type= str, default='mask')

args = parser.parse_args()

testset_name = args.testset_name
testset_resolution = args.testset_resolution
train_type = args.train_type
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def model_set(train_type,dev):
    if train_type == 'mask':
        net_G = NestedUNet_3down(scale_output=False)
        net_G = net_G.to(dev)
        net_G_path = r'./save_models/SN.pth'
        net_G.load_state_dict(torch.load(net_G_path, map_location=dev))
        return {'net_G': net_G, 'net_M': None, 'net_D': None}


def forward(net_G,train_type,input,net_M=None, net_D=None):
    net_G.eval()
    with torch.no_grad():
        output = net_G(input.float())
        output = nn.Sigmoid()(output)
        output[output>0.9]=1  #一般为0.5
        output[output<=0.9]=0
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


use_gpu = True
if use_gpu & torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False
dev = torch.device('cuda') if use_gpu else torch.device('cpu')


test_img_dir = './dataset/evaluate/'+testset_name+'/'+testset_resolution+'/img_proj_noise'
test_img_gt_dir = './dataset/evaluate/'+testset_name+'/'+testset_resolution+'/img_proj_par'
save_output_dir = './dataset/evaluate_output/'+testset_name+'_'+testset_resolution+'/'+train_type

if not os.path.exists(save_output_dir):
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
        # break
        print('finish%d个图像'%(i+1))