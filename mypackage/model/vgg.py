import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataset import gamma_correction
from mypackage.model.unet_standard import NestedUNet
from utils import guiyi
import numpy as np
from torchvision import transforms
import os
from math import log10, sqrt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# model = torchvision.models.vgg16(pretrained=False)
# model.load_state_dict(torch.load('/home/sc/xin/de_background/models/vgg16-397923af.pth'))
# model = model.features

use_gpu = True
if use_gpu & torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False
dev = torch.device('cuda') if use_gpu else torch.device('cpu')

model = NestedUNet()




img_ip_path = '/home/sc/xin/de_background/test/tt/00139010_noise.png'
img_op_path = '/home/sc/xin/de_background/test/00139010_noise_output.png'
img_gt_path = '/home/sc/xin/de_background/test_gt/00139010.png'
img_ms_path = '/home/sc/xin/de_background/test/00139010_noise_mask.png'

def preprocess_test_img(img_path, dev,transform=transforms.Compose([transforms.ToTensor()])):
    with Image.open(img_path) as lr:
        lr = lr.convert("L")
        lr_size = lr.size
        lr = lr.resize((512,512),Image.BICUBIC)
        lr = np.array(lr)
        lr = lr / 255.
        min_lr = lr.min()
        max_lr = lr.max()
        lr = (lr - min_lr)/(max_lr-min_lr)
        # lr = lr[0:512, 0:512]
        lr = transform(lr).to(dev)
    return lr, lr_size, min_lr, max_lr

def plot_img(image_numpy, title='output'):
    image_numpy = image_numpy.squeeze(0)
    im_max = image_numpy.max()
    im_min = image_numpy.min()
    img = (image_numpy-im_min)/(im_max-im_min)
    img = gamma_correction(img)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

def save_image(tensor, img_path, input_size, min_lr, max_lr):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = image*(max_lr-min_lr)+min_lr
    plot_img(image, title='output')
    image = transforms.ToPILImage()(image.float())
    image = image.resize(input_size, Image.BICUBIC)
    # image.save(os.path.splitext(img_path)[0]+'_output.png')

# model.to(dev)
# model.load_state_dict(torch.load('/home/sc/xin/de_background/models/450/450_nest_xinloss_119B.pth',map_location=dev))
# model.eval()

# img_ip, input_size, min_lr, max_lr= preprocess_test_img(img_ip_path, dev=dev)
# img_ip = img_ip.unsqueeze(0)
# output = model(img_ip.float())

# with torch.no_grad():
#     save_image(output,'',input_size, min_lr, max_lr)
# print(img_ip.shape, img_ip.max(), img_ip.min())
img_ip, _,_,_= preprocess_test_img(img_ip_path, dev='cpu')
img_gt, _,_,_= preprocess_test_img(img_gt_path, dev='cpu')
img_op, _,_,_ = preprocess_test_img(img_op_path,dev='cpu')
img_ms,_,_,_ = preprocess_test_img(img_ms_path,dev='cpu')

img_ip = img_ip.numpy()
img_op = img_op.numpy()
img_gt = img_gt.numpy()
img_ms = img_ms.numpy()
img_ms[img_ms>=0.9]=1
img_ms[img_ms<0.9]=0
img_op = img_op*img_ms
img_error = abs(img_gt-img_op)
plot_img(img_ip, title='op')
plot_img(img_op, title='op')
plot_img(img_gt, title='gt')
plot_img(img_ms, title='ms')
plot_img(img_error, title='error')
mse = ((img_op-img_gt)**2.).mean()
psnr=20*log10(1/sqrt(mse))
print(psnr)
# img_ip.requires_grad_()
# img_gt.requires_grad_()
# img_op.requires_grad_()
# img_ms.requires_grad_()



#
# class TVLoss(nn.Module):
#     def __init__(self, TVLoss_weight=1):
#         super(TVLoss, self).__init__()
#         self.TVLoss_weight = TVLoss_weight
#
#     def forward(self, x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self._tensor_size(x[:, :, 1:, :])
#         count_w = self._tensor_size(x[:, :, :, 1:])
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
#
#     def _tensor_size(self, t):
#         return t.size()[1] * t.size()[2] * t.size()[3]

# total_feat_out = []
# total_feat_in = []
# def hook_fn_forward(module, input, output):
#     # print(module) # 用于区分模块
#     # print('input', input) # 首先打印出来
#     print('output', output.shape)
#     total_feat_out.append(output) # 然后分别存入全局 list 中
#     # total_feat_in.append(input)
#
# modules = model.named_children() #
# for name, module in modules:
#     module.register_forward_hook(hook_fn_forward)
#
# out_img1 = model(img_ip.float())
# inp = img_ip.cpu().detach().numpy().squeeze(0).squeeze(0)
# plot_img(inp, title='input')
# out = out_img1.detach().numpy().squeeze(0).squeeze(0)
# plot_img(out, title='output')
#

# out_img1 = out_img1.mean()
# out_img1.backward()
# print('finish')
#
# for i in range(len(total_feat_out)):
#     feature = total_feat_out[i].cpu().detach().numpy()
#     feature = feature.mean(1)
#     print(feature.shape)
#     plot_img(feature, title=str(i))


# feature1 = out_img1.squeeze(0).mean(0).detach().numpy()
# plt.imshow(feature1)
# plt.show()


# total_feat_out = []
# total_feat_in = []
# out_img2 = model(img_gt)
# out_img2 = out_img2.mean()
# out_img2.backward()
# print('fff')
# feature2 = out_img2.squeeze(0).mean(0).detach().numpy()
# plt.imshow(feature2)
# plt.show()
# feature_mse = ((feature1-feature2)**2.).mean()
# print(feature_mse)
# tvloss = TVLoss()
# loss_gt = tvloss(img_gt)
# loss_op = tvloss(img_op)
# print(loss_gt, loss_op)
# img_ip = img_ip.detach().numpy().squeeze(0).mean(0)
# img_op = img_op.detach().numpy().squeeze(0).mean(0)
# img_gt = img_gt.detach().numpy().squeeze(0).mean(0)
# img_ms = img_ms.detach().numpy().squeeze(0).mean(0)
# plot_img(img_ip, title='ip')
# plot_img(img_op, title='op')
# plot_img(img_gt, title='gt')
# plot_img(img_ms, title='ms')
#
#
# mse_output = (img_op-img_gt)**2
# mse_output = mse_output.mean()
# print(mse_output)
# error_op_gt = abs(img_op-img_gt)
# plot_img(error_op_gt, title='error_op_gt')
#
#
#
# img_ms[img_ms>=0.5]=1
# img_ms[img_ms<0.5]=0
#
# img_op = img_op*img_ms
# plot_img(img_op, title='img_op*img_ms')
# mse_op_ms = (img_op-img_gt)**2
# mse_op_ms = mse_op_ms.mean()
# error_op_ms = abs(img_op-img_gt)
# plot_img(error_op_ms, title='error_op*ms')
#
# img_gt[img_gt>0]=1
# mse_mask = (img_ms - img_gt)**2
# mse_mask = mse_mask.mean()
# mask_error = abs(img_ms-img_gt)
# plot_img(mask_error, title='mask_error')
#
# print('mse_output:', mse_output)
# print('mse_op_ms:', mse_op_ms)
# print('mse_mask:', mse_mask)



# import torch
#
# x = torch.tensor([1.], requires_grad=True)
# y = torch.tensor([1.], requires_grad=True)
# b = torch.tensor([1.], requires_grad=True)
# z = x*x +3*y
# p = 2*z+6
# a = 3*p+6*b
# c = 3+a
#
# c.backward()
# print(x.grad, y.grad)