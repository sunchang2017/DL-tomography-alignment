import numpy as np
from PIL import Image
from matplotlib.pylab import plt
from scipy.signal import convolve2d

def add_noise(lr):
    lr = np.asarray(lr)

    SNR = np.random.randint(10, 20)
    N = (2*SNR)**2

    lr = lr/255.

    IIn = np.random.poisson(lam=lr*N)
    IIn = IIn/N

    noise = np.random.normal(0, 0.01, size = IIn.shape[0]*IIn.shape[1])
    noise = np.reshape(noise, IIn.shape)
    IIn = IIn+noise

    max_gu = IIn.max()
    min_gu = IIn.min()
    IIn = (IIn-min_gu)/(max_gu-min_gu)

    IIn = IIn
    # IIn = Image.fromarray(np.uint8(IIn*255))
    return IIn




def add_noise2(lr):
    lr = np.asarray(lr)
    IIn = lr/255.
    # SNR = np.random.randint(10, 20)
    # N = (2*SNR)**2
    #
    # lr = lr/255.
    # IIn = np.random.poisson(lam=lr*N)
    # IIn = IIn/N


    # max_po = IIn.max()
    # min_po = IIn.min()
    # IIn = (IIn-min_po)/(max_po-min_po)
    # print(max_po, min_po)

    print(IIn.max(), IIn.min())
    noise = np.random.normal(0, 0.01, size = IIn.shape[0]*IIn.shape[1])
    noise = np.reshape(noise, IIn.shape)
    IIn = IIn+noise

    max_gu = IIn.max()
    min_gu = IIn.min()
    IIn = (IIn-min_gu)/(max_gu-min_gu)
    print(max_gu, min_gu)
    IIn = IIn*255
    # IIn = Image.fromarray(np.uint8(IIn*255))
    return IIn

def guiyi(hr):
    hr = np.array(hr)
    hr = hr/255
    max_gu = hr.max()
    min_gu = hr.min()
    hr = (hr - min_gu) / (max_gu - min_gu)
    hr = Image.fromarray(np.uint8(hr * 255))
    return hr

def cal_barycenter(img):
    img = np.array(img)
    x = np.arange(0, img.shape[1])  # [0 1 2 3 ... 899]
    y = np.arange(0, img.shape[0])
    xx, yy = np.meshgrid(x, y)  # yy->row  xx->col
    p_sum = img.sum()
    grav_col = (img*xx).sum() / p_sum
    grav_row = (img*yy).sum() / p_sum
    return grav_row, grav_col

def cal_mse(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    return ((img1-img2)**2).mean()




#--------------------------------------------------------------------------------------
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    # if im1.dtype == np.uint8:
    #     im1 = np.double(im1)
    # if im2.dtype == np.uint8:
    #     im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))
#--------------------------------------------------------------------------------------











def plot_img(img, pos=None):
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    if pos is not None:
        plt.plot(round(pos[1]), round(pos[0]), markerfacecolor='blue', marker='o')
    plt.show()

def gamma_correction(input, gam=0.3):
    input = np.power(input, gam)
    ma = input.max()
    mi = input.min()
    return (input-mi)/(ma-mi)

def translate_label(label, rgba_color=(249, 115, 6, 255)):
    label = Image.fromarray(np.uint8(label * 255))
    label = label.convert('RGBA')  # 转换成'RGBA格式
    x, y = label.size
    for i in range(x):
        for j in range(y):
            color = label.getpixel((i, j))
            Mean = np.mean(list(color[:-1]))
            if Mean < 255:  # 我的标签区域为白色，非标签区域为黑色
                color = color[:-1] + (0,)  # 若非标签区域则设置为透明
            else:
                color = rgba_color
                # color = (137, 254, 5, 255)
                # color = (249, 115, 6, 255)
                # color = (255, 97, 0, 255)  # 标签区域设置为橙色，前3位为RGB值，最后一位为透明度情况，255为完全不透明，0为完全透明
            label.putpixel((i, j), color)
    return label

def draw_mask_on(img, label1, label2, save_path=None):
    label1 = translate_label(label1, rgba_color=(249, 115, 6, 255))
    label2 = translate_label(label2, rgba_color=((137, 254, 5, 255)))
    plt.imshow(gamma_correction(img), cmap='gray')
    plt.imshow(label1)
    plt.imshow(label2)
    plt.show()


# from matplotlib import colors
# x = colors.to_rgba('#89fe05')
# print([round(x*255) for x in x])

# input_img_path = r'C:\Users\sunchang\Desktop\obj_testimage\subs_matlab\dataset\image_proj\00001001.png'
# with Image.open(input_img_path) as lr:
#     lr = lr.convert('L')
#
#
# IIn = add_noise(lr)
# IIn = np.array(IIn)
# lr = np.array(lr)
# print(IIn.max())
# print(lr.max())
# plt.subplot(1,2,1)
# plt.imshow(lr, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(IIn, cmap='gray')
# plt.show()

# with Image.open('/home/sc/xin/de_background/test_gt/00139022.png') as f:
#     f = f.convert("L")
# print(cal_barycenter(f))
# f = f.crop((300, 150, 350+400, 150+500))
# grav_row, grav_col = cal_barycenter(f)
# print(grav_row, grav_col)
# plot_img(f, (grav_row, grav_col))