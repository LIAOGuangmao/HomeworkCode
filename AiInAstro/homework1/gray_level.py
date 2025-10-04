# Name: Guangmao Liao ID: 652024260005 Experiment 1: Gray-Level Equalization and Stretching
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
#-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-#
# The Algorithm for Histgram-Equalization
def histogram_equalization(img):
    img_hist = np.histogram(img.flatten(), bins=256, range=(0,256))[0] # histogram of gray-level distribution
    img_hist_norm = img_hist / len(img.flatten()) # normalize to [0,1]
    img_hist_norm_cdf = np.cumsum(img_hist_norm) # Cumulative Distribution Function
    img_hist_norm_cdf_255 = np.round(img_hist_norm_cdf * 255) # scale to [0,255]
    img_equ = np.zeros_like(img, dtype=np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img_equ[y][x] = img_hist_norm_cdf_255[img[y][x]] # a non-linear mapping based on the CDF
    return img_equ, img_hist_norm_cdf, img_hist_norm_cdf_255

def compare_cv2(filepath, figsave=False):
    img = cv2.imread(filename=filepath, flags=cv2.IMREAD_GRAYSCALE)  # Read initial grayscale image from the path of the file
    img_equ_cv2 = cv2.equalizeHist(img)  # use cv2.equalizeHist as a reference
    img_equ, img_hist_norm_cdf, img_hist_norm_cdf_255 = histogram_equalization(img) # use the self-defined function to do the histogram equalization

    fig, axes = plt.subplots(3,3, figsize=(10,9), layout='constrained')
    axes[0,0].imshow(img, cmap='gray')
    axes[0,1].imshow(img_equ_cv2, cmap='gray')
    axes[0,2].imshow(img_equ, cmap='gray')
    axes[1,0].hist(img.flatten(), bins=256, range=(0,256))
    axes[1,1].hist(img_equ_cv2.flatten(), bins=256, range=(0,256))
    axes[1,2].hist(img_equ.flatten(), bins=256, range=(0,256))
    axes[0,0].set_title('Initial image')
    axes[0,1].set_title('cv2.equalizeHist')
    axes[0,2].set_title('self-defined function')
    axes[2,0].imshow(img_equ-img_equ_cv2, cmap=ListedColormap(['white', 'r']))
    h = np.histogram(img_equ.flatten(), bins=256, range=(0, 256))[0]
    h_cv2 = np.histogram(img_equ_cv2.flatten(), bins=256, range=(0, 256))[0]
    h_diff = h - h_cv2
    axes[2,1].bar(np.linspace(0,255,256),h_diff)
    axes[2,2].plot(np.linspace(0,255,256), img_hist_norm_cdf, ds='steps-pre')
    idx = np.where(h_diff!=0)[0]
    # print(idx)
    for i in idx:
        if h_diff[i] > 0:
            axes[2,2].axvline(i, color='r')
            axes[2,2].axhline(img_hist_norm_cdf[i], color='r')
            j = np.where(img_hist_norm_cdf_255==i)[0][0]
            axes[2,2].text(5, 1.1*img_hist_norm_cdf[i], '{:.6f} * 255 = {:.6f}'.format(img_hist_norm_cdf[j],img_hist_norm_cdf[j]*255))
            axes[2,1].text(1.1*i, 0.5*h_diff[i], str(i))
        else:
            axes[2,1].text(0.7*i, 0.5*h_diff[i], str(i))
    axes[2,0].set_title('Image differences: self - cv2')
    axes[2,1].set_title('Hist differences: self - cv2')
    axes[2,2].set_title('Round-off error')

    plt.show()
    if figsave:
        fig.savefig(filepath.split('.')[0]+'-equalizeHist.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.2)


compare_cv2('Lena.bmp')
compare_cv2('Goldhill.bmp')
compare_cv2('Cameraman.bmp')

#-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-#
# The Algorithm for Histogram-Stretching
def Hist_stretch(img, clip=(0,255), method=('linear',1)):
    img_clip = np.clip(img,clip[0],clip[1]) # clip the data between the range we are interested
    img_norm = img_clip/255 # normalize to [0,1]
    if method[0] == 'linear':
        img_s = (img_norm-img_norm.min())/(img_norm.max()-img_norm.min())*255 # y = (x-xmin)/(xmax-xmin) * 255
        return np.round(img_s).astype(np.uint8)
    if method[0] == 'gamma':
        img_gam = np.power(img_norm,method[1]) # y = x^{\gamma}
        img_s = (img_gam-img_gam.min())/(img_gam.max()-img_gam.min())*255 # scale to [0,255]
        return np.round(img_s).astype(np.uint8)
    if method[0] == 'log':
        img_log = np.log(method[1]*img_norm+1)/(np.log(method[1]+1)) # y = log_{\alpha+1}(\alpha x+1)
        img_s = (img_log-img_log.min())/(img_log.max()-img_log.min())*255
        return np.round(img_s).astype(np.uint8)
    if method[0] == 'power':
        img_pow = (np.power(method[1],img_norm)-1)/(method[1]-1) # y = (\alpha^{x}-1)/(\alpha-1)
        img_s = (img_pow-img_pow.min())/(img_pow.max()-img_pow.min())*255
        return np.round(img_s).astype(np.uint8)

def visual_s(filepath, clip=(0,255), para=(0.25,-0.25,1000,2), figsize=(11,10), figsave=False):
    img = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
    img_lin = Hist_stretch(img, clip=clip, method=('linear',1))
    img_gam1 = Hist_stretch(img, clip=clip, method=('gamma',para[0]))
    img_gam2 = Hist_stretch(img, clip=clip, method=('gamma',para[1]))
    img_log = Hist_stretch(img, clip=clip, method=('log',para[2]))
    img_pow = Hist_stretch(img, clip=clip, method=('power',para[3]))

    fig, axes = plt.subplots(4, 3, figsize=figsize, layout='constrained')
    axes[0,0].imshow(img, cmap='gray')
    axes[0,1].imshow(img_lin, cmap='gray')
    axes[0,2].imshow(img_gam1, cmap='gray')
    axes[1,0].hist(img.flatten(), bins=256, range=(0, 256))
    axes[1,1].hist(img_lin.flatten(), bins=256, range=(0, 256))
    axes[1,2].hist(img_gam1.flatten(), bins=256, range=(0, 256))
    axes[1,0].axvline(clip[0], color='r')
    axes[1,0].axvline(clip[1], color='r')
    axes[1,0].text((clip[0]+clip[1])/(2*255), 0.9, 'clip', color='r', transform=axes[1,0].transAxes)
    axes[0,0].set_title('Initial image')
    axes[0,1].set_title('Linear: $y=kx+b$')
    axes[0,2].set_title(r'Gamma: $y=x^{\gamma},\,\gamma=$'+str(para[0]))
    axes[2,0].imshow(img_gam2, cmap='gray')
    axes[2,1].imshow(img_log, cmap='gray')
    axes[2,2].imshow(img_pow, cmap='gray')
    axes[3,0].hist(img_gam2.flatten(), bins=256, range=(0, 256))
    axes[3,1].hist(img_log.flatten(), bins=256, range=(0, 256))
    axes[3,2].hist(img_pow.flatten(), bins=256, range=(0, 256))
    axes[2,0].set_title(r'Gamma: $y=x^{\gamma},\,\gamma=$'+str(para[1]))
    axes[2,1].set_title(r'Log: $y=log_{\alpha+1}(\alpha x+1),\,\alpha=$'+str(para[2]))
    axes[2,2].set_title(r'Power: $y=(\alpha^{x}-1)/(\alpha-1),\,\alpha=$'+str(para[3]))

    plt.show()
    if figsave:
        fig.savefig(filepath.split('.')[0] + '-GrayStretch.png', format='png', dpi=600, bbox_inches='tight',pad_inches=0.2)

visual_s('bao-menglong-Jub889kTwnk-unsplash.jpg', clip=(1,90), para=(0.25,-0.25,1000,2))
visual_s('danila-rassokhin-FRX8ne-tYUI-unsplash.jpg', clip=(1,40), para=(0.2,-0.2,1500,5), figsize=(11,16))
visual_s('ilya-yakubovich-h_aN4HB1wCI-unsplash.jpg', clip=(8,40), para=(0.25,-0.25,1000,1.5))
