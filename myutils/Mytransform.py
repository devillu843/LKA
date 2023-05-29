import random
import numpy as np

from PIL import Image, ImageEnhance


class Gaussian(object):
    def __init__(self, p: float, mean: float, var: float) -> None:
        self.p = p
        self.mean = mean
        self.var = var

    def __call__(self, img):
        if np.random.random_sample(1) < self.p:
            ''' 
            添加高斯噪声
            mean : 均值 
            var : 方差
            '''
            img = np.array(img)
            img = np.array(img/255, dtype=float)  # 将像素值归一
            noise = np.random.normal(
                self.mean, self.var ** 0.5, img.shape)  # 产生高斯噪声
            img = img + noise  # 直接将归一化的图片与噪声相加

            '''
            将值限制在(-1/0,1)间，然后乘255恢复
            '''
            if img.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.

            img = np.clip(img, low_clip, 1.0)
            img = np.uint8(img*255)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')

        return img


class pepper_salt(object):
    def __init__(self, p=0.5, percentage=0.1) -> None:
        self.p = p
        self.percentage = percentage

    def __call__(self, img):
        if np.random.random_sample(1) < self.p:
            img = np.array(img)
            num = int(self.percentage*img.shape[0]*img.shape[1])  # 椒盐噪声点数量
            random.randint(0, img.shape[0])
            for _ in range(num):
                # 从0到图像长度之间的一个随机整数,因为是闭区间所以-1
                X = random.randint(0, img.shape[0]-1)
                Y = random.randint(0, img.shape[1]-1)
                # if random.randint(0,1) ==0: #黑白色概率55开
                #     img2[X,Y] = (255,255,255)#白色
                # else:
                #     img2[X,Y] =(0,0,0)#黑色
                img[X, Y] = (0, 0, 0)  # 黑色
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class Add_Pepper_Noise(object):
    """"
    Args:
        snr (float): Signal Noise Rate概率值
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr=0.92, p=0.5):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[
                                    signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            # 转化为PIL的形式
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


class bright_contrast_color_sharpness(object):
    '''
    概率，亮度，对比度，色度，锐度
    '''

    def __init__(self, p=0.5, bright=0, contrast=0, color=0, sharpness=0) -> None:
        self.bright = bright
        self.contrast = contrast
        self.color = color
        self.sharpness = sharpness
        self.p = p

    def __call__(self, img):
        if np.random.random_sample(1) < self.p:
            bright_enhancer = ImageEnhance.Brightness(img)
            img = bright_enhancer.enhance(self.bright)

            constrast_enchancer = ImageEnhance.Contrast(img)
            img = constrast_enchancer.enhance(self.contrast)

            color_enchancer = ImageEnhance.Color(img)
            img = color_enchancer.enhance(self.color)

            sharpness_enchancer = ImageEnhance.Contrast(img)
            img = sharpness_enchancer.enhance(self.sharpness)

        return img


class padding_img(object):
    def __init__(self, padding) -> None:
        self.padding = padding

    def __call__(self, img):
        img = img.convert('RGB')
        w, h = img.size
        background = Image.new('RGB', size=(max(w, h), max(w, h)), color=(
            self.padding, self.padding, self.padding))  # 创建背景图，颜色值为127
        length = int(abs(w - h) // 2)  # 一侧需要填充的长度
        box = (length, 0) if w < h else (0, length)  # 粘贴的位置
        background.paste(img, box)
        return background


class add_railing(object):
    def __init__(self, padding) -> None:
        self.padding = padding

    def __call__(self, img):
        img1 = img.convert('RGB')
        w, h = img.size
        ma = max(w, h)
        img2 = Image.open('栏杆640.png')
        img2 = img2.convert('RGB')
        img2 = img2.resize((ma, ma))
        im = np.array(img2)
        img_ = np.array(img1)
        for i in range(ma):
            for j in range(ma):
                if im[i, j, :].all() != np.array([0, 0, 0]).all():
                    img_[i, j, :] = im[i, j, :]
        return Image.fromarray(img_.astype('uint8')).convert('RGB')
