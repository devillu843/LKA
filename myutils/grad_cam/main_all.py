import os
import numpy as np
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
# from model import resnet101, resnet34
from res import resnet50
from AlexNet import AlexNet
import imageio
# from Mytransform import padding_img

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def padding_img(img):
    w, h = img.size
    background = Image.new('RGB', size=(max(w, h), max(w, h)), color=(128,128,128))  # 创建背景图，颜色值为127
    length = int(abs(w - h) // 2)  # 一侧需要填充的长度
    box = (length, 0) if w < h else (0, length)  # 粘贴的位置
    background.paste(img, box)
    return background


def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # # 为什么vgg权重文件这么大，我删掉了
    # # model = models.vgg16(pretrained=True)
    # # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]
    part = 'all'
    model = resnet50(num_classes=41)
    model_weight_path = "./weights/half-test-41-different/{}/resnet50_pd128.pt".format(part)
    model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # target_layers = [model.children[0:8]]
    target_layers = list(model.children())[7]
    # print(target_layers)

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([
                                        #  padding_img(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])
    toPIL = transforms.ToPILImage()
    # load image
    # 载入图片

    with open('./json_file/class-test-41.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)
    new_json = {v:k for k,v in json_data.items()}

    # --------------------------------------#
    # 创建文件夹
    
    for i in range(1,42):
        if os.path.exists('hotsave/{}/cow_{}_128'.format(part,i)):
            pass
        else:
            os.mkdir('hotsave/{}/cow_{}_128'.format(part,i))

    for i in range(1,42):
        # img_path = "./myutils/grad_cam/400000430.jpg"
        # img_path = r"dataset\test-41-different\head\cow_1"
        img_path = "dataset/test-41-different/{}/cow_{}".format(part,i)
        target_category = int(new_json['cow_{}'.format(i)])
        files = os.listdir(img_path)
        for file in files:
            img_p = img_path +'/'+ file
            assert os.path.exists(
                img_p), "file: '{}' dose not exist.".format(img_p)
            img = Image.open(img_p).convert('RGB')
            img = padding_img(img)
            # plt.figure()
            # plt.imshow(img)
            # plt.show()
            # img = np.array(img, dtype=np.uint8)
            # img = center_crop_img(img, 224)

            # [C, H, W]
            img_tensor = data_transform(img)
            img = img.resize((224,224))
            
            # img = toPIL(img_tensor).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            # plt.figure()
            # plt.imshow(img)
            # plt.show()

            # expand batch dimension
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0)

            # 定义类及所检测目标
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

            

            grayscale_cam = cam(input_tensor=input_tensor,
                                target_category=target_category)

            grayscale_cam = grayscale_cam[0, :]  # 数值在0-1之间
            
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                            grayscale_cam,
                                            use_rgb=True)
            # path ='hotsave/{}/cow_{}_255/'.format(part,i)+str(file)+'.png'
            imageio.imwrite('hotsave/{}/cow_{}_128/'.format(part,i)+str(file)+'.png', visualization)
            # plt.imshow(visualization)
            # plt.show()


if __name__ == '__main__':
    main()
