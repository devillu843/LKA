import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
# from model import resnet101, resnet34
from AlexNet import AlexNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # # 为什么vgg权重文件这么大，我删掉了
    # # model = models.vgg16(pretrained=True)
    # # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    model = AlexNet(num_classes=41)
    model_weight_path = "./weights/test-41-different/all/Alexnet.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    target_layers = [model.features]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    # 载入图片
    img_path = "./myutils/grad_cam/400000430.jpg"
    assert os.path.exists(
        img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # 定义类及所检测目标
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    target_category = 33

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]  # 数值在0-1之间
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
