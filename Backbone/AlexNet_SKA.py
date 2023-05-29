import time
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torchsummary import summary
from thop import profile
import sys

from myutils.grad_cam.utils import GradCAM
sys.path.append('./')
from myutils.Area import I1

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False,) -> None:
        super().__init__()
        self.features1 = nn.Sequential(
            # input[3, 224, 224]  output[48, 55, 55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # output[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[128, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[192, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features2 = nn.Sequential(
            # output[128, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.maxpool = nn.Sequential(
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, num_classes)
        )
        if init_weights:
            self._initialize_weights()  # 初始化权重，自动初始化

    def forward(self, x):
        #CAM = []
        x_last2 = self.features1(x)
        x_last = self.features2(x_last2)
        x = self.maxpool(x_last)
        #CAM = x
        # 展平处理，从1维开始  与x = x.view(-1, 32*5*5)相同
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x_last2, x_last, x

    def _initialize_weights(self):
        # 遍历每一个层结构，进行权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏执置0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class AlexNet_LKA(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False, size=13, original=224) -> None:
        super().__init__()
        self.size = size
        self.origianl = original
        self.model1 = AlexNet(num_classes, init_weights)
        self.target_layers = self.model1.maxpool
        self.model2 = AlexNet(num_classes, init_weights)
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(256*6*6*2,41)


    def forward(self, x, labels):
        x_last2, x_last, x_linear1 = self.model1(x)
        coordinates_grads = []
        for i in range(x.size(0)):
            input_tensor = x[i]
            cam = GradCAM(model=self.model1, target_layers=self.target_layers, use_cuda=False)
            target_category = labels.numpy()[i]
            grayscale_cam = cam(input_tensor=input_tensor,
                                target_category=target_category)
            grayscale_cam = grayscale_cam[0, :]  # 数值在0-1之间
        
            
            m,n = np.where(grayscale_cam>0.5)
            if len(m)==0:
                m = [0,223]
                n = [0,223]

            # il1, it1, ir1, id1 = m[0], n[0], m[-1], n[-1]
            coordinates_grad = [m[0], n[0], m[-1], n[-1]]
            coordinates_grads.append(coordinates_grad)


        coordinates = torch.tensor(I1(x_last, x_last2, self.size, self.origianl))
        batch_size = len(coordinates)
        local_imgs = torch.zeros([batch_size, 3, 224, 224]).to('cuda')  # [N, 3, 448, 448]
        for i in range(batch_size):
            rec1 = coordinates[i]
            rec2 = coordinates_grads[i]
            
            iou = IoU(rec1, rec2)
            if iou==0:
                x0, y0, x1, y1 = 0, 0, 223, 223
            elif iou < 0.5:
                rec = rec1 if area(rec1)<area(rec2) else rec2
                x0 = rec[0]
                y0 = rec[1]
                x1 = rec[2]
                y1 = rec[3]
            else:
                x0 = max(rec1[0], rec2[0])
                y0 = max(rec1[1], rec2[1])
                x1 = min(rec1[2], rec2[2])
                y1 = min(rec1[3], rec2[3])

            # interpolate 上下采样函数，调整大小用
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(224,224),
                                                mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
        _, x_last3, x_linear2 = self.model2(local_imgs)

        x = torch.cat([x_last, x_last3],1)
        x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x_linear1, x, x_linear2
        # return x_linear1, x_linear2, x_last3
    

def IoU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)

def area(rec):
    return (rec[3]-rec[1]) * (rec[2]-rec[0])

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = AlexNet_three_part(num_classes=41).to(device)
    # model = AlexNet_two_part(num_classes=41).to(device)
    model = AlexNet_SKA(num_classes=41).to(device)


    input = torch.randn(1, 3, 224, 224).to(device)
    input2 = torch.randn(1, 3, 224, 224).to(device)
    input3 = torch.randn(1, 3, 224, 224).to(device)

    # summary(model, input_size=[(3, 224, 224)])
    flops, params = profile(model,inputs=(input,))

    # summary(model, input_size=[(3, 224, 224), (3, 224, 224)])
    # flops, params = profile(model,inputs=(input,input2,))

    # summary(model, input_size=[(3, 224, 224), (3, 224, 224),(3, 224, 224)])
    # flops, params = profile(model,inputs=(input,input2,input3,))

    # print(flops)
    # print(params)

    def getModelSize(model):
        param_size = 0
        param_sum = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        buffer_size = 0
        buffer_sum = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()
        all_size = (param_size + buffer_size) / 1024 / 1024
        print('模型总大小为：{:.3f}MB'.format(all_size))
        return (param_size, param_sum, buffer_size, buffer_sum, all_size)
    res = []
    test_loader = []
    for _ in range(100):
        test_loader.append(input)
    for i, out in enumerate(test_loader):
        torch.cuda.synchronize()
        start = time.time()
        predict= model(out.to(device))  # 有待修改
        torch.cuda.synchronize()
        end = time.time()
        res.append(end-start)
    time_sum = 0
    for i in res:
        time_sum += i
    print("FPS: %f"%(1.0/(time_sum/len(res))))
    print('FLOPs:',flops)
    print('params:',params)
    print(getModelSize(model))
