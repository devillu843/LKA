import time
import torch
import torch.nn as nn
# from torch.hub import load_state_dict_from_url

# # ------------------------------------------------------------------------------
# # 暴露接口
# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]
# # ------------------------------------------------------------------------------
# # 配置字典
# cfgs = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
# # ------------------------------------------------------------------------------
# # 预训练权重下载地址
# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
# }


# class VGG(nn.Module):
#     '''
#     VGG通用模型
#     '''

#     def __init__(self, features, num_classes=1000, init_weight=True) -> None:
#         super(VGG, self).__init__()

#         # 提取特征
#         self.features = features

#         # 自适应平均池化，特征图池化到7x7大小
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             # 默认in_place为False，True会改变输入的值，节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
#             nn.ReLU(True),
#             nn.Dropout(),  # 默认0.5
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes)
#         )

#         # 权重初始化
#         if init_weight:
#             self._initialize_weights()

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # 卷积层使用Kaimming初始化
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_out', nonlinearity='relu')
#                 # 偏置初始化为0
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)


# def make_layers(cfg, batch_norm=False):
#     '''
#     根据配置表，返回模型列表
#     '''
#     layers = []  # 层列表初始化

#     in_channels = 3  # 输入3通道图像

#     # 遍历配置列表
#     for v in cfg:
#         if v == 'M':  # 添加池化层
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:  # 添加卷积层
#             # 3×3 卷积
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             # 卷积-->批归一化（可选）--> ReLU激活
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             # 通道数方面，下一层输入即为本层输出
#             in_channels = v
#     # 以sequencial类型返回模型层列表
#     return nn.Sequential(*layers)


# def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
#     '''
#     通用模型构造器
#     '''
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

#     if pretrained:
#         state_dict = load_state_dict_from_url(
#             model_urls[arch], progress=progress)
#         model.load_state_dict(state_dict)
#     return model


# def vgg11(pretrained=False, progress=True, **kwargs):
#     return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


# def vgg11_bn(pretrained=False, progress=True, **kwargs):
#     return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


# def vgg13(pretrained=False, progress=True, **kwargs):
#     return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


# def vgg13_bn(pretrained=False, progress=True, **kwargs):
#     return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


# def vgg16(pretrained=False, progress=True, **kwargs):
#     return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


# def vgg16_bn(pretrained=False, progress=True, **kwargs):
#     return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


# def vgg19(pretrained=False, progress=True, **kwargs):
#     return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


# def vgg19_bn(pretrained=False, progress=True, **kwargs):
#     return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)


# def test():
#     x = torch.randn(1, 3, 224, 224)
#     net = vgg16(num_classes=80)
#     y = net(x)
#     print(y.size())


# # test()

class VGG16(nn.Module):
    def __init__(self, num_classes=1000, last_height=7):
        super(VGG16, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32    32*32*64      
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)   #(32-2)/2+1=16         16*16*64
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),  #(16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), #(16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)    #(16-2)/2+1=8     8*8*128
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)     #(8-2)/2+1=4      4*4*256
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),  #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)    #(4-2)/2+1=2     2*2*512
        )

        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)   #(2-2)/2+1=1      1*1*512
        )
        self.fc=nn.Sequential(
            nn.Linear(512*last_height*last_height,num_classes)
            # nn.Linear(512* last_height * last_height,512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(512,256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x,1)
        x=self.fc(x)
        return x


class VGG16_for_more(nn.Module):
    def __init__(self, num_classes=1000, last_height=7):
        super().__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32    32*32*64      
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)   #(32-2)/2+1=16         16*16*64
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),  #(16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), #(16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)    #(16-2)/2+1=8     8*8*128
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)     #(8-2)/2+1=4      4*4*256
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),  #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)    #(4-2)/2+1=2     2*2*512
        )

        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)   #(2-2)/2+1=1      1*1*512
        )
        

    def forward(self,x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return x


class VGG16_two_part(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.vgg = VGG16_for_more(num_classes)
        self.fc=nn.Sequential(
            nn.Linear(512*7*7*2,num_classes)
            # nn.Linear(512* 2 * 7* 7,512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(512,256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(256,num_classes)
        )
    
    def forward(self, x1, x2):
        x1 = self.vgg(x1)
        x2 = self.vgg(x2)
        x = torch.cat([x1, x2], dim=1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class VGG16_two_part_alone(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.vgg1 = VGG16_for_more(num_classes)
        self.vgg2 = VGG16_for_more(num_classes)
        self.fc=nn.Sequential(
            nn.Linear(512* 2 * 7* 7,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )
    
    def forward(self, x1, x2):
        x1 = self.vgg1(x1)
        x2 = self.vgg2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class VGG16_three_part(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.vgg = VGG16_for_more(num_classes)
        self.fc=nn.Sequential(
            nn.Linear(512* 3 * 7* 7,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )
    
    def forward(self, x1, x2, x3):
        x1 = self.vgg(x1)
        x2 = self.vgg(x2)
        x3 = self.vgg(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class VGG16_three_part_alone(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.vgg1 = VGG16_for_more(num_classes)
        self.vgg2 = VGG16_for_more(num_classes)
        self.vgg3 = VGG16_for_more(num_classes)
        self.fc=nn.Sequential(
            nn.Linear(512* 3 * 7* 7,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )
    
    def forward(self, x1, x2, x3):
        x1 = self.vgg1(x1)
        x2 = self.vgg2(x2)
        x3 = self.vgg3(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


if __name__ == '__main__':

    from thop import profile
    from torchsummary import summary
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = AlexNet_three_part(num_classes=41).to(device)
    # model = AlexNet_two_part(num_classes=41).to(device)
    model = VGG16(num_classes=41).to(device)


    input = torch.randn(1, 3, 224, 224).to(device)
    input2 = torch.randn(1, 3, 224, 224).to(device)
    input3 = torch.randn(1, 3, 224, 224).to(device)

    summary(model, input_size=[(3, 224, 224)])
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
