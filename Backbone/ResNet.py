import time
from turtle import forward
import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        #  downsample=None 短接层
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 先短接在relu
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):  # channel第一层channel个数
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel,
                      downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet152(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)



class ResNet_two_part(nn.Module):
    def __init__(self, block, blocks_num, num_classes, include_top) -> None:
        super().__init__()
        self.res1 = ResNet(block, blocks_num, num_classes, include_top)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(block.expansion * 512 * 2, num_classes)

    def forward(self, x1, x2):
        x1 = self.res1(x1)
        x1 = self.avgpool(x1)
        x2 = self.res1(x2)
        x2 = self.avgpool(x2)
        x = torch.cat([x1,x2], dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18_two(num_classes=1000, include_top=False):
    return ResNet_two_part(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34_two(num_classes=1000, include_top=False):
    return ResNet_two_part(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50_two(num_classes=1000, include_top=False):
    return ResNet_two_part(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101_two(num_classes=1000, include_top=False):
    return ResNet_two_part(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet152_two(num_classes=1000, include_top=False):
    return ResNet_two_part(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


class ResNet_two_part_alone(nn.Module):
    def __init__(self, block, blocks_num, num_classes, include_top) -> None:
        super().__init__()
        self.res1 = ResNet(block, blocks_num, num_classes, include_top)
        self.res2 = ResNet(block, blocks_num, num_classes, include_top)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(block.expansion * 512 * 2, num_classes)

    def forward(self, x1, x2):
        x1 = self.res1(x1)
        x1 = self.avgpool1(x1)
        x2 = self.res2(x2)
        x2 = self.avgpool1(x2)
        x = torch.cat([x1,x2], dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



def resnet18_two_alone(num_classes=1000, include_top=False):
    return ResNet_two_part_alone(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34_two_alone(num_classes=1000, include_top=False):
    return ResNet_two_part_alone(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50_two_alone(num_classes=1000, include_top=False):
    return ResNet_two_part_alone(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101_two_alone(num_classes=1000, include_top=False):
    return ResNet_two_part_alone(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet152_two_alone(num_classes=1000, include_top=False):
    return ResNet_two_part_alone(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


class ResNet_three_part(nn.Module):
    def __init__(self, block, blocks_num, num_classes, include_top) -> None:
        super().__init__()
        self.res = ResNet(block, blocks_num, num_classes, include_top)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(block.expansion * 512 * 3, num_classes)

    def forward(self, x1, x2, x3):
        x1 = self.res(x1)
        x1 = self.avgpool(x1)
        x2 = self.res(x2)
        x2 = self.avgpool(x2)
        x3 = self.res(x3)
        x3 = self.avgpool(x3)
        x = torch.cat([x1,x2,x3], dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_three(num_classes=1000, include_top=True):
    return ResNet_three_part(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34_three(num_classes=1000, include_top=True):
    return ResNet_three_part(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50_three(num_classes=1000, include_top=True):
    return ResNet_three_part(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101_three(num_classes=1000, include_top=True):
    return ResNet_three_part(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet152_three(num_classes=1000, include_top=True):
    return ResNet_three_part(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


class ResNet_three_part_alone(nn.Module):

    def __init__(self, block, blocks_num, num_classes, include_top) -> None:
        super().__init__()
        self.res1 = ResNet(block, blocks_num, num_classes, include_top)
        self.res2 = ResNet(block, blocks_num, num_classes, include_top)
        self.res3 = ResNet(block, blocks_num, num_classes, include_top)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(block.expansion * 512 * 3, num_classes)

    def forward(self, x1, x2, x3):
        x1 = self.res1(x1)
        x1 = self.avgpool1(x1)
        x2 = self.res2(x2)
        x2 = self.avgpool2(x2)
        x3 = self.res3(x3)
        x3 = self.avgpool3(x3)
        x = torch.cat([x1,x2,x3], dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_three_alone(num_classes=1000, include_top=True):
    return ResNet_three_part_alone(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34_three_alone(num_classes=1000, include_top=True):
    return ResNet_three_part_alone(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50_three_alone(num_classes=1000, include_top=True):
    return ResNet_three_part_alone(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101_three_alone(num_classes=1000, include_top=True):
    return ResNet_three_part_alone(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet152_three_alone(num_classes=1000, include_top=True):
    return ResNet_three_part_alone(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)

# def getModelSize(model):
#     param_size = 0
#     param_sum = 0
#     for param in model.parameters():
#         param_size += param.nelement() * param.element_size()
#         param_sum += param.nelement()
#     buffer_size = 0
#     buffer_sum = 0
#     for buffer in model.buffers():
#         buffer_size += buffer.nelement() * buffer.element_size()
#         buffer_sum += buffer.nelement()
#     all_size = (param_size + buffer_size) / 1024 / 1024
#     print('模型总大小为：{:.3f}MB'.format(all_size))
#     return (param_size, param_sum, buffer_size, buffer_sum, all_size)

if __name__ == '__main__':

    from thop import profile
    from torchsummary import summary
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = AlexNet_three_part(num_classes=41).to(device)
    # model = AlexNet_two_part(num_classes=41).to(device)
    model = resnet101(num_classes=41).to(device)


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