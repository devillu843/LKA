import torch.nn as nn
import torch
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
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
            # output[128, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()  # 初始化权重，自动初始化

    def forward(self, x):
        #CAM = []
        x = self.features(x)
        #CAM = x
        # 展平处理，从1维开始  与x = x.view(-1, 32*5*5)相同
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

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


# if __name__ == '__main__':
#     net = AlexNet(10)
#     device = 'cuda'
#     net.to(device)
#     summary(net, input_size=(3, 224, 224))


def test():
    x = torch.randn(1, 3, 224, 224)
    net = AlexNet(80)
    y = net(x)
    print(y.size())


test()
