import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import os
import time
from sklearn.metrics import classification_report
from sklearn import metrics
from torchsummary import summary


from Backbone.AlexNet import AlexNet
from Backbone.ResNet import resnet18, resnet34, resnet101, resnet152, resnet50
from Backbone.ConfusionMatrix import ConfusionMatrix
from loss.Focal_Loss import FocalLoss2d, focal_loss
from loss.soft_Dice_Loss import SoftDiceLoss

# ------------------------------------------
# 参数调整
# ------------------------------------------
batch_size = 8
epochs = 40
lr = 0.0001
num_classes = 144
test = True


# ------------------------------------------
# 模型调整
# ------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

json_file = open('class-MIX-144.json', 'r')
class_dict = json.load(json_file)
model = resnet34(num_classes=num_classes).to(device)
loss_function = nn.CrossEntropyLoss()
# loss_function = SoftDiceLoss() # 学不到内容
optimizer = torch.optim.Adam(model.parameters(), lr)

data_transform = {
    'test': transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225])
                                ])
}
image_path = './dataset/MIX'
test_dataset = datasets.ImageFolder(root=image_path + "/test",
                                    transform=data_transform['test'])
test_num = len(test_dataset)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size, shuffle=True,
                                          num_workers=0, drop_last=True)

labels = [label for _, label in class_dict.items()]
confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)
# model_weight_path = "./Alexnet/AlexNet105.pth"
model.load_state_dict(torch.load('./weights/MIX/ResNet34_train_val.pth'))
model.eval()
acc = 0
best_acc = 0
with torch.no_grad():
    y_true = []
    y_pred = []

    # predict class
    t1 = time.perf_counter()
    loop_test = tqdm(enumerate(test_loader), total=len(test_loader))
    for _, (test_images, test_labels) in loop_test:
        outputs = model(test_images.to(device))  # 指认设备
        predict_y = torch.max(outputs, dim=1)[1]

        y_true.extend(predict_y.to("cpu").numpy())
        y_pred.extend(test_labels.to("cpu").numpy())

        confusion.update(predict_y.to("cpu").numpy().astype(
            'int64'), test_labels.to("cpu").numpy().astype('int64'))
        acc += (predict_y == test_labels.to(device)).sum().item()  # 指认设备
    accurate_test = acc / test_num

    t2 = time.perf_counter() - t1

    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    print('test time:', t2)
    print('pre test time:', (t2 / test_num) * 1000)
    # print(y_true, y_pred)
    # print('hamming;', metrics.hamming_loss(y_true, y_pred))
    # print('jaccard:', metrics.jaccrd_similarity_score(y_true, y_pred))
    # print('kappa:', metrics.cohen_kappa_score(y_true, y_pred))

# confusion.plot()
# confusion.summary()
print("accurate_test:", accurate_test)
#     output = torch.squeeze(model(img))
#     predict = torch.softmax(output, dim=0)
#     predict_cla = torch.argmax(predict).numpy()
# print(class_indict[str(predict_cla)], predict[predict_cla].item())
# plt.show()
