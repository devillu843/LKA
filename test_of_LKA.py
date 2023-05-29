from ast import parse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import os
import time
import argparse


from sklearn.metrics import classification_report
from sklearn import metrics
from torchsummary import summary
from myutils.read_split_data import read_split_data
from myutils.Mydataset import MyDataset
from myutils.write_into_file import pd_toExcel
from myutils.Mytransform import Gaussian,bright_contrast_color_sharpness, padding_img, pepper_salt



from Backbone.AlexNet_SKA import AlexNet_LKA
from Backbone.VGG_SKA import VGG16_SKA
from Backbone.ResNet_SKA import SKA_resnet50
from Backbone.MobileNetV3_SKA import MobileNetV3_SKA
from Backbone.ConfusionMatrix import ConfusionMatrix



# ------------------------------------------
# 参数调整
# ------------------------------------------
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--batch_size', default=8, type=int, help ='batch_size')
parser.add_argument('--epochs', default = 100, type=int, help='epoch')
parser.add_argument('--learning_rate', default=0.1, type=int, help='learning_rate')
parser.add_argument('--num_classes', default=41, type=int, help='change as your classes') 
parser.add_argument('--test', default=True, type=bool, help='if test: True')
parser.add_argument('--train', default=False, type=bool, help='if train: True')   
parser.add_argument('--model_name', default='AlexNet_LKA', type=str, help='save simply in a txt file')
parser.add_argument('--part', default='all', type=str, help='decide on your part, only in this code')
parser.add_argument('--padding_value', default=0, type=int, help='padding the image with the RGB value')
parser.add_argument('--load_pt', default='', type=str, help='continue or pretained model')
parser.add_argument('--pretrained', default=r'weights\test-41-different\head\AlexNet_pd0.pt')
args = parser.parse_args()




# ------------------------------------------
# 模型调整
# ------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
s = eval(args.model_name)
model = s(num_classes=args.num_classes).to(device)

if args.load_pt:
    model.load_state_dict(torch.load(args.load_pt))


loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)



# ------------------------------------------
# 存储调整
# ------------------------------------------
image_path = './dataset/test-41-different/{}'.format(args.part)
# 数据结果存储路径
write_home = './logs/test-41-different/{}'.format(args.part)
write_name = '/{}_pd{}/'.format(args.model_name,args.padding_value)
write_path = write_home + write_name
# 权重文件存储路径
save_home = './weights/test-41-different/{}'.format(args.part)
save_name = '/{}_pd{}.pt'.format(args.model_name,args.padding_value)
save_path = save_home + save_name
# 分类文件
json_class = 'json_file/class-test-41.json'
excel_name = 'excel/{}-{}_pd{}_41_all_.xlsx'.format(args.part,args.model_name,args.padding_value)
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)


# ------------------------------------------
# 一堆创建路径
# ------------------------------------------
if os.path.exists(image_path):
    pass
else:
    os.mkdir(image_path)
if os.path.exists(write_home):
    pass
else:
    os.mkdir(write_home)
if os.path.exists(write_path):
    pass
else:
    os.mkdir(write_path)
if os.path.exists(save_home):
    pass
else:
    os.mkdir(save_home)


# 根据需要可写在循环内部或外部，查看相应的数据变化
now = time.localtime()
nowt = time.strftime("%Y-%m-%d-%H_%M_%S", now)


writer = SummaryWriter(log_dir=write_path+nowt)

data_transform = {
    'train': transforms.Compose([
                                padding_img(args.padding_value),
                                transforms.Resize((256,256)),
                                transforms.CenterCrop((224,224)),
                                transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平翻转
                                Gaussian(0.5,0.1,0.2),
                                bright_contrast_color_sharpness(p=0.5,bright=0.5),
                                pepper_salt(p=0.5,percentage=0.15),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                                     0.229, 0.224, 0.225]),
                                transforms.RandomErasing(0.3,(0.2,1),(0.2,3.3),value=0),
                                # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                                ]),
    'val': transforms.Compose([ 
                                padding_img(args.padding_value),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                  0.229, 0.224, 0.225])
                               ]),
    'test': transforms.Compose([
                                padding_img(args.padding_value),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225])
                                ])
}


train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data(
    root=image_path, class_json=json_class)

train_data_set = MyDataset(image_path=train_images_path,
                           label=train_images_label,
                           transform=data_transform["train"])
val_data_set = MyDataset(image_path=val_images_path,
                         label=val_images_label,
                         transform=data_transform["val"])
test_data_set = MyDataset(image_path=test_images_path,
                          label=test_images_label,
                          transform=data_transform["test"])
val_num = len(val_data_set)
train_num = len(train_data_set)
test_num = len(test_data_set)


train_loader = torch.utils.data.DataLoader(train_data_set,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           collate_fn=train_data_set.collate_fn)
val_loader = torch.utils.data.DataLoader(val_data_set,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=0,
                                         collate_fn=val_data_set.collate_fn)
test_loader = torch.utils.data.DataLoader(test_data_set,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          drop_last=False,
                                          collate_fn=test_data_set.collate_fn)

# 模型大小
# summary(model, input_size=(3, 224, 224))
train_acc = []
train_loss = []
val_acc = []
val_loss = []
# print(save_path)
if args.train:
    model.model1.load_state_dict(torch.load(args.pretrained))
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()  # 训练时dropout有效
        if epoch < 5:
            model.model1.eval()
        loss = 0.0
        acc = 0
        loop_train = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, (images, labels) in loop_train:
            optimizer.zero_grad()
            output1, outputs, output2 = model(images.to(device), labels)
            predict = torch.max(output1, dim=1)[1]
            loss_each_step = loss_function(output1,labels.to(device)) + loss_function(output2,labels.to(device)) + loss_function(outputs,labels.to(device))
            loss_each_step.backward()
            optimizer.step()


            loss = loss+loss_each_step.item()
            
            acc += (predict == labels.to(device)).sum().item()
            loop_train.set_description(f'Train Epoch [{epoch+1}/{args.epochs}]')
            loop_train.set_postfix(loss=loss)
        # 写入loss,loss值，每一个epoch记录一次
        acc_train = acc / train_num
        loss /= len(train_loader)
        train_acc.append(acc_train)
        train_loss.append(loss)
        writer.add_scalar('Train_acc', acc_train, epoch)
        writer.add_scalar('Train_loss', loss, epoch)

        model.eval()
        acc_val = 0.0
        loss = 0.0

        with torch.no_grad():
            loop_val = tqdm(enumerate(val_loader), total=len(val_loader))
            for _, (val_images, val_labels) in loop_val:
                _,outputs,_ = model(val_images.to(device), val_labels)
                predict = torch.max(outputs, dim=1)[1]
                acc_val += (predict == val_labels.to(device)).sum().item()
                loop_val.set_description(f'Val Epoch [{epoch+1}/{args.epochs}]')
                loss_each_step = loss_function(outputs, val_labels.to(device))

                loss += loss_each_step.item()
                loop_val.set_postfix(val_loss = loss)
                # loop_val.set_postfix(val_acc=acc_test)
            loss /= len(val_loader)
            acc_val = acc_val / val_num
            if acc_val >= best_acc:
                best_acc = acc_val
                torch.save(model.state_dict(), save_path)
                print('save the model:%.4f' % best_acc)

            # 写入loss,loss值，每一个epoch记录一次
            writer.add_scalar('Val_acc', acc_val, epoch)
            writer.add_scalar('Val_loss', loss_each_step, epoch)
            val_acc.append(acc_val)
            val_loss.append(loss)


    writer.close()
    print('finished. the precision of the weight is %.4f' % best_acc)
    f = 'jilu.txt'
    with open(f,"a") as file:
        file.write(args.model_name+"\n")
        file.write(str(best_acc)+"\n")
        file.write("train_acc:\n")
        file.write(str(train_acc)+'\n')
        file.write("train_loss:\n")
        file.write(str(train_loss)+'\n')
        file.write("val_acc:\n")
        file.write(str(val_acc)+'\n')
        file.write("val_loss:\n")
        file.write(str(val_loss)+'\n')



if args.test:
    json_file = open(json_class, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)
    # model_weight_path = "./Alexnet/AlexNet105.pth"
    model.load_state_dict(torch.load(save_path))
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
            outputs,_,_ = model(test_images.to(device), test_labels)  # 指认设备
            predict_y = torch.max(outputs, dim=1)[1]

            y_true.extend(predict_y.to("cpu").numpy())
            y_pred.extend(test_labels.to("cpu").numpy())

            confusion.update(predict_y.to("cpu").numpy().astype(
                'int64'), test_labels.to("cpu").numpy().astype('int64'))
            acc += (predict_y == test_labels.to(device)).sum().item()  # 指认设备
        accurate_test = acc / test_num

        t2 = time.perf_counter() - t1

        # print(classification_report(y_true, y_pred, target_names=labels, digits=4))
        print('test time:', t2)
        print('pre test time:', (t2 / test_num) * 1000)
        # print(y_true, y_pred)
        print('hamming;', metrics.hamming_loss(y_true, y_pred))
        # print('jaccard:', metrics.jaccrd_similarity_score(y_true, y_pred))
        print('kappa:', metrics.cohen_kappa_score(y_true, y_pred))

    # confusion.plot()
    # confusion.summary()
    print("accurate_test:", accurate_test)
    #     output = torch.squeeze(model(img))
    #     predict = torch.softmax(output, dim=0)
    #     predict_cla = torch.argmax(predict).numpy()
    # print(class_indict[str(predict_cla)], predict[predict_cla].item())
    # plt.show()
f = 'jilu.txt'
with open(f,"a") as file:
    file.write(str(accurate_test)+"\n")
a = 1
    