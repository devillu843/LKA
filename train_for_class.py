from wsgiref.simple_server import demo_app
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
from myutils.Mytransform import Gaussian, bright_contrast_color_sharpness, padding_img, pepper_salt


from Backbone.AlexNet import AlexNet
from Backbone.VGG import VGG16
from Backbone.ResNet import resnet50
from Backbone.MobileNetV3 import MobileNetV3
from Backbone.ConfusionMatrix import ConfusionMatrix


# ------------------------------------------
# 参数调整
# ------------------------------------------
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
parser.add_argument('--epochs', default=100, type=int, help='epoch')
parser.add_argument('--learning_rate', default=0.0001,
                    type=int, help='learning_rate')
parser.add_argument('--num_classes', default=41, type=int,
                    help='change as your classes')
parser.add_argument('--test', default=True, type=bool, help='if test: True')
parser.add_argument('--train', default=True, type=bool, help='if train: True')
parser.add_argument('--model_name', default='AlexNet',
                    type=str, help='save simply in a txt file')
parser.add_argument('--part', default='all', type=str,
                    help='decide on your part, only in this code')
parser.add_argument('--padding_value', default=0, type=int,
                    help='padding the image with the RGB value')
parser.add_argument(
    '--json_file', default='./json_file/class-test-41.json', type=str, help='json file')
parser.add_argument('--save_txt_file', default='./jilu.txt',
                    type=str, help='save the result')
args = parser.parse_args()


# ------------------------------------------
# 模型调整
# ------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
if args.model_name == 'resnet50':
    model = resnet50(num_classes=args.num_classes).to(device)

elif args.model_name == 'AlexNet':
    model = AlexNet(num_classes=args.num_classes).to(device)

elif args.model_name == 'VGG16':
    model = VGG16(num_classes=args.num_classes).to(device)

elif args.model_name == 'MobileNetV3':
    model = MobileNetV3(num_classes=args.num_classes).to(device)

# if args.load_pt:
#     model.load_state_dict(torch.load(args.load_pt))
loss_function = nn.CrossEntropyLoss()
# loss_function = SoftDiceLoss() # 学不到内容
optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
# stepLR
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)
# 指数衰减
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
# 余弦退火
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=0.001)


# ------------------------------------------
# 存储调整
# ------------------------------------------
image_path = './dataset/test-41-different/{}'.format(args.part)
# 数据结果存储路径
write_home = './logs/test-41-different/{}'.format(args.part)
write_name = '/{}_pd{}/'.format(args.model_name, args.padding_value)
write_path = write_home + write_name
# 权重文件存储路径
save_home = './weights/test-41-different/{}'.format(args.part)
save_name = '/{}_pd{}.pt'.format(args.model_name, args.padding_value)
save_path = save_home + save_name
# 分类文件
json_class = args.json_file
excel_name = 'excel/{}-{}_pd{}_41_all_.xlsx'.format(
    args.model_name, args.part, args.padding_value)
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)


# ------------------------------------------
# 一堆创建路径
# ------------------------------------------

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

# os.makedirs(write_path+'train_val'+nowt)
writer = SummaryWriter(log_dir=write_path+nowt)
# 写入特征层，特征图大小
# if torch.cuda.is_available():
#     graph_inputs = torch.from_numpy(np.random.rand(
#         1, 3, input_shape[0], input_shape[1])).type(torch.FloatTensor).cuda()
# else:
#     graph_inputs = torch.from_numpy(np.random.rand(
#         1, 3, input_shape[0], input_shape[1])).type(torch.FloatTensor)
# write.add_graph(model, (graph_inputs,))
# 写入loss,loss值，每一个step记录一次
# write.add_scalar('Train_loss', loss, (epoch*epoch_size+iteration))


data_transform = {
    'train': transforms.Compose([
                                padding_img(args.padding_value),
                                transforms.Resize((224, 224)),
                                # transforms.CenterCrop((224,224)),
                                transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平翻转
                                Gaussian(0.5, 0.1, 0.2),
                                bright_contrast_color_sharpness(
                                    p=0.5, bright=0.5),
                                pepper_salt(p=0.5, percentage=0.15),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                                     0.229, 0.224, 0.225]),
                                transforms.RandomErasing(
                                    0.3, (0.2, 1), (0.2, 3.3), value=0),
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


# image_path = './dataset/test-14'
# train_dataset = datasets.ImageFolder(
#     root=image_path + '/train', transform=data_transform['train'])
# train_num = len(train_dataset)

# # 获取类
# class_list = train_dataset.class_to_idx
# # print(class_list.items())
# class_dict = dict((key, value) for value, key in class_list.items())
# # 写如json文件
# json_str = json.dumps(class_dict, indent=4)
# with open(json_class, 'w') as json_file:
#     json_file.write(json_str)


# train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data(
#     root=image_path, class_json=json_class)


# 模型大小
# summary(model, input_size=(3, 224, 224))

# print(save_path)

train_acc = []
train_loss = []
val_acc = []
val_loss = []
if args.train:
    with open('txt_file/{}/train.txt'.format(args.part), 'r') as file:
        train_path = file.read().splitlines()
        train_images_path, train_images_label = [], []
        for path in train_path:
            train_images_path.append(path.split(' ')[0])
            train_images_label.append(int(path.split(' ')[1]))
    with open('txt_file/{}/val.txt'.format(args.part), 'r') as file:
        val_path = file.read().splitlines()
        val_images_path, val_images_label = [], []
        for path in train_path:
            val_images_path.append(path.split(' ')[0])
            val_images_label.append(int(path.split(' ')[1]))
    train_data_set = MyDataset(image_path=train_images_path,
                               label=train_images_label,
                               transform=data_transform["train"])
    val_data_set = MyDataset(image_path=val_images_path,
                             label=val_images_label,
                             transform=data_transform["val"])
    val_num = len(val_data_set)
    train_num = len(train_data_set)
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
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()  # 训练时dropout有效
        loss = 0.0
        acc = 0
        loop_train = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, (images, labels) in loop_train:
            optimizer.zero_grad()
            outputs = model(images.to(device))
            predict = torch.max(outputs, dim=1)[1]
            # outputs = nn.functional.log_softmax(outputs)
            # output = nn.functional.soft_margin_loss(outputs,labels.to(device))
            loss_each_step = loss_function(outputs, labels.to(device))
            loss_each_step.backward()
            optimizer.step()
            # scheduler.step()

            loss += loss_each_step.item()
            acc += (predict == labels.to(device)).sum().item()
            loop_train.set_description(
                f'Train Epoch [{epoch+1}/{args.epochs}]')
            loop_train.set_postfix(loss=loss)
        # 写入loss,loss值，每一个epoch记录一次
        acc_train = acc / train_num
        loss /= len(train_loader)
        writer.add_scalar('Train_acc', acc_train, epoch)
        writer.add_scalar('Train_loss', loss, epoch)
        train_acc.append(acc_train)
        train_loss.append(loss)

        model.eval()
        acc = 0.0
        loss = 0
        with torch.no_grad():
            loop_val = tqdm(enumerate(val_loader), total=len(val_loader))
            for _, (val_images, val_labels) in loop_val:
                outputs = model(val_images.to(device))
                predict = torch.max(outputs, dim=1)[1]
                acc += (predict == val_labels.to(device)).sum().item()
                loop_val.set_description(
                    f'Val Epoch [{epoch+1}/{args.epochs}]')
                loss_each_step = loss_function(outputs, val_labels.to(device))

                loss += loss_each_step.item()
                loss_each_step = loss / val_num
                loop_val.set_postfix(val_loss=loss)
                # loop_val.set_postfix(val_acc=acc_test)
            acc_val = acc / val_num
            loss_each_step = loss / val_num
            if acc_val >= best_acc:
                best_acc = acc_val
                torch.save(model.state_dict(), save_path)
                print('save the model:%.4f' % best_acc)

            # 写入loss,loss值，每一个epoch记录一次
            writer.add_scalar('Val_acc', acc_val, epoch)
            writer.add_scalar('Val_loss', loss_each_step, epoch)
            val_acc.append(acc_val)
            val_loss.append(loss_each_step)

    writer.close()
    print('finished. the precision of the weight is %.4f' % best_acc)
    f = args.save_txt_file
    with open(f, "a") as file:
        file.write(args.model_name+' '+args.part+"\n")
        file.write(str(best_acc)+"\n")
        file.write("loss:\n")
        file.write(str(train_loss)+'\n')
        file.write("train_acc:\n")
        file.write(str(train_acc)+'\n')
        file.write("train_loss:\n")
        file.write(str(train_loss)+'\n')
        file.write("val_acc:\n")
        file.write(str(val_acc)+'\n')
        file.write("val_loss:\n")
        file.write(str(val_loss)+'\n')


if args.test:
    with open('txt_file/{}/test.txt'.format(args.part), 'r') as file:
        test_path = file.read().splitlines()
        test_images_path, test_images_label = [], []
        for path in train_path:
            test_images_path.append(path.split(' ')[0])
            test_images_label.append(int(path.split(' ')[1]))
    test_data_set = MyDataset(image_path=test_images_path,
                              label=test_images_label,
                              transform=data_transform["test"])
    test_num = len(test_data_set)

    test_loader = torch.utils.data.DataLoader(test_data_set,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              drop_last=False,
                                              collate_fn=test_data_set.collate_fn)
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

        picture = []
        id1 = []
        p1 = []
        id2 = []
        p2 = []
        id3 = []
        p3 = []
        for path in test_images_path:
            picture.append(os.path.basename(path))
        batch_num = 1

        for _, (test_images, test_labels) in loop_test:
            outputs = model(test_images.to(device))  # 指认设备
            predict_y = torch.max(outputs, dim=1)[1]

            # -----------------------------------------------
            # 保存k个数据，与训练本身无关，郑老师的要求,写入excel文件
            k = 3
            output = F.softmax(outputs)
            out, pred_num = output.topk(k=k)

            if batch_num*args.batch_size <= test_num:
                image_num_in_batch = args.batch_size
                batch_num += 1
            else:
                image_num_in_batch = args.batch_size - args.batch_size*batch_num + test_num

            for i in range(image_num_in_batch):
                for j in range(k):
                    # print(pred_num[i][j])
                    cow_name = class_indict[str(
                        pred_num[i][j].to('cpu').numpy())]
                    probability = out[i][j].to('cpu').numpy() * 100
                    if j == 0:
                        id1.append(cow_name)
                        p1.append(probability)
                    elif j == 1:
                        id2.append(cow_name)
                        p2.append(probability)
                    else:
                        id3.append(cow_name)
                        p3.append(probability)
            # -----------------------------------------------

            y_true.extend(predict_y.to("cpu").numpy())
            y_pred.extend(test_labels.to("cpu").numpy())

            confusion.update(predict_y.to("cpu").numpy().astype(
                'int64'), test_labels.to("cpu").numpy().astype('int64'))
            acc += (predict_y == test_labels.to(device)).sum().item()  # 指认设备
        accurate_test = acc / test_num
        pd_toExcel(picture, id1, p1, id2, p2, id3, p3, excel_name)

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

f = args.save_txt_file
with open(f, "a") as file:
    file.write(str(accurate_test)+"\n")
a = 1
