import os
import json
import pickle
import random
import matplotlib.pyplot as plt


def read_split_data(root: str, class_json: str = 'class_name.json', val_rate: float = 0.2, test_rate: float = 0.2, plot_image: bool = False):
    # 随机种子保证可复现
    random.seed(100)
    # 验证路径存在
    assert os.path.exists(root), 'the root:{} does not exist'.format(root)

    # 判断是否是文件夹，是则保存为一个类
    all_class = [cla for cla in os.listdir(
        root) if os.path.isdir(os.path.join(root, cla))]
    all_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(all_class))
    json_str = json.dumps(dict((key, val)
                          for val, key in class_indices.items()), indent=4)
    with open(class_json, 'w') as json_file:
        json_file.write(json_str)

    train_image_path = []
    train_image_label = []
    val_image_path = []
    val_image_label = []
    test_image_path = []
    test_image_label = []
    every_class_num = []
    support = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

    for cla in all_class:
        class_path = os.path.join(root, cla)
        # 获取图片
        images = [os.path.join(root, cla, i) for i in os.listdir(
            class_path) if os.path.splitext(i)[-1] in support]
        # 类别索引
        label = class_indices[cla]
        # 记录类别样本数量
        every_class_num.append(len(images))

        val_num = int(val_rate*len(images))
        test_num = int(test_rate*len(images))
        val_path = random.sample(images, val_num)
        test_path = random.sample(images, test_num)

        for img_path in images:
            if img_path in val_path:
                val_image_path.append(img_path)
                val_image_label.append(label)
            elif img_path in test_path:
                test_image_path.append(img_path)
                test_image_label.append(label)
            else:
                train_image_path.append(img_path)
                train_image_label.append(label)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_image_path)))
    print("{} images for validation.".format(len(val_image_path)))
    print("{} images for test.".format(len(test_image_path)))

    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(all_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(all_class)), all_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_image_path, train_image_label, val_image_path, val_image_label, test_image_path, test_image_label
