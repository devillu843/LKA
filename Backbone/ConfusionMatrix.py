
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use('TkAgg')
from prettytable import PrettyTable


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        Pre = []
        Rec = []
        macio_Pre = []
        macio_Rec = []

        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "TP_FN_Pre", "TP_FN_Rec"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 8)
            Recall = round(TP / (TP + FN), 8)
            Specificity = round(TN / (TN + FP), 8)
            if TP==0 :
                Precision = 0
                Recall=0
                Specificity=0

            TP_FN_Pre = round((Precision*(TP + FN) / np.sum(self.matrix)), 8)
            TP_FN_Rec = round((Recall * (TP + FN) / np.sum(self.matrix)), 8)


            Pre.append(TP_FN_Pre)
            Rec.append(TP_FN_Rec)
            macio_Pre.append(Precision)
            macio_Rec.append(Recall)
            # avg_pre = sum(Pre)



            table.add_row([self.labels[i], Precision, Recall, Specificity, TP_FN_Pre, TP_FN_Rec])
        print(table)
        print('avg_pre:', sum(Pre))
        print('avg_rec:', sum(Rec))
        print('macio_Pre:', (sum(macio_Pre)) / self.num_classes)
        print('macio_Rec:', (sum(macio_Rec)) / self.num_classes)





    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()