import torch
import torch.nn as nn
from torch.autograd import Function

# class CenterLoss(nn.Module):
#     """Center loss.
    
#     Reference:
#     Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
#     Args:
#         num_classes (int): number of classes.
#         feat_dim (int): feature dimension.
#     """
#     def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.use_gpu = use_gpu

#         if self.use_gpu:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
#         else:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

#     def forward(self, x, labels):
#         """
#         Args:
#             x: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (batch_size).
#         """
#         batch_size = x.size(0)
#         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#         distmat.addmm_(1, -2, x, self.centers.t())

#         classes = torch.arange(self.num_classes).long()
#         if self.use_gpu: classes = classes.cuda()
#         labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
#         mask = labels.eq(classes.expand(batch_size, self.num_classes))

#         dist = distmat * mask.float()
#         loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

#         return loss


class CenterLoss(nn.Module):
    """
    paper: http://ydwen.github.io/papers/WenECCV16.pdf
    code:  https://github.com/pangyupo/mxnet_center_loss
    pytorch code: https://blog.csdn.net/sinat_37787331/article/details/80296964
    """

    def __init__(self, features_dim, num_class=10, lamda=1., scale=1.0, batch_size=64):
        """
        初始化
        :param features_dim: 特征维度 = c*h*w
        :param num_class: 类别数量
        :param lamda   centerloss的权重系数 [0,1]
        :param scale:  center 的梯度缩放因子
        :param batch_size:  批次大小
        """
        super(CenterLoss, self).__init__()
        self.lamda = lamda
        self.num_class = num_class
        self.scale = scale
        self.batch_size = batch_size
        self.feat_dim = features_dim
        # store the center of each class , should be ( num_class, features_dim)
        self.feature_centers = nn.Parameter(torch.randn([num_class, features_dim]))
        # self.lossfunc = CenterLossFunc.apply

    def forward(self, output_features, y_truth):
        """
        损失计算
        :param output_features: conv层输出的特征,  [b,c,h,w]
        :param y_truth:  标签值  [b,]
        :return:
        """
        batch_size = y_truth.size(0)
        output_features = output_features.view(batch_size, -1)
        assert output_features.size(-1) == self.feat_dim
        factor = self.scale / batch_size
        # return self.lamda * factor * self.lossfunc(output_features, y_truth, self.feature_centers))

        centers_batch = self.feature_centers.index_select(0, y_truth.long())  # [b,features_dim]
        diff = output_features - centers_batch
        loss = self.lamda * 0.5 * factor * (diff.pow(2).sum())
        #########
        return loss


class CenterLossFunc(Function):
    # https://blog.csdn.net/xiewenbo/article/details/89286462
    @staticmethod
    def forward(ctx, feat, labels, centers):
        ctx.save_for_backward(feat, labels, centers)
        centers_batch = centers.index_select(0, labels.long())
        diff = feat - centers_batch
        return diff.pow(2).sum() / 2.0

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output 是最外层的梯度, 一般=1.0
        feature, label, centers, superparams = ctx.saved_tensors
        batch_size = label.size(0)
        # 记录下想相同类别的索引, 求梯度时使用
        label_occur = dict()
        for i, label_v in enumerate(label.cpu().numpy()):
            label_occur.setdefault(int(label_v), []).append(i)

        delta_center = torch.zeros_like(centers).cuda()
        centers_batch = centers.index_select(0, label.long())
        diff = feature - centers_batch

        # 存储per class 的diff 总和
        grad_class_sum = torch.zeros([1, centers.size(-1)]).cuda()
        for label_v, sample_index in label_occur.items():
            grad_class_sum[:] = 0
            for i in sample_index:
                grad_class_sum += diff[i]
            # 求per class的梯度均值
            delta_center[label_v] = -1 * grad_class_sum / (1 + len(sample_index))

        ## forced update center, 由opt执行
        # centers -= alpha * grad_output * delta_center

        # backward输入参数和forward输出参数必须一一对应
        grad_center = grad_output * delta_center
        grad_feat = grad_output * diff
        grad_label = None
        return grad_feat, grad_label, grad_center