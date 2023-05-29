# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

# class AngleSoftmaxLoss(nn.Module):#设置loss，超参数gamma，最小比例，和最大比例
#     def __init__(self, gamma=0, lambda_min=5, lambda_max=1500):
#         super(AngleSoftmaxLoss, self).__init__()
#         self.gamma = gamma
#         self.it = 0
#         self.lambda_min = lambda_min
#         self.lambda_max = lambda_max

#     def forward(self, x, y): #分别是output和target
#         self.it += 1
#         cos_theta, phi_theta = x #output包括上面的[cos_theta, phi_theta]
#         y = y.view(-1, 1)

#         index = cos_theta.data * 0.0
#         index.scatter_(1, y.data.view(-1, 1), 1)#将label存成稀疏矩阵
#         index = index.byte()
#         index = Variable(index)

#         lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.it))#动态调整lambda，来调整cos(\theta)和\phi(\theta)的比例
#         output = cos_theta * 1.0
#         output[index] -= cos_theta[index]*(1.0+0)/(1 + lamb)#减去目标\cos(\theta)的部分
#         output[index] += phi_theta[index]*(1.0+0)/(1 + lamb)#加上目标\phi(\theta)的部分

#         logpt = F.log_softmax(output)
#         logpt = logpt.gather(1, y)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         loss = -1 * (1-pt)**self.gamma * logpt
#         loss = loss.mean()

#         return loss

import torch
import torch.nn as nn
import torch.nn.functional as F
class ArcSoftmax(nn.Module):
    def __init__(self,feature_num,cls_num):
         super().__init__()
         self.W=nn.Parameter(torch.randn(feature_num,cls_num))
    def forward(self, feature):
        _W = F.normalize(self.W, dim=0) # 得到W/W模，而norm(W)是的得到W的模
        _X = F.normalize(feature, dim=1)
        cosine = torch.matmul(_X, _W)/10
        s=1
        a = torch.acos(cosine*0.999)
        top = torch.exp(s* torch.cos(a + 1)*10)
        _top = torch.exp(s * cosine*10)
        bottom = torch.sum(torch.exp(s*cosine*10), dim=1,keepdim=True)

        return top / (bottom - _top + top)