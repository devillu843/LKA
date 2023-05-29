import cv2
import numpy as np
import torch


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """
    # 我试着翻译一下：从中间层中提取激活函数并记录梯度

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                # 每一个层注册一个正向传播的钩子函数，保存输出结果
                # 为什么这么写？官方文档中看去吧，不想看就留个印象
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            # hasattr() 函数用于判断对象是否包含对应的属性
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    # 每一个层注册一个反向传播的钩子函数
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    '''
    惊天大问题，up说下面两个函数保存的顺序是相反的，我没看出来啊
    '''
    # 获取该网络层的输出，保存到self.activation中，

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        # detach对输出结果进行分离 TODO：https://blog.csdn.net/qq_27825451/article/details/95498211 这里查看细节与不同
        self.activations.append(activation.cpu().detach())

    # 在反向传播过程中，保存反向传播的梯度信息
    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        '''
        model:模型
        target_layers:输出目标的网络层
        reshape_transform:转换
        '''
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    # 对类进行声明，使其在不使用实例的情况下调用该函数
        # 2，3维度为高度和宽度，0，1维度为batch和channel，得到每一个通道的权重
    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    # 功能和名字一样，获取loss值，TODO：看看实例怎么调用的，特别是output的格式
    # output:[batch_size,class_number]
    # 即将每张图片预测分数累加
    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    # 获取CAM图
    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    # 获取宽高
    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    # 计算每层的CAM
    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            # works like mute the min-max scale in the function of scale_cam_image
            cam[cam < 0] = 0  # 将小于0的数置零，相当于ReLU函数
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    # 将所有网络层的CAM进行融合
    def aggregate_multi_layers(self, cam_per_target_layer):
        if cam_per_target_layer:
            cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
            cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
            result = np.mean(cam_per_target_layer, axis=1)
            return self.scale_cam_image(result)
        else:
            return np.ones((1,224,224), dtype = np.float32)

    # CAM后处理部分，将值缩放到0-1之间，再调整到原图大小
    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()
        if len(input_tensor.shape)==3:
            input_tensor=torch.unsqueeze(input_tensor,0)
        # 正向传播得到网络输出logits(未经过softmax)
        _,_,output = self.activations_and_grads(input_tensor)
        if target_category.size==1 and not isinstance(target_category, int) :
            target_category = int(target_category)
        if isinstance(target_category, int):
            # batch_size，一次性求多个图片
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            # 不给出目标，网络自动赋值为预测分数最大的类别索引
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        # print(output.shape) # torch.Size([1, 1000])
        loss = self.get_loss(output, target_category)
        loss.requires_grad_(True)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    # 将0-1的CAM图转化为0-255的BGR图
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # 0-1
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img
