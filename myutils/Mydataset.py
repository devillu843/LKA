from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, image_path: list, label: list, transform=None) -> None:
        super().__init__()
        self.image_path = image_path
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        img = Image.open(self.image_path[item])
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.image_path[item]))
        label = self.label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py

        imgs, labels = tuple(zip(*batch))

        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels)

        return imgs, labels


class MyDataset2(Dataset):
    def __init__(self, image_path1: list, image_path2, label: list, transform=None) -> None:
        super().__init__()
        self.image_path1 = image_path1
        self.image_path2 = image_path2
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_path1)

    def __getitem__(self, item):
        img1 = Image.open(self.image_path1[item])
        img2 = Image.open(self.image_path2[item])
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.image_path[item]))
        label = self.label[item]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py

        img1, img2, labels = tuple(zip(*batch))

        img1 = torch.stack(img1, dim=0)
        img2 = torch.stack(img2, dim=0)
        labels = torch.as_tensor(labels)

        return img1, img2, labels

class MyDataset_QT(Dataset):
    def __init__(self, image_path: list,  transform=None) -> None:
        super().__init__()
        self.image_path1 = image_path
        
        self.transform = transform

    def __len__(self):
        return len(self.image_path1)

    def __getitem__(self, item):
        img1 = Image.open(self.image_path1[item])
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.image_path[item]))

        if self.transform is not None:
            img1 = self.transform(img1)

        return img1

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py

        img1 = tuple(zip(*batch))

        img1 = torch.stack(img1, dim=0)
       
        return img1

class MyDataset3(Dataset):
    def __init__(self, image_path1: list, image_path2:list, image_path3:list, label: list, transform=None) -> None:
        super().__init__()
        self.image_path1 = image_path1
        self.image_path2 = image_path2
        self.image_path3 = image_path3
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_path1)

    def __getitem__(self, item):
        img1 = Image.open(self.image_path1[item])
        img2 = Image.open(self.image_path2[item])
        img3 = Image.open(self.image_path3[item])
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.image_path[item]))
        label = self.label[item]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py

        img1, img2, img3, labels = tuple(zip(*batch))

        img1 = torch.stack(img1, dim=0)
        img2 = torch.stack(img2, dim=0)
        img3 = torch.stack(img3, dim=0)
        labels = torch.as_tensor(labels)

        return img1, img2, img3, labels