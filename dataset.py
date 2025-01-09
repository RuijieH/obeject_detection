import torch
from torchvision.datasets import Cityscapes
import torchvision.transforms as T
from label_mapping import name2label
from utils import extract_boxes_from_polygon
class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.transforms =T.Compose([
            T.ToTensor(),
        ])
        self.dataset = Cityscapes(root, split=split, mode="fine",
                              target_type='polygon'
                              )

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        # 图像处理
        if self.transforms is not None:
            img = self.transforms(img)


        #制作标签
        # 提取目标的边界框和标签
        filterd_label = list(name2label.keys())
        boxes = extract_boxes_from_polygon(target,filterd_label=filterd_label)
        labels = [name2label[obj['label']].trainId for obj in target['objects'] if obj['label'] in filterd_label]
        # 将边界框和标签打包为字典，符合 Faster R-CNN 的格式
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 检查是否有合法的目标标签和边界框
        if len(boxes) == 0 or len(labels) == 0:
            return self.__getitem__((idx + 1) % len(self))  # 跳过没有合法标签的图像，获取下一个图像
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)  # 假设没有 crowd 实例
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        return img, target

    def __len__(self):
        return len(self.dataset)