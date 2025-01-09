from tqdm import tqdm
from label_mapping import name2label
from dataset import CityscapesDataset
import torch
import torch.optim as optim

#模型引入
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn




def main():
    # 构造数据加载器
    dataset = CityscapesDataset('./data/cityscapes',split="train")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))


    #构建模型
    # 过滤后的标签数量（加上背景类）
    num_classes = len(name2label.keys()) + 1  # 加1是因为要包括背景类
    # 加载预训练的 Faster R-CNN 模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # 修改分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # 将模型移动到设备 (GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # 优化器和学习率调度器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 开始训练
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 前向传播和计算损失
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

    # 训练结束后保存模型
    torch.save(model.state_dict(), 'fasterrcnn_cityscapes.pth')


if __name__ == '__main__':
    main()





