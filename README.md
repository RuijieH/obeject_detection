# 基于二阶段的目标检测
+ 参考论文：Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
+ 参考代码：https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
## 准备环境
```bash
conda create -n object_det python=3.8.19 --no-default-packages
conda activate object_det
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tqdm
```
## 准备数据
+ 从官网下载精细注释和原始图片2个压缩包
  + https://www.cityscapes-dataset.com/downloads/
  + 注释：gtFine_trainvaltest.zip 
  + 原始图片：leftImg8bit_trainvaltest.zip
+ 解压
    ```bash
    mv gtFine_trainvaltest.zip  leftImg8bit_trainvaltest.zip ./data/cityscapes
    unzip gtFine_trainvaltest.zip
    unzip leftImg8bit_trainvaltest.zip
    #目录结构
    #data
    #└── cityscapes
    #    ├── gtFine
    #    │   ├── test
    #    │   ├── train
    #    │   └── val
    #    ├── leftImg8bit
    #    │   ├── test
    #    │   ├── train
    #    │   └── val
    #    ├── license.txt
    #    └── README
    ```
# 训练
```bash
#目前只训练检测10个类别，如果需要检测更多类别，在label_mapping.py变量label2label中自行添加
#需要注意：由于citescape只给了实例分割注释，我们在dataset.py中额外进行了bounding box的计算
python train.py
```
# 推理
在infer.ipynb使用提供的权重对验证集进行推理和可视化(可以使用我提供的权重快速看下效果：[fasterrcnn_cityscapes.pth](https://pan.baidu.com/s/1eVwlUb15vNoWqbAASYQNwg?pwd=1234)

# 目标检测的相关评估指标
+ AP
+ mAP
+ Precision
+ Recall
+ F1 Score
+ FPS
