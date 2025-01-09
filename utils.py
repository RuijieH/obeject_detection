import numpy as np
import cv2
import matplotlib.pyplot as plt
def extract_boxes_from_polygon(target,filterd_label):
    """
    因为citescapes数据集没有提供boundingbox 注释，需要从实例分割中提取boundingbox
    从多边形目标数据中提取边界框。
    每个对象包含一个或多个多边形，使用多边形坐标来计算最小外接矩形。
    """
    boxes = []

    # 遍历目标中的每个对象
    for obj in target['objects']:
        polygon = obj['polygon']  # 获取多边形坐标
        label = obj["label"]
        if label not in filterd_label:
            continue

        # 将多边形转换为 NumPy 数组
        poly_np = np.array(polygon)

        # 计算多边形的最小外接矩形
        xmin, ymin = poly_np.min(axis=0)
        xmax, ymax = poly_np.max(axis=0)

        # 将边界框存储为 [x_min, y_min, x_max, y_max]
        boxes.append([xmin, ymin, xmax, ymax])

    return boxes
def visualize_predictions(image, predictions, gt_boxes, vis_only_pred=False):
    # 将图像转换为 NumPy 格式，并确保形状正确
    image_pred = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    image_pred = (image_pred * 255).astype(np.uint8).copy()

    image_gt = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    image_gt = (image_gt * 255).astype(np.uint8).copy()


    # 生成随机颜色以区分不同的真实框
    colors = [(0, 255, 0) for _ in gt_boxes]  # 默认绿色表示真实框

    # 绘制预测框 (predicted boxes) 到 image_pred
    for pred_box, label in zip(predictions['boxes'].cpu().numpy(), predictions['labels']):
        x1, y1, x2, y2 = pred_box
        cv2.rectangle(image_pred, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 绿色表示预测框
        cv2.putText(image_pred, "Pred", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 绘制真实框 (ground truth boxes) 到 image_gt
    for i, gt_box in enumerate(gt_boxes):
        color = colors[i]  # 使用预定义的颜色
        x1, y1, x2, y2 = gt_box.int().tolist()
        cv2.rectangle(image_gt, (x1, y1), (x2, y2), color, 2)  # 绿色表示真实框
        cv2.putText(image_gt, "GT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 显示左右两张图：左边显示预测框，右边显示真实框
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 左侧显示预测框
    axes[0].imshow(cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Predicted Boxes')
    axes[0].axis('off')  # 隐藏坐标轴

    # 右侧显示真实框
    axes[1].imshow(cv2.cvtColor(image_gt, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Ground Truth Boxes')
    axes[1].axis('off')  # 隐藏坐标轴

    # 显示结果
    plt.show()
