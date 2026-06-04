"""
VisDA aeroplane类分类可视化 - 修复版
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import sys

sys.path.insert(0, '/root/autodl-tmp/OSDA_Project/code/test')

from VisDA_test_02_train import (
    ImprovedFeatureDisentanglementModel,
    FeatureVAE,
    get_val_transform,
    cfg
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/root/autodl-tmp/OSDA_Project"
MODEL_PATH = os.path.join(BASE_DIR, "results/saved_models_visda_joint_train_v1/best_visda_joint_model_v1.pth")
VIS_SAVE_DIR = os.path.join(BASE_DIR, "results/visualization/visda_classification")
os.makedirs(VIS_SAVE_DIR, exist_ok=True)

CLASS_NAMES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
               'motercycle', 'person', 'ship', 'truck']


def load_visda_images():
    """直接加载VisDA图像（不经过transform）用于可视化"""
    test_mat = os.path.join(cfg.RAW_VISDA_ROOT, "test_32x32.mat")
    target_label_csv = os.path.join(cfg.VISDA_DATA_ROOT, "target_test_labels.csv")

    data = sio.loadmat(test_mat)
    images = data.get('X', None)  # (32, 32, 3, N)
    labels = data.get('y', None).squeeze() - 1  # (N,)

    # 转换维度：(32,32,3,N) -> (N,32,32,3)
    images = np.transpose(images, (3, 0, 1, 2))
    labels = labels.squeeze()

    # 标签文件
    label_df = pd.read_csv(target_label_csv)
    class_type = label_df['class_type'].values

    print(f"VisDA图像形状: {images.shape}, 标签范围: {labels.min()}-{labels.max()}")
    print(f"数据类型: {images.dtype}, 值范围: {images.min()}-{images.max()}")

    return images, labels, class_type


def visualize_aeroplane_fixed(images, labels, predictions, confidences, num_samples=12):
    """展示aeroplane类（class_id=0）分类结果"""

    # 筛选aeroplane类(class_id=0)的样本
    aeroplane_mask = labels == 0
    aeroplane_indices = np.where(aeroplane_mask)[0]

    # 正确分类的样本
    correct_mask = aeroplane_mask & (predictions == labels)
    correct_indices = aeroplane_indices[predictions[aeroplane_mask] == 0]

    print(f"\nAeroplane类样本统计:")
    print(f"  总样本数: {len(aeroplane_indices)}")
    print(f"  正确分类: {len(correct_indices)}")
    if len(aeroplane_indices) > 0:
        print(f"  准确率: {len(correct_indices)/len(aeroplane_indices)*100:.1f}%")

    if len(correct_indices) == 0:
        print("没有正确分类的样本！")
        return

    # 按置信度排序，高的在前面
    correct_conf = confidences[correct_indices]
    sorted_order = np.argsort(-correct_conf)
    top_indices = correct_indices[sorted_order[:num_samples]]

    # 创建网格图
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle('Aeroplane Classification Results (Correct Only)\n'
                 'Sorted by Confidence (High → Low)', fontsize=14)

    for idx, ax in enumerate(axes.flat):
        if idx < len(top_indices):
            sample_idx = top_indices[idx]

            # 直接获取原始图像（uint8格式，0-255）
            img = images[sample_idx]  # (32, 32, 3), uint8

            # 确保值在0-255范围内
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # 显示图像
            ax.imshow(img)

            conf = confidences[sample_idx]
            pred = predictions[sample_idx]
            ax.set_title(f'Conf: {conf:.3f}\nPred: {CLASS_NAMES[pred]}', fontsize=9, color='darkgreen')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(VIS_SAVE_DIR, "aeroplane_classification_fixed.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nAeroplane分类结果已保存: {save_path}")


def visualize_all_classes_sample(images, labels, predictions, confidences, num_per_class=3):
    """展示所有类别的样本"""

    fig, axes = plt.subplots(10, num_per_class, figsize=(10, 18))
    fig.suptitle('Sample Classifications per Class\n(Top 3 by Confidence, Correct Only)', fontsize=14)

    for class_id in range(10):
        class_mask = labels == class_id
        class_correct_mask = class_mask & (predictions == labels)

        if class_correct_mask.sum() > 0:
            class_indices = np.where(class_correct_mask)[0]
            class_conf = confidences[class_correct_mask]
            sorted_order = np.argsort(-class_conf)
            top_indices = class_indices[sorted_order[:num_per_class]]

            for j in range(num_per_class):
                ax = axes[class_id, j]
                if j < len(top_indices):
                    sample_idx = top_indices[j]
                    img = images[sample_idx]

                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                    ax.imshow(img)
                    conf = confidences[sample_idx]
                    ax.set_title(f'{conf:.2f}', fontsize=7, color='darkgreen')
                    ax.axis('off')
                else:
                    ax.axis('off')
        else:
            for j in range(num_per_class):
                axes[class_id, j].axis('off')

        if class_id < len(CLASS_NAMES):
            axes[class_id, 0].set_ylabel(CLASS_NAMES[class_id][:8], fontsize=7, rotation=0, ha='right', va='center')

    plt.tight_layout(rect=[0.1, 0, 1, 0.98])
    save_path = os.path.join(VIS_SAVE_DIR, "all_classes_samples_fixed.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"所有类别样本图已保存: {save_path}")


def main():
    print("=" * 60)
    print("VisDA Aeroplane 分类可视化 (修复版)")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/3] 加载模型...")
    disent_model = ImprovedFeatureDisentanglementModel(
        num_classes=cfg.TOTAL_VISDA_CLASSES,
        feature_dim=cfg.FEATURE_DIM
    ).to(DEVICE)

    vae_model = FeatureVAE(
        feature_dim=cfg.FEATURE_DIM,
        latent_dim=cfg.LATENT_DIM
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    disent_model.load_state_dict(checkpoint['disent_model'])
    vae_model.load_state_dict(checkpoint['vae_model'])

    if 'class_centers' in checkpoint and checkpoint['class_centers'] is not None:
        disent_model.class_centers = checkpoint['class_centers'].to(DEVICE)

    disent_model.eval()
    vae_model.eval()
    print("模型加载成功")

    # 2. 加载数据
    print("\n[2/3] 加载图像...")
    images, labels, class_type = load_visda_images()
    val_transform = get_val_transform("VisDA")

    # 3. 推理
    print("\n[3/3] 运行推理...")
    predictions = []
    confidences = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_idx = min(i + batch_size, len(images))
            batch_images = images[i:batch_idx]

            batch_tensors = []
            for img in batch_images:
                # 转换为uint8再transform
                img_uint8 = img.astype(np.uint8)
                transformed = val_transform(image=img_uint8)["image"]
                batch_tensors.append(transformed)

            batch_tensor = torch.stack(batch_tensors).to(DEVICE)
            features, logits, _ = disent_model(batch_tensor)
            probs = F.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)

            predictions.extend(preds.cpu().numpy())
            confidences.extend(conf.cpu().numpy())

            if i % 5000 == 0:
                print(f"  进度: {i}/{len(images)}")

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    # 统计
    known_mask = class_type == 'share'
    known_acc = (predictions[known_mask] == labels[known_mask]).mean() * 100
    print(f"\n已知类准确率: {known_acc:.1f}%")

    # 可视化
    print("\n生成可视化...")
    visualize_aeroplane_fixed(images, labels, predictions, confidences, num_samples=12)
    visualize_all_classes_sample(images, labels, predictions, confidences, num_per_class=3)

    print("\n" + "=" * 60)
    print(f"完成! 结果在: {VIS_SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
