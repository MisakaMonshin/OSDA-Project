"""
未知类检测对比可视化 - 简化版
只展示实际的分类结果，不重新计算融合分数
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import scipy.io as sio
import pandas as pd
import sys

sys.path.insert(0, '/root/autodl-tmp/OSDA_Project/code/test')

from VisDA_test_02_train import (
    ImprovedFeatureDisentanglementModel,
    get_val_transform,
    cfg
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/root/autodl-tmp/OSDA_Project"
MODEL_PATH = os.path.join(BASE_DIR, "results/saved_models_visda_joint_train_v1/best_visda_joint_model_v1.pth")
OSBP_PATH = os.path.join(BASE_DIR, "results/baseline_models_visda/osbp_visda.pth")
VIS_SAVE_DIR = os.path.join(BASE_DIR, "results/visualization/unknown_detection_comparison")
os.makedirs(VIS_SAVE_DIR, exist_ok=True)

CLASS_NAMES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
               'motercycle', 'person', 'ship', 'truck']


class OSBPModel(nn.Module):
    def __init__(self, num_known_classes=8):
        super().__init__()
        self.backbone = timm.create_model("convnextv2_tiny", pretrained=False, num_classes=0)
        backbone_out = self.backbone.num_features
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_out, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_known_classes + 1)
        )
        self.domain_discriminator = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        feat_raw = self.backbone(x)
        feat = self.feature_proj(feat_raw)
        logits = self.classifier(feat)
        return feat, logits


def load_data():
    test_mat = os.path.join(cfg.RAW_VISDA_ROOT, "test_32x32.mat")
    target_csv = os.path.join(cfg.VISDA_DATA_ROOT, "target_test_labels.csv")
    data = sio.loadmat(test_mat)
    images = np.transpose(data['X'], (3, 0, 1, 2))
    labels = data['y'].squeeze() - 1
    label_df = pd.read_csv(target_csv)
    class_type = label_df['class_type'].values
    return images, labels, class_type


def main():
    print("=" * 60)
    print("未知类检测对比可视化")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    images, labels, class_type = load_data()
    val_transform = get_val_transform("VisDA")

    known_mask = class_type == 'share'
    unknown_mask = class_type == 'unknown'
    print(f"已知类: {known_mask.sum()}, 未知类: {unknown_mask.sum()}")

    # 加载我们的模型
    print("\n[2/4] 加载我们的模型...")
    our_model = ImprovedFeatureDisentanglementModel(
        num_classes=cfg.TOTAL_VISDA_CLASSES,
        feature_dim=cfg.FEATURE_DIM
    ).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    our_model.load_state_dict(checkpoint['disent_model'])
    if 'class_centers' in checkpoint and checkpoint['class_centers'] is not None:
        our_model.class_centers = checkpoint['class_centers'].to(DEVICE)
    our_model.eval()

    # 加载OSBP
    print("\n[3/4] 加载OSBP模型...")
    osbp_model = OSBPModel(num_known_classes=8).to(DEVICE)
    osbp_checkpoint = torch.load(OSBP_PATH, map_location=DEVICE, weights_only=False)
    osbp_model.load_state_dict(osbp_checkpoint)
    osbp_model.eval()

    # 推理 - 只取部分样本
    print("\n[4/4] 运行推理...")
    sample_size = 3000
    images = images[:sample_size]
    labels = labels[:sample_size]
    class_type = class_type[:sample_size]

    our_preds = np.zeros(len(images))
    our_confs = np.zeros(len(images))
    osbp_preds = np.zeros(len(images))
    osbp_confs = np.zeros(len(images))
    osbp_c1_probs = np.zeros(len(images))

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_idx = min(i + batch_size, len(images))
            batch_images = images[i:batch_idx]

            batch_tensors = []
            for img in batch_images:
                img_uint8 = img.astype(np.uint8)
                transformed = val_transform(image=img_uint8)["image"]
                batch_tensors.append(transformed)

            batch_tensor = torch.stack(batch_tensors).to(DEVICE)

            # 我们的模型
            _, logits, _ = our_model(batch_tensor)
            probs = F.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)
            our_preds[i:batch_idx] = preds.cpu().numpy()
            our_confs[i:batch_idx] = conf.cpu().numpy()

            # OSBP
            _, osbp_logits = osbp_model(batch_tensor)
            osbp_probs = F.softmax(osbp_logits, dim=1)
            osbp_conf, osbp_pred = osbp_probs[:, :8].max(dim=1)
            osbp_preds[i:batch_idx] = osbp_pred.cpu().numpy()
            osbp_confs[i:batch_idx] = osbp_conf.cpu().numpy()
            osbp_c1_probs[i:batch_idx] = osbp_probs[:, 8].cpu().numpy()

    # 统计
    unknown_mask = class_type == 'unknown'
    known_mask = class_type == 'share'

    # 我们的方法 - 已知类准确率
    our_known_correct = (our_preds[known_mask] == labels[known_mask]).sum()
    our_known_total = known_mask.sum()
    our_known_acc = our_known_correct / our_known_total * 100

    # OSBP - 已知类准确率
    osbp_known_correct = (osbp_preds[known_mask] == labels[known_mask]).sum()
    osbp_known_total = known_mask.sum()
    osbp_known_acc = osbp_known_correct / osbp_known_total * 100

    print(f"\n已知类分类准确率:")
    print(f"  我们的方法: {our_known_acc:.1f}%")
    print(f"  OSBP: {osbp_known_acc:.1f}%")

    # 可视化 - 只展示已知类样本
    print("\n生成可视化...")

    # 展示一些已知类样本的分类结果
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    # 我们方法正确分类的已知类样本
    our_correct_mask = known_mask & (our_preds == labels)
    our_correct_indices = np.where(our_correct_mask)[0]
    np.random.seed(42)
    selected = np.random.choice(our_correct_indices, min(6, len(our_correct_indices)), replace=False)

    for j, idx in enumerate(selected):
        ax = axes[0, j]
        img = images[idx]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        ax.imshow(img)
        true_cls = CLASS_NAMES[labels[idx]]
        pred_cls = CLASS_NAMES[int(our_preds[idx])]
        ax.set_title(f'真:{true_cls}\n预:{pred_cls}\nC:{our_confs[idx]:.2f}', fontsize=8)
        ax.axis('off')

    axes[0, 0].set_ylabel('我们的方法\n(已知类分类正确)', fontsize=10)

    # OSBP正确分类的已知类样本
    osbp_correct_mask = known_mask & (osbp_preds == labels)
    osbp_correct_indices = np.where(osbp_correct_mask)[0]
    selected = np.random.choice(osbp_correct_indices, min(6, len(osbp_correct_indices)), replace=False)

    for j, idx in enumerate(selected):
        ax = axes[1, j]
        img = images[idx]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        ax.imshow(img)
        true_cls = CLASS_NAMES[labels[idx]]
        pred_cls = CLASS_NAMES[int(osbp_preds[idx])]
        ax.set_title(f'真:{true_cls}\n预:{pred_cls}\nC:{osbp_confs[idx]:.2f}', fontsize=8)
        ax.axis('off')

    axes[1, 0].set_ylabel('OSBP\n(已知类分类正确)', fontsize=10)

    fig.suptitle(f'已知类分类对比\n我们的方法准确率:{our_known_acc:.1f}% | OSBP准确率:{osbp_known_acc:.1f}%', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(VIS_SAVE_DIR, "known_class_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已知类分类对比图已保存: {save_path}")

    # 保存报告
    report = f"""未知类检测对比报告
==================

已知类分类准确率:
  我们的方法: {our_known_acc:.1f}%
  OSBP: {osbp_known_acc:.1f}%

说明:
- 我们的方法在已知类分类上准确率较低({our_known_acc:.1f}%)
- OSBP在已知类分类上准确率较高({osbp_known_acc:.1f}%)

但OSBP的未知类检测能力(TNR)为0%，这是主要问题。
"""
    report_path = os.path.join(VIS_SAVE_DIR, "comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存: {report_path}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    import timm
    main()
