"""
跨域特征可视化
用t-SNE展示源域(Synthetic)和目标域(Real)的特征分布差异
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import scipy.io as sio
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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
VIS_SAVE_DIR = os.path.join(BASE_DIR, "results/visualization/cross_domain")
os.makedirs(VIS_SAVE_DIR, exist_ok=True)

CLASS_NAMES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
               'motercycle', 'person', 'ship', 'truck']


def load_source_and_target():
    """加载源域和目标域数据"""
    # 源域
    source_mat = os.path.join(cfg.VISDA_DATA_ROOT, "source_train_32x32.mat")
    source_data = sio.loadmat(source_mat)
    source_images = np.transpose(source_data['X'], (3, 0, 1, 2))
    source_labels = source_data['y'].squeeze() - 1

    # 目标域
    target_mat = os.path.join(cfg.RAW_VISDA_ROOT, "test_32x32.mat")
    target_data = sio.loadmat(target_mat)
    target_images = np.transpose(target_data['X'], (3, 0, 1, 2))
    target_labels = target_data['y'].squeeze() - 1

    # 目标域标签类型
    target_csv = os.path.join(cfg.VISDA_DATA_ROOT, "target_test_labels.csv")
    label_df = pd.read_csv(target_csv)
    class_type = label_df['class_type'].values

    print(f"源域(Synthetic): {source_images.shape}")
    print(f"目标域(Real): {target_images.shape}")
    print(f"目标域已知类: {(class_type=='share').sum()}, 未知类: {(class_type=='unknown').sum()}")

    return source_images, source_labels, target_images, target_labels, class_type


def extract_features(model, images, val_transform, desc=""):
    """提取特征"""
    model.eval()
    all_features = []

    # 每域最多3000样本
    max_samples = 3000
    images = images[:max_samples]

    batch_size = 64
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
            features, logits, _ = model(batch_tensor)
            all_features.append(features.cpu().numpy())

            if i % 500 == 0:
                print(f"  {desc} 进度: {i}/{len(images)}")

    return np.concatenate(all_features, axis=0)


def plot_tsne(source_features, target_features, source_labels, target_labels,
              source_domain_name="Source (Synthetic)", target_domain_name="Target (Real)"):
    """绘制t-SNE跨域可视化"""

    # 合并特征
    all_features = np.concatenate([source_features, target_features], axis=0)

    # t-SNE降维
    print("  执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(StandardScaler().fit_transform(all_features))

    # 分割回源域和目标域
    source_2d = features_2d[:len(source_features)]
    target_2d = features_2d[len(source_features):]

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 图1：按域着色
    ax = axes[0]
    ax.scatter(source_2d[:, 0], source_2d[:, 1], c='#E57373', alpha=0.5, s=15,
               label=f'{source_domain_name} (n={len(source_2d)})')
    ax.scatter(target_2d[:, 0], target_2d[:, 1], c='#64B5F6', alpha=0.5, s=15,
               label=f'{target_domain_name} (n={len(target_2d)})')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Domain Distribution (Source vs Target)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 图2：按类别着色（目标域）
    ax = axes[1]

    # 定义颜色映射
    colors = plt.cm.tab10.colors

    # 绘制源域（灰色背景）
    ax.scatter(source_2d[:, 0], source_2d[:, 1], c='lightgray', alpha=0.3, s=10, label='Source')

    # 绘制目标域已知类（彩色）
    target_known_mask = target_labels == 0  # 这里只显示class 0作为示例
    for class_id in range(8):
        class_mask = (target_labels == class_id)
        if class_mask.sum() > 0:
            ax.scatter(target_2d[class_mask, 0], target_2d[class_mask, 1],
                      c=[colors[class_id]], alpha=0.6, s=20,
                      label=f'{CLASS_NAMES[class_id]}')

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Target Domain Class Distribution', fontsize=14)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(VIS_SAVE_DIR, "cross_domain_tsne.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  t-SNE图已保存: {save_path}")

    return save_path


def plot_per_class_domain_shift(source_features, target_features, source_labels, target_labels):
    """绘制每个类别的域偏移情况"""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Per-Class Domain Shift (Source vs Target)\n'
                 'Each row shows one class in Source (red) and Target (blue)', fontsize=14)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)

    # 对每个已知类做t-SNE
    for class_id in range(8):
        ax = axes[class_id // 4, class_id % 4]

        # 获取该类的源域和目标域样本
        source_mask = source_labels == class_id
        target_mask = target_labels == class_id

        if source_mask.sum() == 0 or target_mask.sum() == 0:
            ax.axis('off')
            continue

        # 取样（最多500个）
        source_class_feat = source_features[source_mask][:500]
        target_class_feat = target_features[target_mask][:500]

        # 合并做t-SNE
        all_feat = np.concatenate([source_class_feat, target_class_feat])
        try:
            feat_2d = tsne.fit_transform(StandardScaler().fit_transform(all_feat))
        except:
            ax.axis('off')
            continue

        source_2d = feat_2d[:len(source_class_feat)]
        target_2d = feat_2d[len(source_class_feat):]

        ax.scatter(source_2d[:, 0], source_2d[:, 1], c='#E57373', alpha=0.6, s=15,
                  label=f'Src: {source_mask.sum()}')
        ax.scatter(target_2d[:, 0], target_2d[:, 1], c='#64B5F6', alpha=0.6, s=15,
                  label=f'Tgt: {target_mask.sum()}')
        ax.set_title(f'{CLASS_NAMES[class_id]}', fontsize=10)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(VIS_SAVE_DIR, "per_class_domain_shift.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  各类别域偏移图已保存: {save_path}")


def main():
    print("=" * 60)
    print("跨域特征可视化")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/4] 加载模型...")
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
    print("模型加载成功")

    # 2. 加载数据
    print("\n[2/4] 加载源域和目标域数据...")
    source_images, source_labels, target_images, target_labels, class_type = load_source_and_target()
    val_transform = get_val_transform("VisDA")

    # 3. 提取特征
    print("\n[3/4] 提取特征...")
    print("  提取源域特征...")
    source_features = extract_features(disent_model, source_images, val_transform, "源域")
    source_labels = source_labels[:len(source_features)]

    print("  提取目标域特征...")
    target_features = extract_features(disent_model, target_images, val_transform, "目标域")
    target_labels = target_labels[:len(target_features)]
    class_type = class_type[:len(target_features)]

    # 4. 可视化
    print("\n[4/4] 生成可视化...")
    plot_tsne(source_features, target_features, source_labels, target_labels)
    plot_per_class_domain_shift(source_features, target_features, source_labels, target_labels)

    print("\n" + "=" * 60)
    print(f"可视化完成! 结果保存在: {VIS_SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
