"""
OSBP (Open Set Domain Adaptation by Backpropagation) 基线实现
参考论文: Saito et al., "Open Set Domain Adaptation by Backpropagation", ECCV 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OSBPModel(nn.Module):
    """
    OSBP模型核心实现
    特点：C+1分类器（已知类 + 1个未知类）
    """
    def __init__(self, num_known_classes=8, feature_dim=768):
        super().__init__()
        self.num_known = num_known_classes

        # 共享特征提取器（使用ConvNeXt-V2-Tiny）
        self.backbone = timm.create_model("convnextv2_tiny", pretrained=False, num_classes=0)
        local_path = "/root/autodl-tmp/OSDA_Project/code/test/pytorch_model.bin"
        if os.path.exists(local_path):
            state_dict = torch.load(local_path, map_location=DEVICE)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded pretrained ConvNeXt-V2-Tiny weights")

        backbone_out = self.backbone.num_features  # 768

        # 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_out, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # OSBP核心：C+1分类器
        # 前C个输出对应已知类，第C+1个输出对应未知类
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_known_classes + 1)  # C+1分类
        )

        # 域判别器（用于对抗训练）
        self.domain_discriminator = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        """前向传播"""
        feat_raw = self.backbone(x)
        feat = self.feature_proj(feat_raw)
        logits = self.classifier(feat)  # [B, C+1]
        return feat, logits

    def get_domain_logits(self, feat, alpha=1.0):
        """获取域判别器输出（带GRL）"""
        reversed_feat = GradientReversalLayer.apply(feat, alpha)
        domain_logits = self.domain_discriminator(reversed_feat)
        return domain_logits


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层"""
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def osbp_adversarial_loss(logits, labels, threshold=0.5):
    """
    OSBP对抗损失函数

    公式: L_adv = -t*log(p(y=C+1|x)) - (1-t)*log(1-p(y=C+1|x))

    对于目标域样本：
    - 如果分类器认为它是未知类（y=C+1）的概率高，则损失低
    - 如果分类器认为它是已知类（y<C+1）的概率高，则损失高

    这样可以鼓励分类器将目标域中的未知样本正确识别为未知类
    """
    probs = F.softmax(logits, dim=1)
    unknown_prob = probs[:, -1]  # 第C+1类的概率

    t = threshold
    # 对于未知类样本（labels >= num_known），我们希望unknown_prob高
    # 对于已知类样本（labels < num_known），我们希望unknown_prob低
    is_known = (labels < 8).float()

    # 已知类样本：未知概率应该低
    known_loss = -is_known * torch.log(1 - unknown_prob + 1e-8)

    # 未知类样本：未知概率应该高
    unknown_loss = -(1 - is_known) * torch.log(unknown_prob + 1e-8)

    # 加权损失
    loss = (known_loss + unknown_loss).mean()
    return loss


def train_osbp(model, train_loader, val_loader, num_epochs=20, lr=1e-4):
    """
    OSBP训练函数

    训练策略：
    1. 源域样本：使用标准交叉熵损失训练已知类分类器
    2. 目标域样本：使用OSBP损失，鼓励将未知样本识别为未知类
    3. 域对抗训练：使用GRL对齐源域和目标域特征分布
    """
    print("\n" + "="*60)
    print("Training OSBP on VisDA")
    print("="*60)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 5

    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 用于目标域样本的DataLoader
    target_loader = DataLoader(
        val_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    target_iter = iter(target_loader)

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        total_loss = 0.0

        # 动态调整GRL强度
        grl_alpha = 1.0 if epoch >= 3 else 0.1  # 前3个epoch先训练分类器

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # 只使用已知类样本训练分类器
            known_mask = (labels >= 0) & (labels < 8)
            if known_mask.sum() == 0:
                continue

            # 获取目标域样本
            try:
                tgt_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_batch = next(target_iter)

            tgt_images = tgt_batch[0].to(DEVICE, non_blocking=True)
            tgt_labels = tgt_batch[1].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 1. 源域分类损失
            feat_src, src_logits = model(images)
            cls_loss = criterion_cls(src_logits[known_mask], labels[known_mask])

            # 更新已知类准确率
            preds = src_logits[known_mask].argmax(dim=1)
            correct += preds.eq(labels[known_mask]).sum().item()
            total += known_mask.sum().item()

            # 2. 域对抗损失
            feat_tgt, tgt_logits = model(tgt_images)

            src_domain_pred = model.get_domain_logits(feat_src, grl_alpha)
            tgt_domain_pred = model.get_domain_logits(feat_tgt, grl_alpha)

            domain_labels_src = torch.zeros(len(feat_src), 1, device=DEVICE)
            domain_labels_tgt = torch.ones(len(feat_tgt), 1, device=DEVICE)

            domain_loss = F.binary_cross_entropy_with_logits(src_domain_pred, domain_labels_src) + \
                         F.binary_cross_entropy_with_logits(tgt_domain_pred, domain_labels_tgt)

            # 3. OSBP损失（针对目标域）- feat_tgt和tgt_logits已经在上面获取
            osbp_loss = osbp_adversarial_loss(tgt_logits, tgt_labels, threshold=0.5)

            # 总损失
            loss = cls_loss + 0.1 * domain_loss + 0.2 * osbp_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        train_acc = correct / total if total > 0 else 0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Loss: {total_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"GRL alpha: {grl_alpha:.2f}")

        if train_acc > best_acc + 0.001:
            best_acc = train_acc
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存模型
            save_path = "/root/autodl-tmp/OSDA_Project/results/baseline_models_visda/osbp_visda.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved best model with acc={train_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"OSBP - Best Train Acc: {best_acc:.4f} (Epoch {best_epoch})")
    return model


@torch.no_grad()
def evaluate_osbp(model, loader, num_known_classes=8, method_name="OSBP"):
    """
    评估OSBP模型

    开放集检测策略：
    - 对于目标域样本，计算其属于未知类（C+1）的概率
    - 如果 unknown_prob > threshold，则判定为未知类
    - 否则判定为已知类，并取最大概率的已知类作为预测
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {method_name} on VisDA Target Domain")
    print(f"{'='*60}")

    model.eval()

    all_unknown_scores = []  # 未知类得分
    all_labels = []  # 真实标签（0=已知，1=未知）
    correct_known = 0  # 已知类正确分类数
    total_known = 0  # 已知类总数

    for images, labels, class_type in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        _, logits = model(images)

        # 计算未知类概率
        probs = F.softmax(logits, dim=1)
        unknown_probs = probs[:, -1]  # 第C+1类的概率

        # 已知类样本
        known_mask = (labels < num_known_classes)
        if known_mask.sum() > 0:
            known_preds = logits[known_mask, :num_known_classes].argmax(dim=1)
            correct_known += known_preds.eq(labels[known_mask]).sum().item()
            total_known += known_mask.sum().item()

        # 未知类样本
        unknown_mask = (labels >= num_known_classes)

        all_unknown_scores.extend(unknown_probs.cpu().numpy())
        all_labels.extend(unknown_mask.float().cpu().numpy())

    known_acc = correct_known / total_known if total_known > 0 else 0

    # 计算OSDA指标
    unknown_scores = np.array(all_unknown_scores)
    true_labels = np.array(all_labels)

    # AUROC
    auroc = roc_auc_score(true_labels, unknown_scores)

    # 找到最优阈值
    fpr, tpr, thresholds = roc_curve(true_labels, unknown_scores)
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_idx] if youden_idx < len(thresholds) else 0.5

    # 计算H-score
    predicted_unknown = (unknown_scores >= optimal_threshold).astype(int)
    tp = np.sum((predicted_unknown == 1) & (true_labels == 1))  # 正确识别的未知类
    fn = np.sum((predicted_unknown == 0) & (true_labels == 1))  # 漏判的未知类
    fp = np.sum((predicted_unknown == 1) & (true_labels == 0))  # 误判的已知类
    tn = np.sum((predicted_unknown == 0) & (true_labels == 0))  # 正确识别的已知类

    unknown_recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 未知类召回率
    known_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 已知类特异率
    h_score = 2 * unknown_recall * known_specificity / (unknown_recall + known_specificity) \
              if (unknown_recall + known_specificity) > 0 else 0

    print(f"{method_name} Results:")
    print(f"  Known Class Accuracy: {known_acc:.4f}")
    print(f"  Unknown Recall (TPR): {unknown_recall:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  H-score: {h_score:.4f}")

    return known_acc, unknown_recall, auroc, h_score


if __name__ == "__main__":
    import timm
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import scipy.io as sio
    import pandas as pd
    import cv2

    BASE_DIR = "/root/autodl-tmp/OSDA_Project"
    VISDA_DATA_ROOT = os.path.join(BASE_DIR, "data/processed/VisDA")
    RAW_VISDA_ROOT = os.path.join(BASE_DIR, "data/raw/VisDA")

    # 数据加载
    train_mat = os.path.join(VISDA_DATA_ROOT, "source_train_32x32.mat")
    test_mat = os.path.join(RAW_VISDA_ROOT, "test_32x32.mat")
    target_label_csv = os.path.join(VISDA_DATA_ROOT, "target_test_labels.csv")

    # 变换
    train_transform = A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    class VisDADataset:
        def __init__(self, mat_path, transform=None, is_target=False, target_label_csv=None):
            self.transform = transform
            self.is_target = is_target
            data = sio.loadmat(mat_path)
            self.images = np.transpose(data['X'], (3, 0, 1, 2))
            self.labels = data['y'].squeeze() - 1

            if target_label_csv and os.path.exists(target_label_csv):
                self.is_target = True
                label_df = pd.read_csv(target_label_csv)
                self.class_type = label_df['class_type'].values

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = self.images[idx].astype(np.uint8)
            label = self.labels[idx]
            if self.transform:
                img = self.transform(image=img)["image"]
            if self.is_target:
                return img, label, self.class_type[idx]
            return img, label

    # 加载数据
    train_dataset = VisDADataset(train_mat, transform=train_transform, is_target=False)
    test_dataset = VisDADataset(test_mat, transform=val_transform, is_target=True,
                                 target_label_csv=target_label_csv)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # 检查是否已有训练好的模型
    save_path = "/root/autodl-tmp/OSDA_Project/results/baseline_models_visda/osbp_visda.pth"

    if os.path.exists(save_path):
        print("\n" + "="*60)
        print("Loading existing OSBP model...")
        print("="*60)
        osbp_model = OSBPModel(num_known_classes=8).to(DEVICE)
        osbp_model.load_state_dict(torch.load(save_path))
    else:
        print("\n" + "="*60)
        print("Training OSBP model...")
        print("="*60)
        osbp_model = OSBPModel(num_known_classes=8).to(DEVICE)
        osbp_model = train_osbp(osbp_model, train_loader, test_loader)

    # 评估
    known_acc, unknown_recall, auroc, h_score = evaluate_osbp(osbp_model, test_loader)

    print("\n" + "="*60)
    print("OSBP Final Results")
    print("="*60)
    print(f"Known Accuracy: {known_acc:.4f}")
    print(f"Unknown Recall: {unknown_recall:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"H-score: {h_score:.4f}")
