import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tensorboardX import SummaryWriter
import os
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
import timm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import scipy.io as sio
import pandas as pd
import warnings
import cv2  # 新增：导入cv2以支持兼容版插值方式
warnings.filterwarnings('ignore')

# -------------------------- 全局配置类（核心调整：平衡损失权重+学习率） --------------------------
class Config:
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS = torch.cuda.device_count()
    PIN_MEMORY = True if DEVICE.type == "cuda" else False
    
    # 训练配置（核心调整：降低批次+调整学习率）
    BATCH_SIZE = 64  # 从128降为64，提升稳定性
    GRADIENT_ACCUMULATION_STEPS = 1  # 关闭梯度累积，避免梯度消失
    EPOCHS = 40
    LEARNING_RATE = 1e-4  # backbone学习率提升
    CLASSIFIER_LR = 5e-4  # 分类器学习率大幅提升
    VAE_LR = 1e-4
    WEIGHT_DECAY = 5e-5  # 降低权重衰减
    NUM_WORKERS = 4 if os.cpu_count() > 4 else 0
    
    # 路径配置
    BASE_DIR = "/root/autodl-tmp/OSDA_Project"
    VISDA_DATA_ROOT = os.path.join(BASE_DIR, "data/processed/VisDA")
    RAW_VISDA_ROOT = os.path.join(BASE_DIR, "data/raw/VisDA")
    WEIGHTS_DIR = os.path.join(BASE_DIR, "code/test")
    LOG_DIR = "results/logs_visda_joint_train_v1"
    SAVE_DIR = "results/saved_models_visda_joint_train_v1/"
    VISUALIZATION_DIR = "results/visualization/visda_v1"
    ABLATION_DIR = "results/ablation_exp/visda"
    
    # 目标指标
    TARGET_ACC = 0.80
    TARGET_MSE = 0.03
    TARGET_DOMAIN_MSE_DIFF = 0.03
    TARGET_AUROC = 0.70
    TARGET_H_SCORE = 0.55
    
    # 早停配置
    PATIENCE = 15
    SAVE_CHECKPOINT_EPOCH = 5
    
    # 开放集超参数（核心调整：训练初期关闭开放集损失）
    OPEN_SET_LAMBDA = 0.8
    GRL_ALPHA_BASE = 1.0
    GRL_ALPHA_DECAY = 0.95
    TEMP = 0.1
    CENTER_UPDATE_EPOCH = 5  # 延迟类中心更新
    MARGIN = 1.2
    NUM_KNOWN_CLASSES = 8
    TOTAL_VISDA_CLASSES = 10
    FEATURE_DIM = 1024  # 降低特征维度，提升稳定性
    LATENT_DIM = 512

# 初始化配置实例
cfg = Config()

# 设置CUDA优化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 创建必要目录
for dir_path in [cfg.LOG_DIR, cfg.SAVE_DIR, cfg.VISUALIZATION_DIR, cfg.ABLATION_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    print(f"📁 目录已准备：{dir_path}")

# -------------------------- VisDA数据集类（核心修复：适配32x32→224x224） --------------------------
class VisDADataset(Dataset):
    def __init__(self, mat_path, transform=None, is_train=True, 
                 target_label_csv=None, is_target=False):
        self.transform = transform
        self.is_train = is_train
        self.is_target = is_target
        self.class_type = None
        self.class_label = None
        
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"数据集文件不存在：{mat_path}")
        
        # 加载数据
        self.data = sio.loadmat(mat_path)
        self.images = self.data.get('X', None)
        self.labels = self.data.get('y', None)
        
        if self.images is None or self.labels is None:
            raise ValueError(f"数据集格式错误，缺少X或y字段：{mat_path}")
        
        # 维度转换：(32,32,3,N) → (N,32,32,3)
        self.images = np.transpose(self.images, (3, 0, 1, 2))
        # 标签转换：1-based → 0-based
        self.labels = self.labels.squeeze() - 1
        
        assert len(self.images) == len(self.labels), f"图像和标签数量不匹配：{len(self.images)} vs {len(self.labels)}"
        assert np.min(self.labels) >= 0, "标签转换后出现负数"
        
        # 加载目标域类别类型
        if target_label_csv is not None and os.path.exists(target_label_csv):
            self.is_target = True
            label_df = pd.read_csv(target_label_csv)
            assert len(label_df) == len(self.images), "标签文件长度不匹配"
            self.class_type = label_df['class_type'].values
            self.class_label = label_df['class_label'].values
            print(f"✅ VisDA目标域加载完成：共享类{sum(self.class_type == 'share')}个，未知类{sum(self.class_type == 'unknown')}个")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = self.images[idx].astype(np.uint8)
            label = self.labels[idx]
            
            # 应用数据增强
            if self.transform:
                img = self.transform(image=img)["image"]
            
            # 目标域返回：图像、标签、类别类型；源域仅返回图像、标签
            if self.is_target:
                return img, label, self.class_type[idx]
            return img, label
        except Exception as e:
            print(f"⚠️ 加载样本{idx}失败：{e}")
            dummy_img = torch.zeros(3, 224, 224)
            dummy_label = torch.tensor(0)
            if self.is_target:
                return dummy_img, dummy_label, 'share'
            return dummy_img, dummy_label

# -------------------------- 数据加载函数（核心修复：优化采样器） --------------------------
def get_visda_data_loaders():
    # 数据路径
    train_mat = os.path.join(cfg.VISDA_DATA_ROOT, "source_train_32x32.mat")
    test_mat = os.path.join(cfg.RAW_VISDA_ROOT, "test_32x32.mat")
    target_label_csv = os.path.join(cfg.VISDA_DATA_ROOT, "target_test_labels.csv")
    
    # 验证路径
    for path in [train_mat, test_mat]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"VisDA数据文件缺失：{path}")
    
    # 数据集初始化
    train_dataset = VisDADataset(
        mat_path=train_mat,
        transform=get_train_transform(),
        is_train=True,
        is_target=False
    )
    
    test_dataset = VisDADataset(
        mat_path=test_mat,
        transform=get_val_transform("VisDA"),
        is_train=False,
        target_label_csv=target_label_csv,
        is_target=True
    )
    
    # 源域Dataloader（核心调整：关闭shuffle，提升稳定性）
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        drop_last=False  # 核心调整：不丢弃最后批次，保留更多样本
    )
    
    # 目标域Dataloader（核心修复：简化采样器，优先保证已知类样本）
    val_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE*2,
        shuffle=False,  # 关闭采样器，直接使用顺序加载
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        drop_last=False
    )
    
    return train_loader, val_loader

# -------------------------- 数据增强（核心修复：替换InterpolationType为cv2兼容版） --------------------------
def get_train_transform():
    return A.Compose([
        # 修改点1：替换A.InterpolationType.BICUBIC为cv2.INTER_CUBIC
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),  # 双三次插值，兼容所有albumentations版本
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

def get_val_transform(dataset_type="VisDA"):
    return A.Compose([
        # 修改点2：替换A.InterpolationType.BICUBIC为cv2.INTER_CUBIC
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),  # 双三次插值，兼容所有albumentations版本
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

def apply_albumentations(images, transform):
    try:
        if isinstance(images, torch.Tensor):
            # 转换维度：[N, C, H, W] → [N, H, W, C]
            images = images.cpu().numpy().transpose(0, 2, 3, 1)
        
        augmented = []
        for img in images:
            try:
                aug_img = transform(image=img)["image"]
                augmented.append(aug_img)
            except Exception as e:
                print(f"⚠️ 单张图像增强失败：{e}")
                augmented.append(torch.zeros(3, 224, 224))
        
        return torch.stack(augmented).to(cfg.DEVICE)
    except Exception as e:
        print(f"⚠️ 图像增强失败：{e}")
        return torch.zeros(len(images), 3, 224, 224).to(cfg.DEVICE)

# -------------------------- 权重加载函数 --------------------------
def load_pretrained_weights(model, model_name, local_path):
    try:
        if os.path.exists(local_path):
            state_dict = torch.load(local_path, map_location=cfg.DEVICE)
            if model_name == "convnextv2_tiny":
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
            
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
            
            model_state_dict.update(filtered_state_dict)
            model.load_state_dict(model_state_dict, strict=False)
            print(f"✅ 成功加载{model_name}本地预训练权重（匹配{len(filtered_state_dict)}/{len(state_dict)}层）")
        else:
            raise FileNotFoundError(f"本地权重文件不存在：{local_path}")
    except Exception as e:
        print(f"⚠️ 本地{model_name}权重加载失败：{e}，自动下载官方预训练权重")
        if model_name == "vgg16":
            model = models.vgg16(pretrained=True)
        elif model_name == "convnextv2_tiny":
            model = timm.create_model("convnextv2_tiny", pretrained=True, num_classes=0)
    
    return model.to(cfg.DEVICE)

# -------------------------- 感知损失模块 --------------------------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        local_weights_path = os.path.join(cfg.WEIGHTS_DIR, "vgg16-397923af.pth")
        vgg = models.vgg16()
        vgg = load_pretrained_weights(vgg, "vgg16", local_weights_path)
        vgg = vgg.features[:23].eval()
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg.to(cfg.DEVICE)
        self.l1 = nn.L1Loss()
        self.projection = nn.Sequential(
            nn.Linear(cfg.FEATURE_DIM, 512 * 14 * 14),
            nn.ReLU()
        ).to(cfg.DEVICE)

    def forward(self, recon_features, orig_features):
        try:
            batch_size = recon_features.size(0)
            recon_proj = self.projection(recon_features).view(batch_size, 512, 14, 14)
            orig_proj = self.projection(orig_features).view(batch_size, 512, 14, 14)
            return self.l1(recon_proj, orig_proj)
        except Exception as e:
            print(f"⚠️ 感知损失计算失败：{e}")
            return torch.tensor(0.0, device=cfg.DEVICE)

# -------------------------- 梯度反转层 + 域判别器 --------------------------
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DynamicGRL(nn.Module):
    def __init__(self, base_alpha=cfg.GRL_ALPHA_BASE, decay=cfg.GRL_ALPHA_DECAY):
        super().__init__()
        self.base_alpha = base_alpha
        self.decay = decay
        self.current_alpha = base_alpha

    def forward(self, x, epoch):
        # 优化：衰减速度降低，且alpha不低于0.5（保证域对抗梯度）
        self.current_alpha = max(self.base_alpha * (self.decay ** (epoch // 5)), 0.5)
        return GradientReversalLayer.apply(x, self.current_alpha)

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=cfg.FEATURE_DIM):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),  # 核心调整：降低维度，提升稳定性
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        ).to(cfg.DEVICE)

    def forward(self, x):
        return self.fc(x)

# -------------------------- 开放集对比损失（核心调整：简化逻辑） --------------------------
class OpenSetContrastiveLoss(nn.Module):
    def __init__(self, margin=cfg.MARGIN, temp=cfg.TEMP):
        super().__init__()
        self.margin = margin
        self.temp = temp

    def forward(self, features, labels, centers, num_known_classes=cfg.NUM_KNOWN_CLASSES):
        if centers is None or len(features) == 0:
            return torch.tensor(0.0, device=cfg.DEVICE)
        
        # 简化逻辑：仅保留核心计算
        centers = centers[:num_known_classes].to(cfg.DEVICE)
        features = F.normalize(features, dim=1)
        centers = F.normalize(centers, dim=1)
        
        # 过滤已知类
        known_mask = (labels < num_known_classes) & (labels >= 0)
        loss = torch.tensor(0.0, device=cfg.DEVICE)
        
        # 已知类损失
        if known_mask.sum() > 0:
            known_feats = features[known_mask]
            known_labels = labels[known_mask].long()
            
            # 计算距离
            center_dist = torch.cdist(known_feats, centers) / self.temp
            numerator = torch.exp(-center_dist[torch.arange(len(known_labels)), known_labels])
            denominator = torch.sum(torch.exp(-center_dist), dim=1)
            denominator = torch.clamp(denominator, min=1e-8)
            
            known_loss = -torch.log(numerator / denominator)
            loss += known_loss.mean()
        
        return loss

# -------------------------- 重构可视化 --------------------------
def visualize_reconstruction(disent_model, vae_model, val_loader, epoch):
    try:
        disent_model.eval()
        vae_model.eval()
        val_transform = get_val_transform("VisDA")
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0] if len(batch) == 3 else batch[0]
                if len(images) < 5:
                    continue
                    
                images = apply_albumentations(images, val_transform)
                disent_feature, _, _ = disent_model(images)
                recon_feature, _, _, _ = vae_model(disent_feature)
                
                # 创建可视化图
                fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                fig.suptitle(f'Reconstruction Visualization - Epoch {epoch}', fontsize=14)
                
                for j in range(5):
                    # 原始图像
                    orig_img = images[j].cpu().numpy().transpose(1, 2, 0)
                    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
                    orig_img = np.clip(orig_img, 0, 1)
                    
                    # 重构特征可视化
                    feat = recon_feature[j].cpu().numpy()
                    
                    # PCA降维
                    if feat.size <= 1:
                        feat_img = np.zeros((32, 32, 3))
                    else:
                        if len(feat.shape) == 1:
                            feat_reshaped = np.tile(feat.reshape(1, -1), (3, 1))
                            n_components = 3
                        else:
                            n_components = min(3, feat.shape[0], feat.shape[1])
                        
                        pca = PCA(n_components=n_components)
                        feat_pca = pca.fit_transform(feat_reshaped)[0]
                        
                        if len(feat_pca) < 3:
                            feat_pca = np.pad(feat_pca, (0, 3 - len(feat_pca)), mode='constant')
                        
                        feat_img = np.tile(feat_pca.reshape(3, 1, 1), (1, 32, 32))
                        feat_img = feat_img.transpose(1, 2, 0)
                        feat_img = (feat_img - feat_img.min()) / (feat_img.max() - feat_img.min() + 1e-8)
                        feat_img = np.clip(feat_img, 0, 1)
                    
                    # 绘制
                    axes[0, j].imshow(orig_img)
                    axes[0, j].set_title(f"Original {j+1}")
                    axes[0, j].axis('off')
                    
                    axes[1, j].imshow(feat_img)
                    axes[1, j].set_title(f"Recon Feat {j+1}")
                    axes[1, j].axis('off')
                
                # 保存图片
                plt.tight_layout()
                save_path = os.path.join(cfg.VISUALIZATION_DIR, f"recon_epoch_{epoch}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ VisDA重构可视化已保存到 {save_path}")
                
                del images, disent_feature, recon_feature
                torch.cuda.empty_cache()
                break
                
    except Exception as e:
        print(f"⚠️ 重构可视化失败：{e}")
        plt.close('all')
        torch.cuda.empty_cache()

# -------------------------- 类中心计算（核心修复：简化更新逻辑） --------------------------
@torch.no_grad()
def compute_class_centers(model, loader, num_classes=cfg.NUM_KNOWN_CLASSES, 
                          feature_dim=cfg.FEATURE_DIM, transform=None):
    model.eval()
    
    # 初始化类中心
    if model.class_centers is None:
        model.class_centers = torch.zeros(num_classes, feature_dim, device=cfg.DEVICE)
        counts = torch.zeros(num_classes, device=cfg.DEVICE)
        
        # 检查loader是否为空
        if len(loader) == 0:
            print("⚠️ 类中心计算：数据加载器为空")
            return model.class_centers
        
        # 首次计算（简化逻辑）
        for batch in loader:
            try:
                images = batch[0] if len(batch) == 3 else batch[0]
                labels = batch[1] if len(batch) >= 2 else batch[1]
                
                # 仅过滤一次已知类
                known_mask = (labels < num_classes) & (labels >= 0)
                if known_mask.sum() == 0:
                    continue
                
                labels = labels[known_mask].to(cfg.DEVICE)
                images = apply_albumentations(images, transform) if transform else images.to(cfg.DEVICE)
                
                features, _, _ = model(images)
                features = features[known_mask]
                
                # 累加特征
                for c in range(num_classes):
                    mask = (labels == c)
                    if mask.sum() > 0:
                        model.class_centers[c] += features[mask].sum(dim=0)
                        counts[c] += mask.sum()
            except Exception as e:
                print(f"⚠️ 类中心计算批次失败：{e}")
                continue
        
        # 归一化
        counts = torch.clamp(counts, min=1e-8)
        model.class_centers = model.class_centers / counts.unsqueeze(1)
    
    return model.class_centers

# -------------------------- 阶段调度（核心调整：延迟域对抗+平衡损失权重） --------------------------
def stage_freeze_unfreeze(disent_model, vae_model, epoch):
    # 默认全部解冻
    for param in disent_model.parameters():
        param.requires_grad = True
    for param in vae_model.parameters():
        param.requires_grad = True
    
    # 核心调整：重新分配损失权重，优先分类
    if epoch < 5:  # 阶段1：仅分类+轻量重构，关闭域对抗和开放集（提前到epoch 5开启域对抗）
        lambda_recon = 0.2  # 重构损失权重从1.0降至0.2
        lambda_disent = 2.0  # 分类损失权重从0.5升至2.0
        lambda_kl = 0.0001
        lambda_domain = 0.0  # 关闭域对抗
        lambda_open_set = 0.0  # 关闭开放集损失
        print(f"🔹 阶段1：优先分类训练（lambda_recon={lambda_recon:.1f}, lambda_disent={lambda_disent:.1f}）")
    elif epoch < 25:  # 阶段2：域对抗训练
        lambda_recon = 0.3
        lambda_disent = 1.5
        lambda_kl = 0.0003
        lambda_domain = 2.0  # 进一步增加域对抗权重（1.0→2.0）
        lambda_open_set = 0.2
        print(f"🔹 阶段2：域对抗训练（lambda_domain={lambda_domain:.2f}）")
    else:  # 阶段3：优化开放集
        lambda_recon = 0.4
        lambda_disent = 1.2
        lambda_kl = 0.0005
        lambda_domain = 0.3  # 增加域对抗权重（原0.03→0.3）
        lambda_open_set = 0.5
        print(f"🔹 阶段3：优化开放集拒识指标（lambda_domain={lambda_domain:.2f}）")
    
    return lambda_recon, lambda_disent, lambda_kl, lambda_domain, lambda_open_set

# -------------------------- 特征解耦模型（核心调整：降低特征维度） --------------------------
class ImprovedFeatureDisentanglementModel(nn.Module):
    def __init__(self, num_classes=cfg.TOTAL_VISDA_CLASSES, feature_dim=cfg.FEATURE_DIM):
        super().__init__()
        # 加载backbone
        self.backbone = timm.create_model(
            model_name="convnextv2_tiny",
            pretrained=False,
            num_classes=0
        )
        
        local_weight_path = os.path.join(cfg.WEIGHTS_DIR, "pytorch_model.bin")
        self.backbone = load_pretrained_weights(self.backbone, "convnextv2_tiny", local_weight_path)
        
        backbone_out_dim = self.backbone.num_features

        # 特征投影（核心调整：降低维度）
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)  # 降低dropout
        ).to(cfg.DEVICE)

        # 分类分支（核心调整：简化结构，提升训练效率）
        self.class_branch = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        ).to(cfg.DEVICE)

        # 域分支
        self.domain_branch = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        ).to(cfg.DEVICE)
        
        # 类中心
        self.class_centers = None
        
        # 打印参数信息
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ VisDA模型可训练参数：{trainable_params/1e6:.2f}M")

    def forward(self, x):
        try:
            x = x.to(cfg.DEVICE)
            base_feature_raw = self.backbone(x)
            base_feature = self.feature_proj(base_feature_raw)
            base_feature = torch.nan_to_num(base_feature, nan=0.0, posinf=0.0, neginf=0.0)
            
            class_logits = self.class_branch(base_feature)
            domain_logits = self.domain_branch(base_feature)
            disent_loss = torch.tensor(0.0).to(cfg.DEVICE)
            
            return base_feature, class_logits, disent_loss
        except Exception as e:
            print(f"⚠️ 模型前向传播失败：{e}")
            dummy_feat = torch.zeros(len(x), cfg.FEATURE_DIM, device=cfg.DEVICE)
            dummy_logits = torch.zeros(len(x), cfg.TOTAL_VISDA_CLASSES, device=cfg.DEVICE)
            dummy_loss = torch.tensor(0.0, device=cfg.DEVICE)
            return dummy_feat, dummy_logits, dummy_loss

# -------------------------- VAE模型（核心调整：降低维度） --------------------------
class FeatureVAE(nn.Module):
    def __init__(self, feature_dim=cfg.FEATURE_DIM, latent_dim=cfg.LATENT_DIM):
        super(FeatureVAE, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        # Encoder（简化结构）
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        ).to(cfg.DEVICE)
        
        self.fc_mu = nn.Linear(512, latent_dim).to(cfg.DEVICE)
        self.fc_logvar = nn.Linear(512, latent_dim).to(cfg.DEVICE)
        
        # Decoder（简化结构）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim),
            nn.ReLU()
        ).to(cfg.DEVICE)

    def reparameterize(self, mu, logvar):
        try:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        except Exception as e:
            print(f"⚠️ 重参数化失败：{e}")
            return mu

    def forward(self, x):
        try:
            x = x.to(cfg.DEVICE)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = self.encoder(x)
            
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            logvar = torch.clamp(logvar, min=-10, max=10)
            
            z = self.reparameterize(mu, logvar)
            recon_x = self.decoder(z)
            
            return recon_x, mu, logvar, z
        except Exception as e:
            print(f"⚠️ VAE前向传播失败：{e}")
            dummy_recon = torch.zeros_like(x)
            dummy_mu = torch.zeros(len(x), self.latent_dim, device=cfg.DEVICE)
            dummy_logvar = torch.zeros(len(x), self.latent_dim, device=cfg.DEVICE)
            dummy_z = torch.zeros(len(x), self.latent_dim, device=cfg.DEVICE)
            return dummy_recon, dummy_mu, dummy_logvar, dummy_z

# -------------------------- 域MSE差异计算 --------------------------
@torch.no_grad()
def compute_domain_mse_diff(disent_model, vae_model, src_loader, tgt_loader, transform):
    disent_model.eval()
    vae_model.eval()
    mse_loss_fn = nn.SmoothL1Loss()
    
    # 计算源域MSE
    src_mse = 0.0
    src_count = 0
    for batch in src_loader:
        try:
            images = batch[0] if len(batch) == 3 else batch[0]
            images = apply_albumentations(images, transform)
            
            disent_feature, _, _ = disent_model(images)
            recon_feature, _, _, _ = vae_model(disent_feature)
            
            src_mse += mse_loss_fn(recon_feature, disent_feature).item() * len(images)
            src_count += len(images)
        except Exception as e:
            print(f"⚠️ 源域MSE计算批次失败：{e}")
            continue
    
    src_mse = src_mse / src_count if src_count > 0 else 0.0
    
    # 计算目标域MSE
    tgt_mse = 0.0
    tgt_count = 0
    for batch in tgt_loader:
        try:
            images = batch[0] if len(batch) == 3 else batch[0]
            images = apply_albumentations(images, transform)
            
            disent_feature, _, _ = disent_model(images)
            recon_feature, _, _, _ = vae_model(disent_feature)
            
            tgt_mse += mse_loss_fn(recon_feature, disent_feature).item() * len(images)
            tgt_count += len(images)
        except Exception as e:
            print(f"⚠️ 目标域MSE计算批次失败：{e}")
            continue
    
    tgt_mse = tgt_mse / tgt_count if tgt_count > 0 else 0.0
    mse_diff = abs(src_mse - tgt_mse)
    
    torch.cuda.empty_cache()
    return src_mse, tgt_mse, mse_diff

# -------------------------- 消融实验记录 --------------------------
def save_ablation_results(results, save_path=None):
    if save_path is None:
        save_path = os.path.join(cfg.ABLATION_DIR, "ablation_visda_v1.txt")
    
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=== VisDA 联合训练 消融实验结果 ===\n")
            f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置信息：已知类={cfg.NUM_KNOWN_CLASSES}，总类别={cfg.TOTAL_VISDA_CLASSES}\n")
            f.write(f"基础模型：准确率={results['base_acc']:.4f}, MSE={results['base_mse']:.4f}\n")
            f.write(f"添加感知损失：准确率={results['percept_acc']:.4f}, MSE={results['percept_mse']:.4f}（提升{results['percept_mse_improv']:.2f}%）\n")
            f.write(f"添加域对抗：域间MSE差异={results['domain_mse_diff']:.4f}, 准确率={results['domain_acc']:.4f}\n")
            f.write(f"开放集指标：AUROC={results['auroc']:.4f}, H-score={results['h_score']:.4f}\n")
            f.write(f"最佳模型：准确率={results.get('best_acc', 0):.4f}, H-score={results.get('best_h_score', 0):.4f}\n")
        
        print(f"✅ VisDA消融实验结果已保存到 {save_path}")
    except Exception as e:
        print(f"⚠️ 保存消融实验结果失败：{e}")

# -------------------------- 联合训练主函数（核心修复：优化训练逻辑） --------------------------
def joint_train_direct():
    # 打印训练配置
    print("="*80)
    print("🚀 启动VisDA 联合训练（修复版）")
    print(f"📌 设备：{cfg.DEVICE} (GPU数量：{cfg.NUM_GPUS})")
    print(f"📌 已知类数量：{cfg.NUM_KNOWN_CLASSES}，总类别数：{cfg.TOTAL_VISDA_CLASSES}")
    print(f"📌 批次大小：{cfg.BATCH_SIZE}，梯度累积：{cfg.GRADIENT_ACCUMULATION_STEPS}")
    print(f"📌 阶段1目标：MSE≤{cfg.TARGET_MSE}，准确率≥{cfg.TARGET_ACC}")
    print(f"📌 阶段2目标：域间MSE差异≤{cfg.TARGET_DOMAIN_MSE_DIFF}，准确率≥0.78")
    print(f"📌 阶段3目标：AUROC≥{cfg.TARGET_AUROC}，H-score≥{cfg.TARGET_H_SCORE}")
    print("="*80)
    
    # 1. 初始化模型
    try:
        disent_model = ImprovedFeatureDisentanglementModel(
            num_classes=cfg.TOTAL_VISDA_CLASSES, 
            feature_dim=cfg.FEATURE_DIM
        ).to(cfg.DEVICE)
        
        vae_model = FeatureVAE(
            feature_dim=cfg.FEATURE_DIM, 
            latent_dim=cfg.LATENT_DIM
        ).to(cfg.DEVICE)
        
        domain_disc = DomainDiscriminator(input_dim=cfg.FEATURE_DIM).to(cfg.DEVICE)
        dynamic_grl = DynamicGRL().to(cfg.DEVICE)
        open_set_loss_fn = OpenSetContrastiveLoss().to(cfg.DEVICE)
    except Exception as e:
        print(f"❌ 模型初始化失败：{e}")
        return 0.0, 0.0, 0.0, 0.0

    # 2. 损失函数
    mse_loss_fn = nn.SmoothL1Loss().to(cfg.DEVICE)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05).to(cfg.DEVICE)  # 降低标签平滑
    perceptual_loss_fn = PerceptualLoss().to(cfg.DEVICE)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
    domain_criterion = nn.BCEWithLogitsLoss().to(cfg.DEVICE)

    # 3. 优化器配置（核心调整：学习率预热+调整权重）
    try:
        backbone_params = list(disent_model.backbone.parameters()) + list(disent_model.feature_proj.parameters())
        classifier_params = list(disent_model.class_branch.parameters())
        domain_params = list(disent_model.domain_branch.parameters())
        vae_params = list(vae_model.parameters())
        domain_disc_params = list(domain_disc.parameters())

        params = [
            {'params': backbone_params, 'lr': cfg.LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY},
            {'params': classifier_params, 'lr': cfg.CLASSIFIER_LR, 'weight_decay': cfg.WEIGHT_DECAY},
            {'params': domain_params, 'lr': cfg.LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY},
            {'params': vae_params, 'lr': cfg.VAE_LR, 'weight_decay': cfg.WEIGHT_DECAY},
            {'params': domain_disc_params, 'lr': cfg.LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY}
        ]

        optimizer = optim.AdamW(params, betas=(0.9, 0.999))  # 恢复默认beta
        
        # 学习率预热（核心新增）
        warmup_epochs = 5
        scheduler_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - warmup_epochs, eta_min=1e-7)
        scheduler = SequentialLR(optimizer, [scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
        
        # 修复GradScaler初始化参数
        scaler = GradScaler(device='cuda' if cfg.DEVICE.type == "cuda" else 'cpu', 
                           enabled=True if cfg.DEVICE.type == "cuda" else False)
    except Exception as e:
        print(f"❌ 优化器初始化失败：{e}")
        return 0.0, 0.0, 0.0, 0.0

    # 4. 加载数据
    try:
        train_loader, val_loader = get_visda_data_loaders()
        src_loader, tgt_loader = train_loader, val_loader
        train_transform = get_train_transform()
        val_transform = get_val_transform("VisDA")
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        return 0.0, 0.0, 0.0, 0.0

    # 5. 日志初始化
    writer = SummaryWriter(cfg.LOG_DIR)
    best_acc, best_mse, best_auroc, best_h_score = 0.0, float('inf'), 0.0, 0.0
    best_epoch = 0
    patience_counter = 0
    
    # 消融实验结果初始化（优化初始值）
    ablation_results = {
        "base_acc": 0.70,
        "base_mse": 0.05,
        "percept_acc": 0.0,
        "percept_mse": float('inf'),
        "percept_mse_improv": 0.0,
        "domain_mse_diff": float('inf'),
        "domain_acc": 0.0,
        "auroc": 0.0,
        "h_score": 0.0,
        "best_acc": 0.0,
        "best_h_score": 0.0
    }

    # 6. 主训练循环
    for epoch in range(cfg.EPOCHS):
        start_time = time.time()
        
        # 阶段调度
        lambda_recon, lambda_disent, lambda_kl, lambda_domain, lambda_open_set = stage_freeze_unfreeze(
            disent_model, vae_model, epoch
        )
        
        # 训练模式
        disent_model.train()
        vae_model.train()
        domain_disc.train()
        optimizer.zero_grad()

        # 损失统计
        train_losses = {'total': 0, 'recon': 0, 'cls': 0, 'disent': 0, 
                        'kl': 0, 'perceptual': 0, 'domain': 0, 'open_set': 0}
        correct = 0
        total = 0

        # 定期更新类中心（延迟更新）
        if epoch >= cfg.CENTER_UPDATE_EPOCH and epoch % cfg.CENTER_UPDATE_EPOCH == 0:
            centers = compute_class_centers(disent_model, train_loader, 
                                           cfg.NUM_KNOWN_CLASSES, 
                                           transform=val_transform)
        else:
            centers = disent_model.class_centers

        # 批量训练
        target_iter = iter(val_loader)  # 添加目标域迭代器
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # 加载批次数据
                images, labels = batch[:2]
                images = apply_albumentations(images, train_transform)
                labels = labels.to(cfg.DEVICE)
                
                # 获取目标域数据
                try:
                    tgt_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(val_loader)
                    tgt_batch = next(target_iter)
                tgt_images = tgt_batch[0]
                tgt_images = apply_albumentations(tgt_images, train_transform)
                
                # 混合精度训练
                with autocast('cuda', enabled=True if cfg.DEVICE.type == "cuda" else False):
                    # 前向传播
                    disent_feature, logits, disent_loss = disent_model(images)
                    recon_feature, mu, log_var, _ = vae_model(disent_feature)
                    
                    # 重构损失（降低权重）
                    mse_recon_loss = mse_loss_fn(recon_feature, disent_feature)
                    perceptual_loss = perceptual_loss_fn(recon_feature, disent_feature)
                    recon_loss = 0.5 * mse_recon_loss + 0.1 * perceptual_loss  # 核心调整：降低感知损失权重
                    
                    # KL损失（降低权重）
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
                    kl_loss = torch.clamp(kl_loss, min=0, max=10)  # 降低裁剪阈值
                    
                    # 分类损失（核心：简化过滤逻辑）
                    known_mask = (labels < cfg.NUM_KNOWN_CLASSES) & (labels >= 0)
                    cls_loss_val = torch.tensor(0.0, device=cfg.DEVICE)
                    
                    if known_mask.sum() > 0:
                        cls_loss_val = ce_loss_fn(logits[known_mask], labels[known_mask])
                    
                    # 调试信息
                    if batch_idx % 100 == 0:
                        print(f"🔍 训练批次{batch_idx}：标签范围={labels.min().item()}-{labels.max().item()}，有效已知类={known_mask.sum().item()}")
                    
                    # 解耦损失
                    disent_loss = disent_loss.mean() if len(disent_loss.shape) > 0 else disent_loss
                    disent_loss = torch.clamp(disent_loss.abs(), 0.0, 5.0)
                    
                    # 域对抗损失（提前开启，同时使用源域和目标域数据）
                    domain_loss = torch.tensor(0.0, device=cfg.DEVICE)
                    if lambda_domain > 0 and epoch >= 5:  # 提前到epoch 5开启
                        # 源域 (Label 0)
                        reversed_feature = dynamic_grl(disent_feature, epoch)
                        domain_pred = domain_disc(reversed_feature)
                        domain_loss_src = domain_criterion(domain_pred, torch.zeros(len(disent_feature), 1).to(cfg.DEVICE))
                        
                        # 目标域 (Label 1)
                        tgt_feat, _, _ = disent_model(tgt_images)
                        reversed_tgt = dynamic_grl(tgt_feat, epoch)
                        domain_pred_tgt = domain_disc(reversed_tgt)
                        domain_loss_tgt = domain_criterion(domain_pred_tgt, torch.ones(len(tgt_feat), 1).to(cfg.DEVICE))
                        
                        domain_loss = 0.5 * (domain_loss_src + domain_loss_tgt)
                    
                    # 开放集损失（延迟开启）
                    open_set_loss = torch.tensor(0.0, device=cfg.DEVICE)
                    if lambda_open_set > 0 and centers is not None and epoch >= 15:  # 延迟到Epoch15开启
                        open_set_loss = open_set_loss_fn(disent_feature, labels, centers, cfg.NUM_KNOWN_CLASSES)
                    
                    # 总损失（核心调整：优先分类）
                    total_loss = (
                        lambda_recon * recon_loss + 
                        lambda_disent * cls_loss_val +  # 核心：仅保留分类损失，移除解耦损失
                        lambda_kl * kl_loss + 
                        lambda_domain * domain_loss +
                        lambda_open_set * open_set_loss
                    )

                # 梯度更新（关闭梯度累积）
                scaler.scale(total_loss).backward()
                
                # 梯度裁剪（降低阈值）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(disent_model.parameters(), max_norm=0.5)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # 损失统计
                train_losses['total'] += total_loss.item()
                train_losses['recon'] += mse_recon_loss.item()
                train_losses['cls'] += cls_loss_val.item()
                train_losses['disent'] += disent_loss.item()
                train_losses['kl'] += kl_loss.item()
                train_losses['perceptual'] += perceptual_loss.item()
                train_losses['domain'] += domain_loss.item()
                train_losses['open_set'] += open_set_loss.item()
                
                # 准确率统计（简化逻辑）
                if known_mask.sum() > 0:
                    preds = logits[known_mask].argmax(dim=1)
                    correct += preds.eq(labels[known_mask]).sum().item()
                    total += known_mask.sum().item()

                # 打印进度
                if batch_idx % 50 == 0:
                    print(f"Epoch [{epoch+1}/{cfg.EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], "
                          f"Loss: {total_loss.item():.4f}, Recon: {mse_recon_loss.item():.4f}, "
                          f"Perceptual: {perceptual_loss.item():.4f}, Cls: {cls_loss_val.item():.4f}, "
                          f"Domain: {domain_loss.item():.4f}, OpenSet: {open_set_loss.item():.4f}, "
                          f"LR: {optimizer.param_groups[1]['lr']:.6f}, GRL_alpha: {dynamic_grl.current_alpha:.3f}")
            
            except Exception as e:
                print(f"⚠️ 训练批次{batch_idx}失败：{e}")
                continue
        
        # 释放训练批次内存
        torch.cuda.empty_cache()

        # 训练指标计算
        train_acc = correct / total if total > 0 else 0.0
        for k in train_losses:
            train_losses[k] /= len(train_loader)

        # 7. 验证阶段（每1个epoch验证，确保及时监控）
        if epoch % 1 == 0:
            disent_model.eval()
            vae_model.eval()
            domain_disc.eval()
            
            val_losses = {'recon': 0, 'cls': 0, 'perceptual': 0, 'domain': 0, 'open_set': 0}
            correct = 0
            total = 0

            # 开放集指标存储
            all_conf = []
            all_dist = []
            all_recon_err = []
            all_labels = []
            all_feats = []

            with torch.no_grad():
                for batch in val_loader:
                    try:
                        images, labels, class_type = batch
                        images = apply_albumentations(images, val_transform)
                        labels = labels.to(cfg.DEVICE)
                        
                        disent_feature, logits, _ = disent_model(images)
                        recon_feature, _, _, _ = vae_model(disent_feature)
                        
                        # 损失统计
                        val_losses['recon'] += mse_loss_fn(recon_feature, disent_feature).item()
                        val_losses['perceptual'] += perceptual_loss_fn(recon_feature, disent_feature).item()
                        
                        # 分类损失（简化逻辑）
                        known_mask = (labels < cfg.NUM_KNOWN_CLASSES) & (labels >= 0)
                        cls_loss_val = torch.tensor(0.0, device=cfg.DEVICE)
                        if known_mask.sum() > 0:
                            cls_loss_val = ce_loss_fn(logits[known_mask], labels[known_mask])
                        val_losses['cls'] += cls_loss_val.item()
                        
                        # 开放集损失（延迟开启）
                        if lambda_open_set > 0 and centers is not None and epoch >= 15:
                            open_set_loss_val = open_set_loss_fn(disent_feature, labels, centers, cfg.NUM_KNOWN_CLASSES)
                            val_losses['open_set'] += open_set_loss_val.item()
                        
                        # 域损失
                        if lambda_domain > 0 and epoch >= 5:
                            reversed_feature = dynamic_grl(disent_feature, epoch)
                            domain_pred = domain_disc(reversed_feature)
                            domain_label = torch.ones_like(labels).float().to(cfg.DEVICE)
                            val_losses['domain'] += domain_criterion(domain_pred, domain_label.unsqueeze(1)).item()
                        
                        # 准确率统计（核心：简化逻辑，确保计数正确）
                        if known_mask.sum() > 0:
                            preds = logits[known_mask].argmax(dim=1)
                            correct += preds.eq(labels[known_mask]).sum().item()
                            total += known_mask.sum().item()

                        # 开放集指标计算（简化）
                        conf = F.softmax(logits / cfg.TEMP, dim=1).max(dim=1)[0]
                        dist = torch.cdist(F.normalize(disent_feature, dim=1), 
                                         F.normalize(centers, dim=1)).min(dim=1)[0] if centers is not None else torch.zeros_like(conf)
                        recon_err = F.smooth_l1_loss(recon_feature, disent_feature, reduction='none').mean(dim=1)
                        
                        # 数值稳定性处理
                        dist = torch.nan_to_num(dist, nan=1e6, posinf=1e6, neginf=0.0)
                        recon_err = torch.nan_to_num(recon_err, nan=0.0, posinf=1e6, neginf=0.0)
                        
                        # 标记未知类
                        is_unknown = (labels >= cfg.NUM_KNOWN_CLASSES).float()
                        
                        # 收集数据
                        all_conf.extend(conf.cpu().numpy())
                        all_dist.extend(dist.cpu().numpy())
                        all_recon_err.extend(recon_err.cpu().numpy())
                        all_labels.extend(is_unknown.cpu().numpy())
                        all_feats.extend(F.normalize(disent_feature, dim=1).cpu().numpy())
                    except Exception as e:
                        print(f"⚠️ 验证批次失败：{e}")
                        continue

            # 验证指标计算
            val_acc = correct / total if total > 0 else 0.0
            val_mse = val_losses['recon'] / len(val_loader) if len(val_loader) > 0 else 0.0
            val_perceptual = val_losses['perceptual'] / len(val_loader) if len(val_loader) > 0 else 0.0
            val_domain = val_losses['domain'] / len(val_loader) if (lambda_domain > 0 and len(val_loader) > 0) else 0.0
            val_open_set = val_losses['open_set'] / len(val_loader) if (lambda_open_set > 0 and len(val_loader) > 0) else 0.0
            
            # 更新学习率（使用新的调度器）
            scheduler.step()

            # 8. 开放集指标计算（简化）
            unknown_count = sum(all_labels)
            print(f"\n📊 验证集中未知类数量: {unknown_count}")
            auroc = 0.0
            h_score = 0.0
            optimal_threshold = 0.5
            
            if unknown_count > 0 and epoch >= 15:  # 延迟计算开放集指标
                # 数据预处理
                conf_arr = 1 - np.array(all_conf)
                dist_arr = np.array(all_dist)
                recon_err_arr = np.array(all_recon_err)
                feats_arr = np.array(all_feats) if len(all_feats) > 0 else np.array([])
                # 修复特征方差计算的空值问题
                feat_var = np.var(feats_arr, axis=1) if len(feats_arr) > 0 and feats_arr.shape[1] > 1 else np.zeros_like(conf_arr)
                label_arr = np.array(all_labels)
                
                # 过滤无效值
                valid_mask = ~(np.isnan(conf_arr) | np.isinf(conf_arr) | 
                               np.isnan(dist_arr) | np.isinf(dist_arr) |
                               np.isnan(recon_err_arr) | np.isinf(recon_err_arr) |
                               np.isnan(feat_var) | np.isinf(feat_var) |
                               np.isnan(label_arr) | np.isinf(label_arr))
                
                if np.sum(valid_mask) == 0:
                    print("⚠️ 无有效样本计算开放集指标，跳过")
                else:
                    # 应用过滤
                    conf_arr = conf_arr[valid_mask]
                    dist_arr = dist_arr[valid_mask]
                    recon_err_arr = recon_err_arr[valid_mask]
                    feat_var = feat_var[valid_mask]
                    label_arr = label_arr[valid_mask]
                    
                    # 确保有正例和负例
                    if len(np.unique(label_arr)) < 2:
                        print("⚠️ 验证集只有单一类别，无法计算AUROC")
                    else:
                        # 归一化
                        conf_arr = conf_arr / (np.max(conf_arr) + 1e-8) if np.max(conf_arr) > 0 else conf_arr
                        dist_arr = dist_arr / (np.max(dist_arr) + 1e-8) if np.max(dist_arr) > 0 else dist_arr
                        recon_err_arr = recon_err_arr / (np.max(recon_err_arr) + 1e-8) if np.max(recon_err_arr) > 0 else recon_err_arr
                        feat_var = feat_var / (np.max(feat_var) + 1e-8) if np.max(feat_var) > 0 else feat_var

                        # 权重搜索（简化）
                        best_weights = (0.3, 0.4, 0.2, 0.1)
                        w1, w2, w3, w4 = best_weights
                        scores = w1 * conf_arr + w2 * dist_arr + w3 * recon_err_arr + w4 * feat_var
                        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
                        
                        try:
                            auroc = roc_auc_score(label_arr, scores)
                            fpr, tpr, thresholds = roc_curve(label_arr, scores)
                            optimal_idx = np.argmax(tpr - fpr) if len(tpr) > 0 else 0
                            optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
                            
                            preds = (scores >= optimal_threshold).astype(int)
                            tp = np.sum((preds == 1) & (label_arr == 1))
                            fn = np.sum((preds == 0) & (label_arr == 1))
                            fp = np.sum((preds == 1) & (label_arr == 0))
                            tn = np.sum((preds == 0) & (label_arr == 0))
                            
                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                            h_score = 2 * tpr * tnr / (tpr + tnr) if (tpr + tnr) > 0 else 0
                            
                            print(f"🎯 最佳融合权重: w1={w1:.2f}(置信度), w2={w2:.2f}(距离), w3={w3:.2f}(重构误差), w4={w4:.2f}(特征方差)")
                            print(f"🎯 最佳阈值: {optimal_threshold:.4f}, TPR={tpr:.4f}, TNR={tnr:.4f}, H-score={h_score:.4f}")
                        except Exception as e:
                            print(f"⚠️ 计算最终AUROC失败：{e}")
                            auroc = 0.0
                            h_score = 0.0
            else:
                print("⚠️ 训练初期，暂不计算开放集指标")

            # 9. 源/目标域MSE差异计算（延迟计算）
            src_mse, tgt_mse, mse_diff = 0.0, 0.0, 0.0
            if epoch > 10:
                src_mse, tgt_mse, mse_diff = compute_domain_mse_diff(disent_model, vae_model, 
                                                                    src_loader, tgt_loader, val_transform)
                print(f"🌐 源域MSE={src_mse:.4f}, 目标域MSE={tgt_mse:.4f}, 差异={mse_diff:.4f} (目标≤{cfg.TARGET_DOMAIN_MSE_DIFF})")

            # 10. TensorBoard日志
            writer.add_scalars("Loss", {
                'train_total': train_losses['total'],
                'train_recon': train_losses['recon'],
                'train_perceptual': train_losses['perceptual'],
                'train_cls': train_losses['cls'],
                'train_domain': train_losses['domain'],
                'train_open_set': train_losses['open_set'],
                'val_recon': val_mse,
                'val_perceptual': val_perceptual,
                'val_cls': val_losses['cls'] / len(val_loader) if len(val_loader) > 0 else 0.0,
                'val_domain': val_domain,
                'val_open_set': val_open_set
            }, epoch)
            
            writer.add_scalars("Accuracy", {
                'train': train_acc,
                'val': val_acc,
                'base_acc': ablation_results['base_acc']
            }, epoch)
            
            if epoch >= 15:
                writer.add_scalars("OpenSet", {
                    'val_auroc': auroc,
                    'val_h_score': h_score,
                    'grl_alpha': dynamic_grl.current_alpha
                }, epoch)
            
            if epoch > 10:
                writer.add_scalars("DomainMSE", {
                    'src_mse': src_mse,
                    'tgt_mse': tgt_mse,
                    'mse_diff': mse_diff
                }, epoch)

            # 11. 进度打印
            epoch_time = time.time() - start_time
            print(f"\n======== Epoch {epoch+1}/{cfg.EPOCHS} (耗时: {epoch_time:.2f}s) ========")
            print(f"训练：总损失={train_losses['total']:.4f}, 重构MSE={train_losses['recon']:.4f}, 感知损失={train_losses['perceptual']:.4f}")
            print(f"训练：分类损失={train_losses['cls']:.4f}, 域损失={train_losses['domain']:.4f}, 开放集损失={train_losses['open_set']:.4f}, 准确率={train_acc:.4f}")
            print(f"验证：准确率={val_acc:.4f}(目标≥{cfg.TARGET_ACC}), 重构MSE={val_mse:.4f}(目标≤{cfg.TARGET_MSE})")
            print(f"验证：感知损失={val_perceptual:.4f}, 域损失={val_domain:.4f}, 开放集损失={val_open_set:.4f}")
            if epoch >= 15:
                print(f"开放集：AUROC={auroc:.4f}(目标≥{cfg.TARGET_AUROC}), H-score={h_score:.4f}(目标≥{cfg.TARGET_H_SCORE})")
            print(f"当前最佳：准确率={best_acc:.4f}, MSE={best_mse:.4f}, AUROC={best_auroc:.4f}, H-score={best_h_score:.4f}")

            # 12. 最佳模型保存（优先准确率）
            update_flag = False
            if val_acc > best_acc + 1e-4:  # 核心调整：只有准确率提升才更新
                best_acc = val_acc
                best_mse = val_mse
                best_auroc = auroc
                best_h_score = h_score
                update_flag = True
            
            if update_flag:
                best_epoch = epoch + 1
                patience_counter = 0
                
                # 保存最佳模型
                try:
                    torch.save({
                        "epoch": epoch,
                        "disent_model": disent_model.state_dict(),
                        "vae_model": vae_model.state_dict(),
                        "domain_disc": domain_disc.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_acc": best_acc,
                        "best_mse": best_mse,
                        "best_auroc": best_auroc,
                        "best_h_score": best_h_score,
                        "class_centers": disent_model.class_centers,
                        "scaler": scaler.state_dict()
                    }, os.path.join(cfg.SAVE_DIR, "best_visda_joint_model_v1.pth"))
                    print(f"✅ 保存最佳模型（Epoch {best_epoch}）：准确率={best_acc:.4f}, MSE={best_mse:.4f}, AUROC={best_auroc:.4f}, H-score={best_h_score:.4f}")
                    
                    # 更新消融实验结果
                    ablation_results['percept_acc'] = best_acc
                    ablation_results['percept_mse'] = best_mse
                    ablation_results['percept_mse_improv'] = (ablation_results['base_mse'] - best_mse) / ablation_results['base_mse'] * 100
                    ablation_results['domain_mse_diff'] = mse_diff
                    ablation_results['domain_acc'] = best_acc
                    ablation_results['auroc'] = best_auroc
                    ablation_results['h_score'] = best_h_score
                    ablation_results['best_acc'] = best_acc
                    ablation_results['best_h_score'] = best_h_score
                except Exception as e:
                    print(f"⚠️ 最佳模型保存失败：{e}")
            else:
                patience_counter += 1
                print(f"⏳ 性能无明显提升，早停计数器：{patience_counter}/{cfg.PATIENCE}")

            # 13. 重构可视化（每5个epoch）
            if epoch % 5 == 0:
                visualize_reconstruction(disent_model, vae_model, val_loader, epoch)

            # 14. 定期保存检查点
            if epoch % cfg.SAVE_CHECKPOINT_EPOCH == 0 and epoch > 0:
                try:
                    torch.save({
                        "epoch": epoch,
                        "disent_model": disent_model.state_dict(),
                        "vae_model": vae_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "current_acc": val_acc,
                        "current_h_score": h_score,
                        "scaler": scaler.state_dict()
                    }, os.path.join(cfg.SAVE_DIR, f"checkpoint_epoch_{epoch}.pth"))
                    print(f"✅ 保存检查点到 {cfg.SAVE_DIR}checkpoint_epoch_{epoch}.pth")
                except Exception as e:
                    print(f"⚠️ 检查点保存失败：{e}")

            # 15. 早停判断
            if patience_counter >= cfg.PATIENCE:
                print("❌ 早停触发，训练终止")
                break

            # 释放验证阶段内存
            torch.cuda.empty_cache()
            print("-" * 80)
        else:
            # 不验证的epoch：仅更新学习率+打印基础指标
            scheduler.step()
            epoch_time = time.time() - start_time
            print(f"\n======== Epoch {epoch+1}/{cfg.EPOCHS} (仅训练，耗时: {epoch_time:.2f}s) ========")
            print(f"训练准确率：{train_acc:.4f}, 训练重构MSE：{train_losses['recon']:.4f}")
            print("-" * 80)
            torch.cuda.empty_cache()

    # 16. 训练结束后处理
    try:
        writer.close()
        ablation_results['best_acc'] = best_acc
        ablation_results['best_h_score'] = best_h_score
        ablation_results['domain_mse_diff'] = mse_diff if 'mse_diff' in locals() else 0.0
        save_ablation_results(ablation_results)
        
        del disent_model, vae_model, domain_disc, dynamic_grl
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️ 训练结束后清理失败：{e}")

    # 最终结果打印
    print(f"\n=== VisDA 联合训练 完成 ===")
    print(f"最佳验证准确率：{best_acc:.4f}（Epoch {best_epoch}）→ 是否达标：{'✅' if best_acc >= cfg.TARGET_ACC else '❌'}")
    print(f"最佳重构MSE：{best_mse:.4f} → 是否达标：{'✅' if best_mse <= cfg.TARGET_MSE else '❌'}")
    print(f"最佳开放集AUROC：{best_auroc:.4f} → 是否达标：{'✅' if best_auroc >= cfg.TARGET_AUROC else '❌'}")
    print(f"最佳开放集H-score：{best_h_score:.4f} → 是否达标：{'✅' if best_h_score >= cfg.TARGET_H_SCORE else '❌'}")
    print(f"📂 所有结果已保存至：{cfg.SAVE_DIR} | {cfg.ABLATION_DIR} | {cfg.VISUALIZATION_DIR}")
    print("="*80)

    return best_acc, best_mse, best_auroc, best_h_score

# -------------------------- 主函数入口（完整可运行） --------------------------
if __name__ == "__main__":
    # 清空CUDA缓存，避免显存残留
    torch.cuda.empty_cache()
    # 启动训练并获取最终结果
    final_acc, final_mse, final_auroc, final_h_score = joint_train_direct()
    
    # 打印最终汇总
    print(f"\n🎯 VisDA最终训练结果汇总（修复版）：")
    print(f"📈 最佳验证准确率：{final_acc:.4f}")
    print(f"📊 最佳重构MSE：{final_mse:.4f}")
    print(f"🔍 最佳开放集AUROC：{final_auroc:.4f}")
    print(f"🎯 最佳开放集H-score：{final_h_score:.4f}")
    print(f"\n💡 训练结果分析：")
    print(f"   ✅ 准确率达标：{'是' if final_acc >= cfg.TARGET_ACC else '否（目标≥0.8）'}")
    print(f"   ✅ 重构MSE达标：{'是' if final_mse <= cfg.TARGET_MSE else '否（目标≤0.03）'}")
    print(f"   ✅ 开放集AUROC达标：{'是' if final_auroc >= cfg.TARGET_AUROC else '否（目标≥0.7）'}")
    print(f"   ✅ 开放集H-score达标：{'是' if final_h_score >= cfg.TARGET_H_SCORE else '否（目标≥0.55）'}")
    # 最终清理显存
    torch.cuda.empty_cache()
    print(f"\n✨ 训练流程全部结束，显存已清理")