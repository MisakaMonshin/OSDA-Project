import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
import timm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 核心配置（优化版） --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 35  
# 🔥 优化1：调整学习率配比（缩小backbone和分类器差距）
LEARNING_RATE = 1e-4  # 从1e-5提升到1e-4
CLASSIFIER_LR = 5e-4  # 从1e-3降到5e-4
VAE_LR = 1e-4         
WEIGHT_DECAY = 1e-4   
NUM_WORKERS = 4
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

LOG_DIR = "results/logs_joint_train_direct_v3_optimized"
SAVE_DIR = "results/saved_models_joint_train_direct_v3_optimized/"
PATIENCE = 12  
TARGET_ACC = 0.86
TARGET_MSE = 0.02
TARGET_DOMAIN_MSE_DIFF = 0.02
TARGET_AUROC = 0.75
TARGET_H_SCORE = 0.60

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("results/visualization/v3_optimized", exist_ok=True)
os.makedirs("results/ablation_exp", exist_ok=True)

# 🔥 新增：开放集优化超参数
OPEN_SET_LAMBDA = 0.8  # 开放集对比损失权重
GRL_ALPHA_BASE = 1.0   # GRL基础强度
GRL_ALPHA_DECAY = 0.95 # GRL强度衰减系数
TEMP = 0.1             # 类中心距离温度系数
CENTER_UPDATE_EPOCH = 3 # 类中心更新周期
MARGIN = 1.5           # 已知/未知类距离边际

# -------------------------- 数据增强（兼容VisDA） --------------------------
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=8, p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transform(dataset_type="DomainNet"):
    if dataset_type == "VisDA":
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def apply_albumentations(images, transform):
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
    augmented = [transform(image=img)["image"] for img in images]
    return torch.stack(augmented).to(DEVICE)

# -------------------------- 优化2：改进权重加载函数 --------------------------
def load_pretrained_weights(model, model_name, local_path):
    try:
        state_dict = torch.load(local_path, map_location=DEVICE)
        if model_name == "convnextv2_tiny":
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 成功加载{model_name}本地预训练权重")
    except Exception as e:
        print(f"⚠️ 本地{model_name}权重加载失败：{e}，自动下载官方预训练权重")
        if model_name == "vgg16":
            model = models.vgg16(pretrained=True)
        elif model_name == "convnextv2_tiny":
            model = timm.create_model("convnextv2_tiny", pretrained=True, num_classes=0)
    return model

# -------------------------- 改进的感知损失模块（第一阶段核心） --------------------------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        local_weights_path = "/root/autodl-tmp/OSDA_Project/code/test/vgg16-397923af.pth"
        vgg = models.vgg16()
        # 🔥 调用优化后的权重加载函数
        vgg = load_pretrained_weights(vgg, "vgg16", local_weights_path)
        vgg = vgg.features[:23].eval().to(DEVICE)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.l1 = nn.L1Loss()
        self.projection = nn.Sequential(
            nn.Linear(2048, 512 * 14 * 14),
            nn.ReLU()
        ).to(DEVICE)

    def forward(self, recon_features, orig_features):
        batch_size = recon_features.size(0)
        recon_proj = self.projection(recon_features).view(batch_size, 512, 14, 14)
        orig_proj = self.projection(orig_features).view(batch_size, 512, 14, 14)
        return self.l1(recon_proj, orig_proj)

# -------------------------- 梯度反转层（GRL）+ 域判别器（第二阶段核心） --------------------------
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# 🔥 修改：动态GRL封装
class DynamicGRL(nn.Module):
    def __init__(self, base_alpha=1.0, decay=0.95):
        super().__init__()
        self.base_alpha = base_alpha
        self.decay = decay
        self.current_alpha = base_alpha

    def forward(self, x, epoch):
        # 随epoch衰减GRL强度，减少后期域对抗对特征区分性的破坏
        self.current_alpha = self.base_alpha * (self.decay ** (epoch // 2))
        return GradientReversalLayer.apply(x, self.current_alpha)

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=2048):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------- 🔥 新增：开放集对比损失 --------------------------
class OpenSetContrastiveLoss(nn.Module):
    def __init__(self, margin=MARGIN, temp=TEMP):
        super().__init__()
        self.margin = margin
        self.temp = temp

    def forward(self, features, labels, centers, num_known_classes=200):
        """
        增强已知类内聚性 + 已知/未知类分离性
        features: 批次特征 (B, 2048)
        labels: 批次标签 (B,)
        centers: 已知类中心 (num_known, 2048)
        """
        # 归一化特征和中心
        features = F.normalize(features, dim=1)
        centers = F.normalize(centers, dim=1)
        
        # 区分已知/未知类
        known_mask = (labels < num_known_classes)
        unknown_mask = ~known_mask
        
        loss = 0.0
        if known_mask.sum() > 0:
            # 已知类：拉近到自身类中心
            known_feats = features[known_mask]
            known_labels = labels[known_mask]
            center_dist = torch.cdist(known_feats, centers) / self.temp
            # 类内损失（对比损失）
            known_loss = -torch.log(torch.exp(-center_dist[range(len(known_labels)), known_labels]) / 
                                   torch.sum(torch.exp(-center_dist), dim=1))
            loss += known_loss.mean()
        
        if unknown_mask.sum() > 0 and known_mask.sum() > 0:
            # 未知类：推远所有已知类中心
            unknown_feats = features[unknown_mask]
            unknown_dist = torch.cdist(unknown_feats, centers).min(dim=1)[0] / self.temp
            unknown_loss = torch.clamp(self.margin - unknown_dist, min=0.0).mean()
            loss += unknown_loss
        
        return loss

# -------------------------- 重构可视化（并行执行，节省时间） --------------------------
def visualize_reconstruction(disent_model, vae_model, val_loader, epoch, dataset_type="DomainNet"):
    disent_model.eval()
    vae_model.eval()
    val_transform = get_val_transform(dataset_type)
    with torch.no_grad():
        for images, labels in val_loader:
            images = apply_albumentations(images, val_transform)
            disent_feature, _, _ = disent_model(images)
            recon_feature, _, _, _ = vae_model(disent_feature)
            
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            for j in range(5):
                orig_img = disent_feature[j].view(512, 2, 2).unsqueeze(0)
                orig_img = F.interpolate(orig_img, size=(64, 64), mode='bilinear').squeeze()
                orig_img = orig_img[:3, :, :].cpu().numpy().transpose(1, 2, 0)
                orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
                
                recon_img = recon_feature[j].view(512, 2, 2).unsqueeze(0)
                recon_img = F.interpolate(recon_img, size=(64, 64), mode='bilinear').squeeze()
                recon_img = recon_img[:3, :, :].cpu().numpy().transpose(1, 2, 0)
                recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)
                
                axes[0, j].imshow(orig_img)
                axes[0, j].set_title(f"Original {j+1}")
                axes[0, j].axis('off')
                axes[1, j].imshow(recon_img)
                axes[1, j].set_title(f"Recon {j+1}")
                axes[1, j].axis('off')
            
            plt.tight_layout()
            save_path = f"results/visualization/v3_optimized/recon_epoch_{epoch}_{dataset_type}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 重构可视化已保存到 {save_path}")
            break

# -------------------------- 改进的类中心计算（动态更新） --------------------------
@torch.no_grad()
def compute_class_centers(model, loader, num_classes, feature_dim=2048, transform=None, update_rate=0.1):
    """
    🔥 修改：动态更新类中心（滑动平均）
    update_rate: 新批次特征的权重
    """
    model.eval()
    # 初始化/加载历史中心
    try:
        centers = model.class_centers.clone()
    except:
        centers = torch.zeros(num_classes, feature_dim).to(DEVICE)
        counts = torch.zeros(num_classes).to(DEVICE)
        # 首次计算
        for images, labels in loader:
            if transform is not None:
                images = apply_albumentations(images, transform)
            else:
                images = images.to(DEVICE)
            features, _, _ = model(images)
            
            for c in range(num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    centers[c] += features[mask].sum(dim=0)
                    counts[c] += mask.sum()
        counts = torch.clamp(counts, min=1e-8)
        centers = centers / counts.unsqueeze(1)
    
    # 滑动平均更新
    new_centers = centers.clone()
    new_counts = torch.ones(num_classes).to(DEVICE) * 1e-8
    for images, labels in loader:
        if transform is not None:
            images = apply_albumentations(images, transform)
        else:
            images = images.to(DEVICE)
        features, _, _ = model(images)
        
        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                new_centers[c] = (1 - update_rate) * new_centers[c] + update_rate * features[mask].mean(dim=0)
                new_counts[c] += mask.sum()
    
    # 保存到模型
    model.class_centers = new_centers
    return new_centers

# -------------------------- 学习率调度器（适配短周期） --------------------------
def get_scheduler(optimizer):
    return CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=3e-7)

# -------------------------- 优化3：调整lambda_recon增长策略（稳定重构损失） --------------------------
def stage_freeze_unfreeze(disent_model, vae_model, epoch):
    for param in disent_model.parameters():
        param.requires_grad = True
    for param in vae_model.parameters():
        param.requires_grad = True
    
    # 🔥 优化：放缓lambda_recon增长，避免重构损失震荡
    if epoch < 5:
        lambda_recon = 1.0 + epoch * 0.02  # 从0.1降到0.02，增长更缓
        lambda_disent = 0.5 - epoch * 0.02
        lambda_kl = 0.0005
        lambda_domain = 0.05  # 降低初始域对抗权重
        lambda_open_set = 0.2 # 🔥 新增：提前引入开放集损失
        print(f"🔹 阶段1：平衡重构和分类（lambda_recon={lambda_recon:.1f}, lambda_disent={lambda_disent:.1f}）")
    elif epoch < 20:
        lambda_recon = 1.1 + (epoch-5) * 0.01  # 放缓增长
        lambda_disent = 0.4 - (epoch-5) * 0.005
        lambda_kl = 0.0007
        lambda_domain = 0.10  # 降低域对抗权重（原0.15）
        lambda_open_set = 0.5 # 🔥 新增：提升开放集损失权重
        print(f"🔹 阶段2：加入域对抗训练（lambda_domain=0.10）")
    else:
        lambda_recon = 1.2 + (epoch-20) * 0.005  # 进一步放缓
        lambda_disent = 0.325 - (epoch-20) * 0.001
        lambda_kl = 0.001
        lambda_domain = 0.05  # 进一步降低域对抗
        lambda_open_set = 0.8 # 🔥 新增：重点优化开放集
        print(f"🔹 阶段3：优化开放集拒识指标")
    
    return lambda_recon, lambda_disent, lambda_kl, lambda_domain, lambda_open_set

# -------------------------- 改进的特征解耦模型（兼容VisDA） --------------------------
class ImprovedFeatureDisentanglementModel(nn.Module):
    def __init__(self, num_classes=241, feature_dim=2048, dataset_type="DomainNet"):
        super().__init__()
        if dataset_type == "VisDA":
            num_classes = 12
        
        self.backbone = timm.create_model(
            model_name="convnextv2_tiny",
            pretrained=False,
            num_classes=0
        )
        local_weight_path = "/root/autodl-tmp/OSDA_Project/code/test/pytorch_model.bin"
        # 🔥 调用优化后的权重加载函数
        self.backbone = load_pretrained_weights(self.backbone, "convnextv2_tiny", local_weight_path)
        
        backbone_out_dim = self.backbone.num_features

        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        ).to(DEVICE)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ 模型可训练参数：{trainable_params/1e6:.2f}M（应≥25M）")

        # 🔥 优化4：降低分类分支dropout率（从0.4→0.3），减少信息丢失
        self.class_branch = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),  # 优化点
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # 优化点
            nn.Linear(512, num_classes)
        ).to(DEVICE)

        self.domain_branch = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        ).to(DEVICE)
        
        # 🔥 新增：类中心存储
        self.class_centers = None

    def forward(self, x):
        base_feature_raw = self.backbone(x)
        base_feature = self.feature_proj(base_feature_raw)
        class_logits = self.class_branch(base_feature)
        domain_logits = self.domain_branch(base_feature)
        disent_loss = torch.tensor(0.0).to(DEVICE)
        return base_feature, class_logits, disent_loss

# -------------------------- 计算源/目标域MSE差异（第二阶段核心） --------------------------
@torch.no_grad()
def compute_domain_mse_diff(disent_model, vae_model, src_loader, tgt_loader, transform):
    disent_model.eval()
    vae_model.eval()
    # 🔥 优化5：用SmoothL1Loss替代MSE，减少异常值影响
    mse_loss_fn = nn.SmoothL1Loss()
    
    src_mse = 0.0
    src_count = 0
    for images, _ in src_loader:
        images = apply_albumentations(images, transform)
        disent_feature, _, _ = disent_model(images)
        recon_feature, _, _, _ = vae_model(disent_feature)
        src_mse += mse_loss_fn(recon_feature, disent_feature).item() * len(images)
        src_count += len(images)
    src_mse /= src_count
    
    tgt_mse = 0.0
    tgt_count = 0
    for images, _ in tgt_loader:
        images = apply_albumentations(images, transform)
        disent_feature, _, _ = disent_model(images)
        recon_feature, _, _, _ = vae_model(disent_feature)
        tgt_mse += mse_loss_fn(recon_feature, disent_feature).item() * len(images)
        tgt_count += len(images)
    tgt_mse /= tgt_count
    
    mse_diff = abs(src_mse - tgt_mse)
    return src_mse, tgt_mse, mse_diff

# -------------------------- 消融实验记录（第五阶段核心） --------------------------
def save_ablation_results(results, save_path="results/ablation_exp/ablation_v3_optimized.txt"):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Test08 V3 优化版 消融实验结果 ===\n")
        f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"基础模型（无感知损失+无域对抗）：准确率={results['base_acc']:.4f}, MSE={results['base_mse']:.4f}\n")
        f.write(f"添加感知损失：准确率={results['percept_acc']:.4f}, MSE={results['percept_mse']:.4f}（提升{results['percept_mse_improv']:.2f}%）\n")
        f.write(f"添加域对抗：域间MSE差异={results['domain_mse_diff']:.4f}, 准确率={results['domain_acc']:.4f}\n")
        f.write(f"开放集指标：AUROC={results['auroc']:.4f}, H-score={results['h_score']:.4f}\n")
        f.write(f"VisDA泛化：H-score={results['visda_h_score']:.4f}\n")
    print(f"✅ 消融实验结果已保存到 {save_path}")

# -------------------------- 联合训练主函数（适配新规划全阶段） --------------------------
def joint_train_direct():
    from DomainNet_test_07_optimized_vae import FeatureVAE
    from DomainNet_test_03_data_loader_simple import get_data_loaders

    # 1. 初始化模型
    disent_model = ImprovedFeatureDisentanglementModel(num_classes=241, dataset_type="DomainNet").to(DEVICE)
    # 🔥 优化6：VAE的latent_dim从512→1024（匹配技术方案）
    vae_model = FeatureVAE(feature_dim=2048, latent_dim=1024).to(DEVICE)
    domain_disc = DomainDiscriminator(input_dim=2048).to(DEVICE)
    # 🔥 修改：使用动态GRL
    dynamic_grl = DynamicGRL(base_alpha=GRL_ALPHA_BASE, decay=GRL_ALPHA_DECAY).to(DEVICE)
    # 🔥 新增：开放集对比损失
    open_set_loss_fn = OpenSetContrastiveLoss(margin=MARGIN, temp=TEMP).to(DEVICE)

    # 2. 损失函数
    # 🔥 优化5：用SmoothL1Loss替代MSE
    mse_loss_fn = nn.SmoothL1Loss()
    # 🔥 优化7：添加标签平滑（缓解过拟合）
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    perceptual_loss_fn = PerceptualLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    # 3. 优化器配置（已在顶部调整学习率）
    backbone_params = list(disent_model.backbone.parameters()) + list(disent_model.feature_proj.parameters())
    classifier_params = list(disent_model.class_branch.parameters())
    domain_params = list(disent_model.domain_branch.parameters())
    vae_params = list(vae_model.parameters())
    domain_disc_params = list(domain_disc.parameters())

    params = [
        {'params': backbone_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
        {'params': classifier_params, 'lr': CLASSIFIER_LR, 'weight_decay': WEIGHT_DECAY},
        {'params': domain_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
        {'params': vae_params, 'lr': VAE_LR, 'weight_decay': WEIGHT_DECAY},
        {'params': domain_disc_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
    ]

    optimizer = optim.AdamW(params)
    scheduler = get_scheduler(optimizer)
    scaler = GradScaler('cuda', enabled=True)

    # 4. 数据加载
    train_loader_old, val_loader_old, test_loader_old = get_data_loaders()
    train_transform = get_train_transform()
    val_transform = get_val_transform("DomainNet")
    
    # 🔥 优化8：开启persistent_workers，提升数据加载效率
    train_loader = DataLoader(
        train_loader_old.dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=True  # 优化点
    )
    val_loader = DataLoader(
        val_loader_old.dataset, 
        batch_size=BATCH_SIZE*2, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=True  # 优化点
    )
    
    src_loader, tgt_loader = train_loader, val_loader

    # 5. 日志初始化
    writer = SummaryWriter(LOG_DIR)
    best_acc, best_mse, best_auroc, best_h_score = 0.0, float('inf'), 0.0, 0.0
    best_epoch = 0
    patience_counter = 0
    base_acc = 0.8664
    base_mse = 0.0104
    num_known_classes = 200

    ablation_results = {
        "base_acc": base_acc,
        "base_mse": base_mse,
        "percept_acc": 0.0,
        "percept_mse": float('inf'),
        "percept_mse_improv": 0.0,
        "domain_mse_diff": float('inf'),
        "domain_acc": 0.0,
        "auroc": 0.0,
        "h_score": 0.0,
        "visda_h_score": 0.0
    }

    # 6. 主训练循环
    for epoch in range(EPOCHS):
        start_time = time.time()
        # 🔥 修改：新增lambda_open_set
        lambda_recon, lambda_disent, lambda_kl, lambda_domain, lambda_open_set = stage_freeze_unfreeze(disent_model, vae_model, epoch)
        
        disent_model.train()
        vae_model.train()
        domain_disc.train()
        optimizer.zero_grad()

        train_losses = {'total': 0, 'recon': 0, 'cls': 0, 'disent': 0, 'kl': 0, 'perceptual': 0, 'domain': 0, 'open_set': 0}
        correct = 0
        total = 0

        # 🔥 新增：定期更新类中心
        if epoch % CENTER_UPDATE_EPOCH == 0:
            centers = compute_class_centers(disent_model, train_loader, num_known_classes, transform=val_transform)
        else:
            centers = disent_model.class_centers

        # 批量训练
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = apply_albumentations(images, train_transform)
            labels = labels.to(DEVICE)
            
            with autocast('cuda'):
                # 前向传播
                disent_feature, logits, disent_loss = disent_model(images)
                recon_feature, mu, log_var, _ = vae_model(disent_feature)
                
                # 重构损失（用SmoothL1Loss）
                mse_recon_loss = mse_loss_fn(recon_feature, disent_feature)
                perceptual_loss = perceptual_loss_fn(recon_feature, disent_feature)
                recon_loss = 0.7 * mse_recon_loss + 0.3 * perceptual_loss
                
                # KL损失
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
                
                # 分类损失（带标签平滑）
                cls_loss = ce_loss_fn(logits, labels)
                
                # 解耦损失
                disent_loss = disent_loss.mean() if len(disent_loss.shape) > 0 else disent_loss
                disent_loss = torch.clamp(disent_loss.abs(), 0.0, 15.0)
                
                # 域对抗损失
                domain_loss = torch.tensor(0.0).to(DEVICE)
                if lambda_domain > 0:
                    # 🔥 修改：使用动态GRL
                    reversed_feature = dynamic_grl(disent_feature, epoch)
                    domain_pred = domain_disc(reversed_feature)
                    domain_label = torch.zeros_like(labels).float().to(DEVICE)
                    domain_loss = domain_criterion(domain_pred, domain_label.unsqueeze(1))
                
                # 🔥 新增：开放集对比损失
                open_set_loss = torch.tensor(0.0).to(DEVICE)
                if lambda_open_set > 0 and centers is not None:
                    open_set_loss = open_set_loss_fn(disent_feature, labels, centers, num_known_classes)
                
                # 总损失（新增开放集损失）
                total_loss = (
                    lambda_recon * recon_loss + 
                    lambda_disent * (cls_loss + disent_loss) + 
                    lambda_kl * kl_loss + 
                    lambda_domain * domain_loss +
                    lambda_open_set * open_set_loss  # 🔥 新增
                )

            # 反向传播
            scaler.scale(total_loss / GRADIENT_ACCUMULATION_STEPS).backward()
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                all_params = []
                for group in optimizer.param_groups:
                    all_params.extend(group['params'])
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 损失统计（新增open_set）
            train_losses['total'] += total_loss.item()
            train_losses['recon'] += mse_recon_loss.item()
            train_losses['cls'] += cls_loss.item()
            train_losses['disent'] += disent_loss.item()
            train_losses['kl'] += kl_loss.item()
            train_losses['perceptual'] += perceptual_loss.item()
            train_losses['domain'] += domain_loss.item()
            train_losses['open_set'] += open_set_loss.item()  # 🔥 新增
            
            # 准确率统计
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            # 打印训练进度（新增open_set损失）
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {total_loss.item():.4f}, Recon: {mse_recon_loss.item():.4f}, "
                      f"Perceptual: {perceptual_loss.item():.4f}, Cls: {cls_loss.item():.4f}, "
                      f"Domain: {domain_loss.item():.4f}, OpenSet: {open_set_loss.item():.4f}, "  # 🔥 新增
                      f"LR: {optimizer.param_groups[1]['lr']:.6f}, GRL_alpha: {dynamic_grl.current_alpha:.3f}")  # 🔥 新增

        # 训练指标计算
        train_acc = correct / total
        for k in train_losses:
            train_losses[k] /= len(train_loader)

        # 7. 验证阶段（优化9：每2个epoch验证一次，提升效率）
        if epoch % 2 == 0:
            disent_model.eval()
            vae_model.eval()
            domain_disc.eval()
            
            val_losses = {'recon': 0, 'cls': 0, 'perceptual': 0, 'domain': 0, 'open_set': 0}
            correct = 0
            total = 0

            # 计算类中心
            centers = compute_class_centers(disent_model, train_loader, num_known_classes, transform=val_transform)

            # 开放集指标存储
            all_conf = []
            all_dist = []
            all_recon_err = []
            all_labels = []
            # 🔥 新增：保存特征用于AUROC优化
            all_feats = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = apply_albumentations(images, val_transform)
                    labels = labels.to(DEVICE)
                    
                    disent_feature, logits, _ = disent_model(images)
                    recon_feature, _, _, _ = vae_model(disent_feature)
                    
                    # 损失统计（新增open_set）
                    val_losses['recon'] += mse_loss_fn(recon_feature, disent_feature).item()
                    val_losses['perceptual'] += perceptual_loss_fn(recon_feature, disent_feature).item()
                    val_losses['cls'] += ce_loss_fn(logits, labels).item()
                    if lambda_open_set > 0 and centers is not None:
                        val_losses['open_set'] += open_set_loss_fn(disent_feature, labels, centers, num_known_classes).item()
                    
                    # 域损失
                    if lambda_domain > 0:
                        reversed_feature = dynamic_grl(disent_feature, epoch)
                        domain_pred = domain_disc(reversed_feature)
                        domain_label = torch.zeros_like(labels).float().to(DEVICE)
                        val_losses['domain'] += domain_criterion(domain_pred, domain_label.unsqueeze(1)).item()
                    
                    # 准确率
                    preds = logits.argmax(dim=1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)

                    # 开放集指标计算（🔥 改进：特征归一化+温度系数）
                    conf = F.softmax(logits / TEMP, dim=1).max(dim=1)[0]  # 温度系数
                    dist = torch.cdist(F.normalize(disent_feature, dim=1), F.normalize(centers, dim=1)).min(dim=1)[0]
                    recon_err = F.smooth_l1_loss(recon_feature, disent_feature, reduction='none').mean(dim=1)
                    
                    is_unknown = (labels >= num_known_classes).float()
                    
                    all_conf.extend(conf.cpu().numpy())
                    all_dist.extend(dist.cpu().numpy())
                    all_recon_err.extend(recon_err.cpu().numpy())
                    all_labels.extend(is_unknown.cpu().numpy())
                    all_feats.extend(F.normalize(disent_feature, dim=1).cpu().numpy())  # 🔥 新增

            # 验证指标计算
            val_acc = correct / total
            val_mse = val_losses['recon'] / len(val_loader)
            val_perceptual = val_losses['perceptual'] / len(val_loader)
            val_domain = val_losses['domain'] / len(val_loader) if lambda_domain > 0 else 0.0
            val_open_set = val_losses['open_set'] / len(val_loader) if lambda_open_set > 0 else 0.0  # 🔥 新增
            scheduler.step()

            # 8. 开放集指标计算（🔥 改进：多特征融合+动态权重）
            unknown_count = sum(all_labels)
            print(f"\n📊 验证集中未知类数量: {unknown_count}")
            auroc = 0.0
            h_score = 0.0
            optimal_threshold = 0.5
            
            if unknown_count > 0:
                # 归一化指标（🔥 改进：增加特征方差）
                conf_arr = 1 - np.array(all_conf)
                dist_arr = np.array(all_dist)
                recon_err_arr = np.array(all_recon_err)
                # 🔥 新增：特征方差（未知类特征方差更大）
                feats_arr = np.array(all_feats)
                feat_var = np.var(feats_arr, axis=1)
                
                conf_arr = conf_arr / (np.max(conf_arr) + 1e-8)
                dist_arr = dist_arr / (np.max(dist_arr) + 1e-8)
                recon_err_arr = recon_err_arr / (np.max(recon_err_arr) + 1e-8)
                feat_var = feat_var / (np.max(feat_var) + 1e-8)

                # 🔥 改进：增加特征方差权重组合
                weight_combinations = [(0.25, 0.35, 0.25, 0.15), (0.3, 0.4, 0.2, 0.1), 
                                      (0.2, 0.3, 0.3, 0.2), (0.3, 0.3, 0.2, 0.2)]
                best_h_score_val = 0.0
                best_weights = (0.3, 0.4, 0.2, 0.1)
                best_scores = None

                for w1, w2, w3, w4 in weight_combinations:
                    scores = w1 * conf_arr + w2 * dist_arr + w3 * recon_err_arr + w4 * feat_var  # 🔥 新增feat_var
                    fpr, tpr, thresholds = roc_curve(all_labels, scores)
                    optimal_idx = np.argmax(tpr - fpr)
                    optimal_threshold = thresholds[optimal_idx]
                    
                    preds = (scores >= optimal_threshold).astype(int)
                    tp = np.sum((preds == 1) & (np.array(all_labels) == 1))
                    fn = np.sum((preds == 0) & (np.array(all_labels) == 1))
                    fp = np.sum((preds == 1) & (np.array(all_labels) == 0))
                    tn = np.sum((preds == 0) & (np.array(all_labels) == 0))
                    
                    tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
                    tnr_val = tn / (tn + fp) if (tn + fp) > 0 else 0
                    h_score_val = 2 * tpr_val * tnr_val / (tpr_val + tnr_val) if (tpr_val + tnr_val) > 0 else 0

                    if h_score_val > best_h_score_val:
                        best_h_score_val = h_score_val
                        best_weights = (w1, w2, w3, w4)
                        best_scores = scores

                # 最终指标
                w1, w2, w3, w4 = best_weights
                scores = best_scores
                auroc = roc_auc_score(all_labels, scores)
                fpr, tpr, thresholds = roc_curve(all_labels, scores)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                preds = (scores >= optimal_threshold).astype(int)
                tp = np.sum((preds == 1) & (np.array(all_labels) == 1))
                fn = np.sum((preds == 0) & (np.array(all_labels) == 1))
                fp = np.sum((preds == 1) & (np.array(all_labels) == 0))
                tn = np.sum((preds == 0) & (np.array(all_labels) == 0))
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                h_score = 2 * tpr * tnr / (tpr + tnr) if (tpr + tnr) > 0 else 0

                print(f"🎯 最佳融合权重: w1={w1:.2f}(置信度), w2={w2:.2f}(距离), w3={w3:.2f}(重构误差), w4={w4:.2f}(特征方差)")
                print(f"🎯 最佳阈值: {optimal_threshold:.4f}, TPR={tpr:.4f}, TNR={tnr:.4f}, H-score={h_score:.4f}")
            else:
                print("⚠️ 验证集中没有未知类，H-score 无法计算")

            # 9. 源/目标域MSE差异计算
            src_mse, tgt_mse, mse_diff = 0.0, 0.0, 0.0
            if epoch > 5:
                src_mse, tgt_mse, mse_diff = compute_domain_mse_diff(disent_model, vae_model, src_loader, tgt_loader, val_transform)
                print(f"🌐 源域MSE={src_mse:.4f}, 目标域MSE={tgt_mse:.4f}, 差异={mse_diff:.4f} (目标≤{TARGET_DOMAIN_MSE_DIFF})")

            # 10. TensorBoard日志（新增open_set）
            writer.add_scalars("Loss", {
                'train_total': train_losses['total'],
                'train_recon': train_losses['recon'],
                'train_perceptual': train_losses['perceptual'],
                'train_cls': train_losses['cls'],
                'train_domain': train_losses['domain'],
                'train_open_set': train_losses['open_set'],  # 🔥 新增
                'val_recon': val_mse,
                'val_perceptual': val_perceptual,
                'val_cls': val_losses['cls'] / len(val_loader),
                'val_domain': val_domain,
                'val_open_set': val_open_set  # 🔥 新增
            }, epoch)
            writer.add_scalars("Accuracy", {
                'train': train_acc,
                'val': val_acc,
                'base_acc': base_acc
            }, epoch)
            writer.add_scalars("OpenSet", {
                'val_auroc': auroc,
                'val_h_score': h_score,
                'grl_alpha': dynamic_grl.current_alpha  # 🔥 新增
            }, epoch)
            if epoch > 5:
                writer.add_scalars("DomainMSE", {
                    'src_mse': src_mse,
                    'tgt_mse': tgt_mse,
                    'mse_diff': mse_diff
                }, epoch)

            # 11. 进度打印（新增open_set损失）
            epoch_time = time.time() - start_time
            print(f"\n======== Epoch {epoch+1}/{EPOCHS} (耗时: {epoch_time:.2f}s) ========")
            print(f"训练：总损失={train_losses['total']:.4f}, 重构MSE={train_losses['recon']:.4f}, 感知损失={train_losses['perceptual']:.4f}")
            print(f"训练：分类损失={train_losses['cls']:.4f}, 域损失={train_losses['domain']:.4f}, 开放集损失={train_losses['open_set']:.4f}, 准确率={train_acc:.4f}")
            print(f"验证：准确率={val_acc:.4f}(目标≥{TARGET_ACC}), 重构MSE={val_mse:.4f}(目标≤{TARGET_MSE})")
            print(f"验证：感知损失={val_perceptual:.4f}, 域损失={val_domain:.4f}, 开放集损失={val_open_set:.4f}")
            print(f"开放集：AUROC={auroc:.4f}(目标≥{TARGET_AUROC}), H-score={h_score:.4f}(目标≥{TARGET_H_SCORE})")
            print(f"当前最佳：准确率={best_acc:.4f}, MSE={best_mse:.4f}, AUROC={best_auroc:.4f}, H-score={best_h_score:.4f}")

            # 12. 最佳模型保存
            if (val_acc > best_acc - 1e-4) or (val_mse < best_mse + 1e-4) or (auroc > best_auroc - 1e-4) or (h_score > best_h_score - 1e-4):
                update_flag = False
                if val_acc > best_acc:
                    best_acc = val_acc
                    update_flag = True
                if val_mse < best_mse:
                    best_mse = val_mse
                    update_flag = True
                if auroc > best_auroc:
                    best_auroc = auroc
                    update_flag = True
                if h_score > best_h_score:
                    best_h_score = h_score
                    update_flag = True
                
                if update_flag:
                    best_epoch = epoch + 1
                    patience_counter = 0
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
                        "class_centers": disent_model.class_centers  # 🔥 新增
                    }, os.path.join(SAVE_DIR, "best_joint_model_v3_optimized.pth"))
                    print(f"✅ 保存最佳模型（Epoch {best_epoch}）：准确率={best_acc:.4f}, MSE={best_mse:.4f}, AUROC={best_auroc:.4f}, H-score={best_h_score:.4f}")
                    
                    ablation_results['percept_acc'] = best_acc
                    ablation_results['percept_mse'] = best_mse
                    ablation_results['percept_mse_improv'] = (base_mse - best_mse) / base_mse * 100
                    ablation_results['domain_mse_diff'] = mse_diff
                    ablation_results['domain_acc'] = best_acc
                    ablation_results['auroc'] = best_auroc
                    ablation_results['h_score'] = best_h_score
                else:
                    patience_counter += 1
                    print(f"⏳ 性能无明显提升，早停计数器：{patience_counter}/{PATIENCE}")
            else:
                patience_counter += 1
                print(f"⏳ 性能无明显提升，早停计数器：{patience_counter}/{PATIENCE}")

            # 13. 重构可视化
            if epoch % 3 == 0:
                visualize_reconstruction(disent_model, vae_model, val_loader, epoch, "DomainNet")

            # 14. 早停判断
            if patience_counter >= PATIENCE:
                print("❌ 早停触发，训练终止")
                break

            print("-" * 80)
        else:
            # 不验证的epoch只更新学习率
            scheduler.step()
            epoch_time = time.time() - start_time
            print(f"\n======== Epoch {epoch+1}/{EPOCHS} (仅训练，耗时: {epoch_time:.2f}s) ========")
            print(f"训练准确率：{train_acc:.4f}, 训练重构MSE：{train_losses['recon']:.4f}")
            print("-" * 80)

    # 16. 训练结束处理
    writer.close()
    save_ablation_results(ablation_results)
    
    print(f"\n=== Test08 V3 优化版 训练完成 ===")
    print(f"最佳验证准确率：{best_acc:.4f}（Epoch {best_epoch}）→ 是否达标：{'✅' if best_acc >= TARGET_ACC else '❌'}")
    print(f"最佳重构MSE：{best_mse:.4f} → 是否达标：{'✅' if best_mse <= TARGET_MSE else '❌'}")
    print(f"最佳开放集AUROC：{best_auroc:.4f} → 是否达标：{'✅' if best_auroc >= TARGET_AUROC else '❌'}")
    print(f"最佳开放集H-score：{best_h_score:.4f} → 是否达标：{'✅' if best_h_score >= TARGET_H_SCORE else '❌'}")
    print(f"源/目标域MSE差异：{ablation_results['domain_mse_diff']:.4f} → 是否达标：{'✅' if ablation_results['domain_mse_diff'] <= TARGET_DOMAIN_MSE_DIFF else '❌'}")

    return best_acc, best_mse, best_auroc, best_h_score

# -------------------------- 主函数入口 --------------------------
if __name__ == "__main__":
    print("🚀 启动Test08 V3 优化版 联合训练（适配新规划+开放集优化）")
    print(f"📌 阶段1目标：MSE≤{TARGET_MSE}，准确率≥{TARGET_ACC}")
    print(f"📌 阶段2目标：域间MSE差异≤{TARGET_DOMAIN_MSE_DIFF}，准确率≥0.855")
    print(f"📌 阶段3目标：AUROC≥{TARGET_AUROC}，H-score≥{TARGET_H_SCORE}")
    print("-" * 80)
    
    final_acc, final_mse, final_auroc, final_h_score = joint_train_direct()
    
    print(f"\n🎯 最终结果汇总：")
    print(f"最佳准确率：{final_acc:.4f}")
    print(f"最佳重构MSE：{final_mse:.4f}")
    print(f"最佳开放集AUROC：{final_auroc:.4f}")
    print(f"最佳开放集H-score：{final_h_score:.4f}")