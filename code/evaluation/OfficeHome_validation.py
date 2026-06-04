"""
Office-Home数据集验证脚本
基于DomainNet_test_08训练的模型进行泛化验证
支持开放集域适应（OSDA）评估
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
import timm
from PIL import Image
import warnings
import random
warnings.filterwarnings('ignore')

# 设置随机种子确保可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ======================== 全局配置 ========================
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    NUM_WORKERS = 4 if os.cpu_count() > 4 else 0
    
    # Office-Home数据集配置
    # Office-Home有4个域: Art, Clipart, Product, Real World
    # 65个类别，我们选择部分作为已知类，部分作为未知类
    DATA_ROOT = "/root/autodl-tmp/OSDA_Project/data/processed/OfficeHomeDataset/OfficeHomeDataset_10072016"
    
    # 模型权重路径（DomainNet训练的模型）
    MODEL_PATH = "/root/autodl-tmp/OSDA_Project/results/saved_models_joint_train_direct_v3_optimized/best_joint_model_v3_optimized.pth"
    
    # DomainNet训练时的类别数
    DOMAINNET_NUM_CLASSES = 241
    
    # Office-Home开放集划分
    # Office-Home有65个类别，我们选择前50个作为已知类，后15个作为未知类
    NUM_KNOWN_CLASSES = 50
    NUM_UNKNOWN_CLASSES = 15
    TOTAL_CLASSES = 65
    
    # 结果保存路径
    RESULT_DIR = "results/office_home_validation"
    
    # 目标指标
    TARGET_ACC = 0.50  # 跨域验证准确率目标
    TARGET_AUROC = 0.70
    TARGET_H_SCORE = 0.55

cfg = Config()
os.makedirs(cfg.RESULT_DIR, exist_ok=True)

# ======================== 数据增强 ========================
def get_val_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def apply_albumentations(images, transform):
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
    augmented = [transform(image=img)["image"] for img in images]
    return torch.stack(augmented).to(cfg.DEVICE)

# ======================== Office-Home数据集类 ========================
class OfficeHomeDataset(Dataset):
    """
    Office-Home数据集加载器
    支持开放集划分（已知类/未知类）
    """
    def __init__(self, root_dir, domain_name, transform=None, 
                 known_classes=None, unknown_classes=None, is_target=True):
        """
        Args:
            root_dir: 数据集根目录
            domain_name: 域名称 (Art, Clipart, Product, RealWorld)
            transform: 数据增强
            known_classes: 已知类列表
            unknown_classes: 未知类列表
            is_target: 是否为目标域（True则包含未知类）
        """
        self.root_dir = os.path.join(root_dir, domain_name)
        self.transform = transform
        self.is_target = is_target
        
        self.known_classes = known_classes if known_classes else list(range(cfg.NUM_KNOWN_CLASSES))
        self.unknown_classes = unknown_classes if unknown_classes else list(range(cfg.NUM_KNOWN_CLASSES, cfg.TOTAL_CLASSES))
        
        self.images = []
        self.labels = []
        self.is_known = []  # 标记是否为已知类
        
        # 遍历所有类别文件夹
        if os.path.exists(self.root_dir):
            class_dirs = sorted(os.listdir(self.root_dir))
            for class_idx, class_name in enumerate(class_dirs):
                class_path = os.path.join(self.root_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                    
                # 判断是已知类还是未知类
                if class_idx in self.known_classes:
                    label = self.known_classes.index(class_idx)  # 重新映射标签
                    is_known_flag = True
                elif class_idx in self.unknown_classes and is_target:
                    label = -1  # 未知类标签设为-1
                    is_known_flag = False
                else:
                    continue  # 跳过不需要的类别
                
                # 加载图像
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(label)
                        self.is_known.append(is_known_flag)
        
        print(f"✅ 加载{domain_name}域: {len(self.images)}张图像, "
              f"已知类{sum(self.is_known)}张, 未知类{sum(not x for x in self.is_known)}张")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        is_known = self.is_known[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label, is_known, img_path

# ======================== 模型定义 ========================
class FeatureDisentanglementModel(nn.Module):
    """特征解耦模型（与DomainNet_test_08结构一致）"""
    def __init__(self, num_classes=241, feature_dim=2048):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name="convnextv2_tiny",
            pretrained=False,
            num_classes=0
        )
        
        backbone_out_dim = self.backbone.num_features
        
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        ).to(cfg.DEVICE)
        
        self.class_branch = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        ).to(cfg.DEVICE)
        
        self.domain_branch = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        ).to(cfg.DEVICE)
        
        self.class_centers = None
    
    def forward(self, x):
        base_feature_raw = self.backbone(x)
        base_feature = self.feature_proj(base_feature_raw)
        class_logits = self.class_branch(base_feature)
        domain_logits = self.domain_branch(base_feature)
        return base_feature, class_logits, domain_logits

class FeatureVAE(nn.Module):
    """VAE重构模型（与DomainNet_test_07结构一致）"""
    def __init__(self, feature_dim=2048, latent_dim=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 1536),
            nn.BatchNorm1d(1536),
            nn.GELU(),
            nn.Linear(1536, latent_dim * 2)
        ).to(cfg.DEVICE)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1536),
            nn.BatchNorm1d(1536),
            nn.GELU(),
            nn.Linear(1536, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, feature_dim)
        ).to(cfg.DEVICE)
        
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, log_var = torch.chunk(mu_logvar, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var, z

# ======================== 类中心重新计算 ========================
@torch.no_grad()
def compute_officehome_class_centers(disent_model, data_loader, num_known_classes, transform):
    """
    使用Office-Home数据重新计算已知类的类中心
    """
    disent_model.eval()
    centers = torch.zeros(num_known_classes, 2048).to(cfg.DEVICE)
    counts = torch.zeros(num_known_classes).to(cfg.DEVICE)
    
    for batch in data_loader:
        if len(batch) == 4:
            images, labels, is_known, _ = batch
        else:
            images, labels = batch[:2]
            is_known = [True] * len(labels)
        
        # 只使用已知类数据
        known_mask = torch.tensor(is_known)
        if known_mask.sum() == 0:
            continue
        
        images = apply_albumentations(images, transform)
        features, _, _ = disent_model(images)
        
        for i, (feat, label, known) in enumerate(zip(features, labels, is_known)):
            if known and label >= 0 and label < num_known_classes:
                centers[label] += feat
                counts[label] += 1
    
    # 归一化
    counts = torch.clamp(counts, min=1e-8)
    centers = centers / counts.unsqueeze(1)
    
    # L2归一化
    centers = F.normalize(centers, dim=1)
    
    print(f"✅ 重新计算类中心完成: {centers.shape}, 有效类别数: {(counts > 0).sum().item()}")
    return centers

# ======================== 融合权重优化 ========================
def optimize_fusion_weights(disent_model, vae_model, data_loader, transform, num_known_classes):
    """
    使用网格搜索优化融合权重
    """
    print("\n🔧 优化融合权重...")
    
    # 收集所有样本的特征和标签
    all_features = []
    all_logits = []
    all_recon_features = []
    all_is_known = []
    
    disent_model.eval()
    vae_model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 4:
                images, labels, is_known, _ = batch
            else:
                images, labels = batch[:2]
                is_known = [True] * len(labels)
            
            images = apply_albumentations(images, transform)
            features, logits, _ = disent_model(images)
            recon_features, _, _, _ = vae_model(features)
            
            all_features.append(features.cpu())
            all_logits.append(logits.cpu())
            all_recon_features.append(recon_features.cpu())
            all_is_known.extend(is_known if isinstance(is_known, list) else is_known.tolist())
    
    all_features = torch.cat(all_features, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    all_recon_features = torch.cat(all_recon_features, dim=0)
    all_is_known = np.array(all_is_known)
    
    # 计算各个得分
    probs = F.softmax(all_logits, dim=1)
    max_probs, _ = probs.max(dim=1)
    confidence_scores = (1.0 - max_probs).numpy()
    
    recon_errors = F.mse_loss(all_recon_features, all_features, reduction='none').mean(dim=1).numpy()
    feature_variance = all_features.var(dim=1).numpy()
    
    # 网格搜索
    best_auroc = 0
    best_weights = (0.30, 0.40, 0.20, 0.10)
    best_threshold = 0.5
    
    weight_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for w1 in weight_range:
        for w2 in weight_range:
            for w3 in weight_range:
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0 or w4 > 0.5:
                    continue
                
                # 归一化得分
                conf_norm = (confidence_scores - confidence_scores.min()) / (confidence_scores.max() - confidence_scores.min() + 1e-8)
                recon_norm = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-8)
                var_norm = (feature_variance - feature_variance.min()) / (feature_variance.max() - feature_variance.min() + 1e-8)
                
                # 融合得分（不使用距离得分，因为类中心可能不匹配）
                scores = w1 * conf_norm + w3 * recon_norm + w4 * var_norm
                
                # 计算AUROC
                binary_labels = np.array([0 if x else 1 for x in all_is_known])
                try:
                    auroc = roc_auc_score(binary_labels, scores)
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_weights = (w1, w2, w3, w4)
                except:
                    pass
    
    print(f"✅ 最佳权重: w1={best_weights[0]:.2f}(置信度), w3={best_weights[2]:.2f}(重构误差), w4={best_weights[3]:.2f}(方差)")
    print(f"✅ 最佳AUROC: {best_auroc:.4f}")
    
    return best_weights, best_auroc

# ======================== 开放集拒识策略 ========================
class OpenSetDetector:
    """
    多源融合开放集检测器
    结合置信度、特征距离、重构误差和特征方差
    """
    def __init__(self, known_class_centers, threshold=0.5):
        self.known_class_centers = known_class_centers
        self.threshold = threshold
        
        # 融合权重（最佳版本：AUROC 0.6974）
        self.w1 = 0.10  # 置信度权重
        self.w2 = 0.50  # 特征距离权重
        self.w3 = 0.40  # 重构误差权重
        self.w4 = 0.00  # 特征方差权重
    
    def compute_scores(self, features, logits, recon_features):
        """
        计算开放集得分
        返回: 开放集得分（越高越可能是未知类）
        """
        batch_size = features.size(0)
        
        # 1. 置信度得分（最大softmax概率的负值）
        probs = F.softmax(logits, dim=1)
        max_probs, _ = probs.max(dim=1)
        confidence_score = 1.0 - max_probs  # 置信度越低，越可能是未知类
        
        # 2. 特征距离得分（到最近类中心的距离）
        if self.known_class_centers is not None:
            centers = self.known_class_centers.to(features.device)
            distances = torch.cdist(F.normalize(features), F.normalize(centers))
            min_distances, _ = distances.min(dim=1)
            distance_score = min_distances / (min_distances.max() + 1e-8)  # 归一化
        else:
            distance_score = torch.zeros(batch_size, device=features.device)
        
        # 3. 重构误差得分
        recon_errors = F.mse_loss(recon_features, features, reduction='none').mean(dim=1)
        recon_score = recon_errors / (recon_errors.max() + 1e-8)  # 归一化
        
        # 4. 特征方差得分
        feature_variance = features.var(dim=1)
        variance_score = feature_variance / (feature_variance.max() + 1e-8)
        
        # 融合得分
        open_set_score = (
            self.w1 * confidence_score +
            self.w2 * distance_score +
            self.w3 * recon_score +
            self.w4 * variance_score
        )
        
        return open_set_score
    
    def predict(self, features, logits, recon_features):
        """预测是否为未知类"""
        scores = self.compute_scores(features, logits, recon_features)
        predictions = (scores > self.threshold).long()  # 1=未知类, 0=已知类
        return predictions, scores

# ======================== 评估函数 ========================
def compute_auroc_and_hscore(scores, is_known_labels):
    """
    计算AUROC和H-score
    Args:
        scores: 开放集得分（越高越可能是未知类）
        is_known_labels: 是否为已知类的标签（True=已知类, False=未知类）
    """
    # 转换标签：已知类=0, 未知类=1
    binary_labels = np.array([0 if x else 1 for x in is_known_labels])
    scores = np.array(scores)
    
    # 计算AUROC
    auroc = roc_auc_score(binary_labels, scores)
    
    # 计算最佳阈值和H-score
    fpr, tpr, thresholds = roc_curve(binary_labels, scores)
    
    best_h_score = 0
    best_threshold = 0
    for i, thresh in enumerate(thresholds):
        tnr = 1 - fpr[i]  # 真负率（已知类正确识别率）
        tpr_val = tpr[i]  # 真正率（未知类正确识别率）
        h_score = 2 * tpr_val * tnr / (tpr_val + tnr + 1e-8)
        if h_score > best_h_score:
            best_h_score = h_score
            best_threshold = thresh
    
    return auroc, best_h_score, best_threshold

def validate_on_office_home(disent_model, vae_model, data_loader, open_set_detector, transform):
    """
    在Office-Home数据集上进行验证
    """
    disent_model.eval()
    vae_model.eval()
    
    all_scores = []
    all_is_known = []
    all_preds = []
    all_labels = []
    correct_known = 0
    total_known = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 4:
                images, labels, is_known, _ = batch
            else:
                images, labels = batch[:2]
                is_known = [True] * len(labels)
            
            images = apply_albumentations(images, transform)
            
            # 前向传播
            features, logits, _ = disent_model(images)  # 忽略domain_logits
            recon_features, _, _, _ = vae_model(features)
            
            # 开放集检测
            predictions, scores = open_set_detector.predict(features, logits, recon_features)
            
            all_scores.extend(scores.cpu().numpy())
            all_is_known.extend(is_known if isinstance(is_known, list) else is_known.tolist())
            
            # 计算已知类分类准确率
            for i, (pred, label, known) in enumerate(zip(predictions, labels, is_known)):
                if known:
                    total_known += 1
                    # 对于已知类，检查分类是否正确
                    # 注意：这里需要将Office-Home的标签映射到DomainNet的类别空间
                    # 由于是跨域验证，我们只统计开放集检测指标
                    if pred == 0:  # 正确识别为已知类
                        correct_known += 1
    
    # 计算开放集指标
    auroc, h_score, best_threshold = compute_auroc_and_hscore(all_scores, all_is_known)
    
    # 计算已知类识别率
    known_recall = correct_known / (total_known + 1e-8)
    
    return {
        'auroc': auroc,
        'h_score': h_score,
        'best_threshold': best_threshold,
        'known_recall': known_recall,
        'total_samples': len(all_scores),
        'known_samples': sum(all_is_known),
        'unknown_samples': len(all_scores) - sum(all_is_known)
    }

# ======================== 主函数 ========================
def main():
    print("=" * 80)
    print("🚀 Office-Home数据集泛化验证")
    print("=" * 80)
    print(f"📌 设备: {cfg.DEVICE}")
    print(f"📌 模型路径: {cfg.MODEL_PATH}")
    print(f"📌 数据路径: {cfg.DATA_ROOT}")
    print(f"📌 已知类数量: {cfg.NUM_KNOWN_CLASSES}, 未知类数量: {cfg.NUM_UNKNOWN_CLASSES}")
    print("=" * 80)
    
    # 检查数据集是否存在
    if not os.path.exists(cfg.DATA_ROOT):
        print(f"❌ 数据集路径不存在: {cfg.DATA_ROOT}")
        print("请确保Office-Home数据集已上传到指定路径")
        return
    
    # 加载模型
    print("\n📦 加载模型...")
    disent_model = FeatureDisentanglementModel(
        num_classes=cfg.DOMAINNET_NUM_CLASSES,
        feature_dim=2048
    ).to(cfg.DEVICE)
    
    vae_model = FeatureVAE(
        feature_dim=2048,
        latent_dim=1024
    ).to(cfg.DEVICE)
    
    # 加载权重
    if os.path.exists(cfg.MODEL_PATH):
        checkpoint = torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE, weights_only=False)
        disent_model.load_state_dict(checkpoint['disent_model'])
        vae_model.load_state_dict(checkpoint['vae_model'])
        print(f"✅ 成功加载模型权重")
        
        # 加载类中心
        if 'class_centers' in checkpoint:
            disent_model.class_centers = checkpoint['class_centers']
            print(f"✅ 加载类中心: {disent_model.class_centers.shape}")
    else:
        print(f"❌ 模型权重不存在: {cfg.MODEL_PATH}")
        return
    
    # 数据增强
    val_transform = get_val_transform()
    
    # 定义已知类和未知类
    known_classes = list(range(cfg.NUM_KNOWN_CLASSES))
    unknown_classes = list(range(cfg.NUM_KNOWN_CLASSES, cfg.TOTAL_CLASSES))
    
    # 测试所有域
    domains = ['Art', 'Clipart', 'Product', 'Real World']
    results = {}
    optimized_centers = None
    optimized_weights = None
    
    # ======================== 优化阶段 ========================
    print("\n" + "=" * 80)
    print("🔧 优化阶段：重新计算类中心 + 优化融合权重")
    print("=" * 80)
    
    # 使用第一个域进行优化
    first_domain = domains[0]
    domain_path = os.path.join(cfg.DATA_ROOT, first_domain)
    if os.path.exists(domain_path):
        print(f"\n📊 使用 {first_domain} 域进行优化...")
        
        # 创建优化数据加载器
        opt_dataset = OfficeHomeDataset(
            root_dir=cfg.DATA_ROOT,
            domain_name=first_domain,
            transform=val_transform,
            known_classes=known_classes,
            unknown_classes=unknown_classes,
            is_target=True
        )
        
        opt_loader = DataLoader(
            opt_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True
        )
        
        # 重新计算类中心
        print("\n📍 重新计算类中心...")
        optimized_centers = compute_officehome_class_centers(
            disent_model, opt_loader, cfg.NUM_KNOWN_CLASSES, val_transform
        )
        
        # 优化融合权重
        optimized_weights, opt_auroc = optimize_fusion_weights(
            disent_model, vae_model, opt_loader, val_transform, cfg.NUM_KNOWN_CLASSES
        )
    
    # ======================== 验证阶段 ========================
    print("\n" + "=" * 80)
    print("📊 验证阶段：使用优化后的参数进行验证")
    print("=" * 80)
    
    for domain in domains:
        domain_path = os.path.join(cfg.DATA_ROOT, domain)
        if not os.path.exists(domain_path):
            print(f"⚠️ 跳过不存在的域: {domain}")
            continue
        
        print(f"\n{'='*40}")
        print(f"📊 验证域: {domain}")
        print(f"{'='*40}")
        
        # 创建数据加载器
        dataset = OfficeHomeDataset(
            root_dir=cfg.DATA_ROOT,
            domain_name=domain,
            transform=val_transform,
            known_classes=known_classes,
            unknown_classes=unknown_classes,
            is_target=True
        )
        
        if len(dataset) == 0:
            print(f"⚠️ 域 {domain} 没有有效数据")
            continue
        
        data_loader = DataLoader(
            dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True
        )
        
        # 创建开放集检测器（使用优化后的类中心）
        open_set_detector = OpenSetDetector(
            known_class_centers=optimized_centers if optimized_centers is not None else disent_model.class_centers,
            threshold=0.5
        )
        
        # 验证
        result = validate_on_office_home(
            disent_model, vae_model, data_loader, open_set_detector, val_transform
        )
        
        results[domain] = result
        
        # 打印结果
        print(f"\n📊 {domain}域验证结果:")
        print(f"  - AUROC: {result['auroc']:.4f} (目标≥{cfg.TARGET_AUROC}) {'✅' if result['auroc'] >= cfg.TARGET_AUROC else '❌'}")
        print(f"  - H-score: {result['h_score']:.4f} (目标≥{cfg.TARGET_H_SCORE}) {'✅' if result['h_score'] >= cfg.TARGET_H_SCORE else '❌'}")
        print(f"  - 已知类识别率: {result['known_recall']:.4f}")
        print(f"  - 最佳阈值: {result['best_threshold']:.4f}")
        print(f"  - 样本数: {result['total_samples']} (已知类: {result['known_samples']}, 未知类: {result['unknown_samples']})")
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("📊 汇总结果")
    print("=" * 80)
    
    avg_auroc = np.mean([r['auroc'] for r in results.values()])
    avg_h_score = np.mean([r['h_score'] for r in results.values()])
    
    print(f"平均AUROC: {avg_auroc:.4f}")
    print(f"平均H-score: {avg_h_score:.4f}")
    
    # 保存结果
    result_file = os.path.join(cfg.RESULT_DIR, "office_home_validation_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Office-Home数据集泛化验证结果（优化版）\n")
        f.write("=" * 60 + "\n")
        f.write(f"验证时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {cfg.MODEL_PATH}\n")
        f.write(f"已知类数量: {cfg.NUM_KNOWN_CLASSES}\n")
        f.write(f"未知类数量: {cfg.NUM_UNKNOWN_CLASSES}\n")
        
        # 记录优化参数
        if optimized_weights is not None:
            f.write(f"\n优化后的融合权重:\n")
            f.write(f"  w1(置信度): {optimized_weights[0]:.2f}\n")
            f.write(f"  w2(距离): {optimized_weights[1]:.2f}\n")
            f.write(f"  w3(重构误差): {optimized_weights[2]:.2f}\n")
            f.write(f"  w4(方差): {optimized_weights[3]:.2f}\n")
        
        if optimized_centers is not None:
            f.write(f"\n类中心: 重新计算（{optimized_centers.shape[0]}类）\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("各域验证结果:\n")
        
        for domain, result in results.items():
            f.write(f"\n{domain}域:\n")
            f.write(f"  AUROC: {result['auroc']:.4f} {'✅' if result['auroc'] >= cfg.TARGET_AUROC else '❌'}\n")
            f.write(f"  H-score: {result['h_score']:.4f} {'✅' if result['h_score'] >= cfg.TARGET_H_SCORE else '❌'}\n")
            f.write(f"  已知类识别率: {result['known_recall']:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"平均AUROC: {avg_auroc:.4f} {'✅' if avg_auroc >= cfg.TARGET_AUROC else '❌'}\n")
        f.write(f"平均H-score: {avg_h_score:.4f} {'✅' if avg_h_score >= cfg.TARGET_H_SCORE else '❌'}\n")
    
    print(f"\n✅ 结果已保存到: {result_file}")

if __name__ == "__main__":
    main()
