import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import time
import numpy as np
import scipy.io as sio
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
import cv2
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_KNOWN_CLASSES = 8
TOTAL_VISDA_CLASSES = 10
BATCH_SIZE = 64
UAN_BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
PATIENCE = 3

BASE_DIR = "/root/autodl-tmp/OSDA_Project"
VISDA_DATA_ROOT = os.path.join(BASE_DIR, "data/processed/VisDA")
RAW_VISDA_ROOT = os.path.join(BASE_DIR, "data/raw/VisDA")
WEIGHTS_DIR = os.path.join(BASE_DIR, "code/test")
SAVE_DIR = os.path.join(BASE_DIR, "results/baseline_models_visda")
os.makedirs(SAVE_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

class VisDADataset(Dataset):
    def __init__(self, mat_path, transform=None, is_target=False, target_label_csv=None):
        self.transform = transform
        self.is_target = is_target
        self.class_type = None
        
        data = sio.loadmat(mat_path)
        self.images = np.transpose(data['X'], (3, 0, 1, 2))
        self.labels = data['y'].squeeze() - 1
        
        if target_label_csv and os.path.exists(target_label_csv):
            self.is_target = True
            label_df = pd.read_csv(target_label_csv)
            self.class_type = label_df['class_type'].values
            print(f"Target domain: {sum(self.class_type == 'share')} known, {sum(self.class_type == 'unknown')} unknown")
    
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

def get_transforms():
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
    return train_transform, val_transform

def load_pretrained_weights(model, local_path):
    if os.path.exists(local_path):
        state_dict = torch.load(local_path, map_location=DEVICE)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict, strict=False)
        print(f"Loaded pretrained weights: {len(filtered_state_dict)} layers")
    return model

class SourceOnlyModel(nn.Module):
    def __init__(self, num_classes=TOTAL_VISDA_CLASSES):
        super().__init__()
        self.backbone = timm.create_model("convnextv2_tiny", pretrained=False, num_classes=0)
        local_path = os.path.join(WEIGHTS_DIR, "pytorch_model.bin")
        self.backbone = load_pretrained_weights(self.backbone, local_path)
        
        backbone_out = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        ).to(DEVICE)
        
        self.class_centers = None
    
    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return feat, logits, torch.tensor(0.0, device=DEVICE)

class UANModel(nn.Module):
    def __init__(self, num_classes=TOTAL_VISDA_CLASSES, feature_dim=1024):
        super().__init__()
        self.backbone = timm.create_model("convnextv2_tiny", pretrained=False, num_classes=0)
        local_path = os.path.join(WEIGHTS_DIR, "pytorch_model.bin")
        self.backbone = load_pretrained_weights(self.backbone, local_path)
        
        backbone_out = self.backbone.num_features
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_out, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(DEVICE)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        ).to(DEVICE)
        
        self.domain_disc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        ).to(DEVICE)
        
        self.class_centers = None
    
    def forward(self, x):
        feat_raw = self.backbone(x)
        feat = self.feature_proj(feat_raw)
        logits = self.classifier(feat)
        return feat, logits, torch.tensor(0.0, device=DEVICE)

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def train_source_only(model, train_loader, val_loader, train_transform, val_transform):
    print("\n" + "="*60)
    print("Training Source Only Baseline on VisDA")
    print("="*60)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            known_mask = (labels < NUM_KNOWN_CLASSES) & (labels >= 0)
            if known_mask.sum() == 0:
                continue
            
            optimizer.zero_grad()
            _, logits, _ = model(images)
            loss = criterion(logits[known_mask], labels[known_mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits[known_mask].argmax(dim=1)
            correct += preds.eq(labels[known_mask]).sum().item()
            total += known_mask.sum().item()
        
        scheduler.step()
        train_acc = correct / total if total > 0 else 0
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        
        if train_acc > best_acc + 0.001:
            best_acc = train_acc
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}, no improvement for {PATIENCE} epochs")
                break
    
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "source_only_visda.pth"))
    print(f"Source Only - Best Train Acc: {best_acc:.4f} (Epoch {best_epoch})")
    return model

def train_uan(model, train_loader, val_loader, train_transform, val_transform):
    print("\n" + "="*60)
    print("Training UAN Baseline on VisDA")
    print("="*60)
    
    uan_train_loader = DataLoader(
        train_loader.dataset, 
        batch_size=UAN_BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    uan_val_loader = DataLoader(
        val_loader.dataset, 
        batch_size=UAN_BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0
    best_epoch = 0
    grl_alpha = 1.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        total_loss = 0.0
        
        target_iter = iter(uan_val_loader)
        
        for batch_idx, (images, labels) in enumerate(uan_train_loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            known_mask = (labels < NUM_KNOWN_CLASSES) & (labels >= 0)
            if known_mask.sum() == 0:
                continue
            
            try:
                tgt_batch = next(target_iter)
                tgt_images = tgt_batch[0].to(DEVICE, non_blocking=True)
            except StopIteration:
                target_iter = iter(uan_val_loader)
                tgt_batch = next(target_iter)
                tgt_images = tgt_batch[0].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                feat_src, logits, _ = model(images)
                cls_loss = criterion_cls(logits[known_mask], labels[known_mask])
                
                with torch.no_grad():
                    feat_tgt, _, _ = model(tgt_images)
                
                if epoch >= 5:
                    grl_alpha = max(1.0 * (0.95 ** (epoch // 5)), 0.5)
                    src_domain_pred = model.domain_disc(GradientReversalLayer.apply(feat_src.detach(), grl_alpha))
                    tgt_domain_pred = model.domain_disc(GradientReversalLayer.apply(feat_tgt, grl_alpha))
                    
                    domain_loss_src = criterion_domain(src_domain_pred, torch.zeros(len(feat_src), 1, device=DEVICE))
                    domain_loss_tgt = criterion_domain(tgt_domain_pred, torch.ones(len(feat_tgt), 1, device=DEVICE))
                    domain_loss = 0.5 * (domain_loss_src + domain_loss_tgt)
                else:
                    domain_loss = torch.tensor(0.0, device=DEVICE)
                
                loss = cls_loss + 0.1 * domain_loss
            
            loss_val = loss.item()
            
            with torch.no_grad():
                preds = logits[known_mask].argmax(dim=1)
                correct += preds.eq(labels[known_mask]).sum().item()
                total += known_mask.sum().item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss_val
            
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        torch.cuda.empty_cache()
        scheduler.step()
        train_acc = correct / total if total > 0 else 0
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(uan_train_loader):.4f}, Train Acc: {train_acc:.4f}")
        
        if train_acc > best_acc + 0.001:
            best_acc = train_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "uan_visda.pth"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}, no improvement for {PATIENCE} epochs")
                break
    
    print(f"UAN - Best Train Acc: {best_acc:.4f} (Epoch {best_epoch})")
    return model

@torch.no_grad()
def compute_class_centers(model, loader, num_classes):
    model.eval()
    feature_dim = 1024 if hasattr(model, 'feature_proj') else 768
    centers = torch.zeros(num_classes, feature_dim).to(DEVICE)
    counts = torch.zeros(num_classes).to(DEVICE)
    
    for images, labels, *_ in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        feat, _, _ = model(images)
        
        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                centers[c] += feat[mask].sum(dim=0)
                counts[c] += mask.sum()
    
    counts = torch.clamp(counts, min=1e-8)
    centers = centers / counts.unsqueeze(1)
    model.class_centers = centers
    return centers

@torch.no_grad()
def evaluate_osda(model, loader, method_name):
    print(f"\n{'='*60}")
    print(f"Evaluating {method_name} on VisDA Target Domain")
    print(f"{'='*60}")
    
    model.eval()
    
    centers = compute_class_centers(model, loader, NUM_KNOWN_CLASSES)
    
    all_conf = []
    all_dist = []
    all_labels = []
    correct = 0
    total = 0
    
    for images, labels, class_type in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        feat, logits, _ = model(images)
        
        known_mask = (labels < NUM_KNOWN_CLASSES) & (labels >= 0)
        if known_mask.sum() > 0:
            preds = logits[known_mask].argmax(dim=1)
            correct += preds.eq(labels[known_mask]).sum().item()
            total += known_mask.sum().item()
        
        conf = F.softmax(logits, dim=1).max(dim=1)[0]
        feat_norm = F.normalize(feat, dim=1)
        centers_norm = F.normalize(centers, dim=1)
        dist = torch.cdist(feat_norm, centers_norm).min(dim=1)[0]
        
        is_unknown = (labels >= NUM_KNOWN_CLASSES).float()
        
        all_conf.extend((1 - conf).cpu().numpy())
        all_dist.extend(dist.cpu().numpy())
        all_labels.extend(is_unknown.cpu().numpy())
    
    known_acc = correct / total if total > 0 else 0
    
    conf_arr = np.array(all_conf)
    dist_arr = np.array(all_dist)
    label_arr = np.array(all_labels)
    
    conf_arr = conf_arr / (np.max(conf_arr) + 1e-8)
    dist_arr = dist_arr / (np.max(dist_arr) + 1e-8)
    
    scores = 0.5 * conf_arr + 0.5 * dist_arr
    
    auroc = roc_auc_score(label_arr, scores)
    fpr, tpr, thresholds = roc_curve(label_arr, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    preds = (scores >= optimal_threshold).astype(int)
    tp = np.sum((preds == 1) & (label_arr == 1))
    fn = np.sum((preds == 0) & (label_arr == 1))
    fp = np.sum((preds == 1) & (label_arr == 0))
    tn = np.sum((preds == 0) & (label_arr == 0))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    h_score = 2 * tpr * tnr / (tpr + tnr) if (tpr + tnr) > 0 else 0
    
    print(f"{method_name} Results:")
    print(f"  Known Class Accuracy: {known_acc:.4f}")
    print(f"  Unknown Recall (TPR): {tpr:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  H-score: {h_score:.4f}")
    
    return known_acc, tpr, auroc, h_score

def main():
    print("="*60)
    print("VisDA Baseline Comparison (Fair Setting)")
    print("Training from ImageNet pretrained weights")
    print("="*60)
    
    train_mat = os.path.join(VISDA_DATA_ROOT, "source_train_32x32.mat")
    test_mat = os.path.join(RAW_VISDA_ROOT, "test_32x32.mat")
    target_label_csv = os.path.join(VISDA_DATA_ROOT, "target_test_labels.csv")
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = VisDADataset(train_mat, transform=train_transform, is_target=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    test_dataset = VisDADataset(test_mat, transform=val_transform, is_target=True, target_label_csv=target_label_csv)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    source_only_path = os.path.join(SAVE_DIR, "source_only_visda.pth")
    if os.path.exists(source_only_path):
        print("\n" + "="*60)
        print("Loading existing Source Only model...")
        print("="*60)
        source_only_model = SourceOnlyModel(num_classes=TOTAL_VISDA_CLASSES).to(DEVICE)
        source_only_model.load_state_dict(torch.load(source_only_path))
    else:
        source_only_model = SourceOnlyModel(num_classes=TOTAL_VISDA_CLASSES).to(DEVICE)
        source_only_model = train_source_only(source_only_model, train_loader, test_loader, train_transform, val_transform)
    
    print("\n" + "="*60)
    print("Clearing GPU memory before UAN training...")
    print("="*60)
    del source_only_model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    uan_path = os.path.join(SAVE_DIR, "uan_visda.pth")
    if os.path.exists(uan_path):
        print("\n" + "="*60)
        print("Loading existing UAN model...")
        print("="*60)
        uan_model = UANModel(num_classes=TOTAL_VISDA_CLASSES).to(DEVICE)
        uan_model.load_state_dict(torch.load(uan_path))
    else:
        uan_model = UANModel(num_classes=TOTAL_VISDA_CLASSES).to(DEVICE)
        uan_model = train_uan(uan_model, train_loader, test_loader, train_transform, val_transform)
    
    print("\n" + "="*60)
    print("Final Evaluation Results")
    print("="*60)
    
    source_only_model = SourceOnlyModel(num_classes=TOTAL_VISDA_CLASSES).to(DEVICE)
    source_only_model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "source_only_visda.pth")))
    so_known_acc, so_unknown_recall, so_auroc, so_h_score = evaluate_osda(source_only_model, test_loader, "Source Only")
    del source_only_model
    torch.cuda.empty_cache()
    
    uan_known_acc, uan_unknown_recall, uan_auroc, uan_h_score = evaluate_osda(uan_model, test_loader, "UAN")
    
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    print(f"{'Method':<20} {'Known Acc':<12} {'Unknown Recall':<15} {'AUROC':<10} {'H-score':<10}")
    print("-"*67)
    print(f"{'Source Only':<20} {so_known_acc:<12.4f} {so_unknown_recall:<15.4f} {so_auroc:<10.4f} {so_h_score:<10.4f}")
    print(f"{'UAN':<20} {uan_known_acc:<12.4f} {uan_unknown_recall:<15.4f} {uan_auroc:<10.4f} {uan_h_score:<10.4f}")
    print(f"{'Ours (VisDA_test_02)':<20} {'0.8382':<12} {'0.7800':<15} {'0.8619':<10} {'0.8072':<10}")
    
    results = {
        'source_only': {'known_acc': so_known_acc, 'unknown_recall': so_unknown_recall, 'auroc': so_auroc, 'h_score': so_h_score},
        'uan': {'known_acc': uan_known_acc, 'unknown_recall': uan_unknown_recall, 'auroc': uan_auroc, 'h_score': uan_h_score},
        'ours': {'known_acc': 0.8382, 'unknown_recall': 0.7800, 'auroc': 0.8619, 'h_score': 0.8072}
    }
    
    import json
    with open(os.path.join(SAVE_DIR, "comparison_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()
