import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import timm

class FeatureDisentanglementModel(nn.Module):
    def __init__(self, num_classes=241, feature_dim=2048):
        super().__init__()
        self.backbone = timm.create_model(
            model_name="convnextv2_tiny",
            pretrained=False,
            num_classes=0
        )
        local_weight_path = "code/test/pytorch_model.bin"
        state_dict = torch.load(local_weight_path, map_location="cpu")
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        self.backbone.load_state_dict(filtered_state_dict, strict=False)
        
        backbone_out_dim = self.backbone.num_features
        
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_out_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 关键修改：降低冻结比例从30%→10%，增加可训练参数，提升显存占用
        total_layers = len(list(self.backbone.parameters()))
        freeze_layers = int(total_layers * 0.1)  # 仅冻结前10%层
        for param in list(self.backbone.parameters())[:freeze_layers]:
            param.requires_grad = False
        # 打印可训练参数数量
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ 模型可训练参数：{trainable_params/1e6:.2f}M（应≥25M）")
        
        # 双分支结构不变
        self.class_branch = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        self.domain_branch = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        base_feature_raw = self.backbone(x)
        base_feature = self.feature_proj(base_feature_raw)
        class_logits = self.class_branch(base_feature)
        domain_logits = self.domain_branch(base_feature)
        return base_feature, class_logits, domain_logits

# 损失函数部分完全不变，无需修改（已适配MixUp）
class DisentangleLoss(nn.Module):
    def __init__(self, lambda_orth=0.5, lambda_mi=0.05):
        super().__init__()
        self.lambda_orth = lambda_orth
        self.lambda_mi = lambda_mi
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()

    def orthogonal_constraint(self, feature):
        feature = F.normalize(feature, p=2, dim=1)
        gram = torch.mm(feature.T, feature) / feature.size(0)
        orth_loss = torch.norm(gram - torch.eye(gram.size(0)).to(feature.device), p="fro")
        return orth_loss

    def mutual_info_minimization(self, feature):
        feature = F.normalize(feature, p=2, dim=0)
        mi_loss = -torch.mean(torch.sum(feature * torch.log(feature + 1e-8), dim=0))
        return mi_loss

    def forward(self, base_feature, class_logits, domain_logits, class_label, domain_label):
        class_loss = self.class_criterion(class_logits, class_label)
        domain_loss = self.domain_criterion(domain_logits, domain_label.float())
        orth_loss = self.orthogonal_constraint(base_feature) * self.lambda_orth
        mi_loss = self.mutual_info_minimization(base_feature) * self.lambda_mi
        
        total_loss = class_loss + domain_loss + orth_loss + mi_loss
        
        loss_dict = {
            "class_loss": class_loss,
            "domain_loss": domain_loss,
            "orth_loss": orth_loss,
            "mi_loss": mi_loss
        }
        return total_loss, loss_dict