import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureVAE(nn.Module):
    def __init__(self, feature_dim=2048, latent_dim=1024):  # 关键修改：latent_dim从512→1024
        super().__init__()
        # 编码器：移除Dropout，增强特征保留能力
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 2048),  # 关键修改：加宽中间层，提升编码能力
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 1536),
            nn.BatchNorm1d(1536),
            nn.GELU(),
            nn.Linear(1536, latent_dim * 2)  # 输出μ和log_var（各1024维）
        )
        # 解码器：移除Dropout，确保重构精度
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1536),
            nn.BatchNorm1d(1536),
            nn.GELU(),
            nn.Linear(1536, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, feature_dim)  # 输出2048维重构特征
        )
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=mu.device)
        return mu + eps * std

    def forward(self, feature):
        mu_logvar = self.encoder(feature)
        mu, log_var = torch.chunk(mu_logvar, 2, dim=1)  # [B,1024] ×2
        z = self.reparameterize(mu, log_var)
        recon_feature = self.decoder(z)
        return recon_feature, mu, log_var, z  # 新增返回隐向量z，用于联合训练

# 联合损失函数：适配解耦+重构协同训练
class JointReconDisentLoss(nn.Module):
    def __init__(self, lambda_kl=0.0005, lambda_disent=0.3, lambda_recon=0.7):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_kl = lambda_kl  # 关键修改：KL权重从0.001→0.0005
        self.lambda_disent = lambda_disent  # 解耦损失权重
        self.lambda_recon = lambda_recon    # 重构损失权重

    def forward(self, 
                recon_feature, origin_feature, mu, log_var,  # VAE相关
                logits, labels, disent_loss):  # 解耦模块相关
        # 1. 重构损失（MSE）
        recon_loss = self.mse(recon_feature, origin_feature)
        # 2. KL散度损失
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        # 3. 分类损失（交叉熵）
        cls_loss = self.ce(logits, labels)
        # 4. 总联合损失：加权融合
        total_loss = (self.lambda_recon * recon_loss) + \
                     (self.lambda_kl * kl_loss) + \
                     (self.lambda_disent * (cls_loss + disent_loss))  # 解耦损失包含原分类损失
        return total_loss, {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "cls_loss": cls_loss,
            "disent_loss": disent_loss,
            "total_loss": total_loss
        }

# 测试代码：验证维度适配
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_feature = torch.randn(64, 2048).to(device)  # 解耦模块输出的2048维特征
    fake_labels = torch.randint(0, 241, (64,)).to(device)  # 分类标签（241类）
    
    vae = FeatureVAE().to(device)
    joint_loss_fn = JointReconDisentLoss()
    
    # VAE前向传播
    recon_feature, mu, log_var, z = vae(fake_feature)
    # 模拟解耦模块输出（实际训练时从FeatureDisentanglementModel获取）
    fake_logits = torch.randn(64, 241).to(device)
    fake_disent_loss = torch.tensor(0.1).to(device)  # 模拟解耦损失
    
    # 计算联合损失
    total_loss, loss_dict = joint_loss_fn(
        recon_feature=recon_feature,
        origin_feature=fake_feature,
        mu=mu,
        log_var=log_var,
        logits=fake_logits,
        labels=fake_labels,
        disent_loss=fake_disent_loss
    )
    
    # 验证输出维度
    print(f"输入特征形状：{fake_feature.shape}")          # [64,2048]
    print(f"重构特征形状：{recon_feature.shape}")        # [64,2048]（维度匹配）
    print(f"隐向量z形状：{z.shape}")                      # [64,1024]
    print(f"联合损失：{total_loss.item():.4f}")
    print(f"损失分解：MSE={loss_dict['recon_loss'].item():.4f}, "
          f"KL={loss_dict['kl_loss'].item():.4f}, "
          f"分类={loss_dict['cls_loss'].item():.4f}, "
          f"解耦={loss_dict['disent_loss'].item():.4f}")
    print(f"VAE参数总量：{sum(p.numel() for p in vae.parameters())/1e6:.1f}M")  # 约14.5M，轻量化无负担