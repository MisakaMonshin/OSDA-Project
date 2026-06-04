# 代码说明

本目录包含本项目实现的完整代码。

## 📁 目录结构

```
code/
├── core/                # 核心模型与训练
├── evaluation/          # 评估与可视化
└── README.md            # 本文件
```

## 🔧 核心模块说明

### `core/`

| 文件 | 功能 |
|------|------|
| `VisDA_train.py` | VisDA数据集训练主脚本 |
| `DomainNet_train.py` | DomainNet数据集训练主脚本 |
| `VAE_module.py` | 变分自编码器重构模块（创新点3）|
| `Disentanglement.py` | 双分支特征解耦模块（创新点2）|
| `OSBP_baseline.py` | OSBP基线方法复现 |

### `evaluation/`

| 文件 | 功能 |
|------|------|
| `Baseline_comparison.py` | 多基线方法公平对比 |
| `VisDA_visualization.py` | VisDA分类结果可视化 |
| `DomainNet_visualization.py` | DomainNet分类结果可视化 |
| `CrossDomain_tsne.py` | 跨域特征分布t-SNE |
| `Unknown_detection.py` | 未知类检测能力对比 |

## 🚀 训练流程

1. **数据准备**
   - 下载 VisDA / DomainNet / Office-Home 数据集
   - 划分已知类/未知类（参考 `data/processed/` 下的CSV文件）

2. **模型训练**
   ```bash
   python core/VisDA_train.py
   python core/DomainNet_train.py
   ```

3. **评估与可视化**
   ```bash
   python evaluation/Baseline_comparison.py
   ```

## ⚙️ 关键超参数

| 超参数 | 值 | 说明 |
|--------|:---:|------|
| 主干网络 | ConvNeXt-V2-Tiny | 768维特征输出 |
| 特征投影维度 | 1024 | FC + BN + ReLU |
| 类别分支 | 1024→512→C | C=已知类数 |
| 域分支 | 1024→512→1 | 单输出 |
| 融合权重 | 30/40/20/10 | 置信/距离/重构/方差 |
| Batch size | 64 | 单卡 |
| 初始学习率 | 1e-4 | Adam优化器 |
| Epoch | 35 | 三阶段训练 |

## 📊 性能指标

| 指标 | VisDA | DomainNet |
|------|:---:|:---:|
| 已知类准确率 | 83.82% | 85.90% |
| AUROC | 86.19% | 96.37% |
| H-score | 80.72% | 91.90% |
| TNR (未知类召回) | 61.00% | 88.95% |

## 📝 注意事项

1. **GPU要求**：建议 RTX 3090 / 4060 Ti 16GB 或以上
2. **Python版本**：3.8+
3. **依赖安装**：`pip install -r requirements.txt`
4. **数据路径**：根据实际路径修改 `cfg.RAW_*_ROOT` 等配置

## 🤝 贡献

如有问题或建议，请提交Issue。
