# 基于特征解耦与数据重构的开放集域适应方法研究

> 大学生创新创业训练计划（大创）结题项目完整存档

## 📖 项目简介

本项目针对**开放集域适应**（Open Set Domain Adaptation, OSDA）任务，提出了一种基于特征解耦与数据重构的新方法。核心解决两个关键问题：
- 跨域迁移：如何将源域学到的分类知识迁移到分布不同的目标域
- 开放集检测：如何识别目标域中训练时未出现过的未知类别

### 🎯 核心创新

| 创新点 | 描述 |
|--------|------|
| 动态梯度反转层（DGRL）| 训练过程中α系数指数衰减，自适应调整对抗强度 |
| 双分支特征解耦 | 类别分支+域分支，Gram矩阵正交约束 |
| VAE重构辅助检测 | 引入变分自编码器提供独立异常检测信号 |
| 四源融合检测 | 置信度30%+距离40%+重构误差20%+方差10% |

### 📊 实验结果

| 指标 | VisDA | DomainNet | 目标 |
|------|:---:|:---:|:---:|
| 已知类准确率 | 83.82% | 85.90% | ≥76.5% |
| AUROC | 86.19% | 96.37% | ≥75% |
| H-score | 80.72% | 91.90% | ≥65% |

## 📁 仓库结构

```
.
├── README.md                    # 本文件
├── LICENSE                      # 许可证
├── .gitignore                   # Git忽略规则
├── code/                        # 项目代码
│   ├── core/                    # 核心模块
│   │   ├── VisDA_train.py       # VisDA训练脚本
│   │   ├── DomainNet_train.py   # DomainNet训练脚本
│   │   ├── VAE_module.py        # VAE重构模块
│   │   ├── Disentanglement.py   # 特征解耦模块
│   │   └── OSBP_baseline.py     # OSBP基线
│   ├── evaluation/              # 评估脚本
│   │   ├── Baseline_comparison.py
│   │   ├── VisDA_visualization.py
│   │   ├── DomainNet_visualization.py
│   │   ├── CrossDomain_tsne.py
│   │   └── Unknown_detection.py
│   └── README.md                # 代码说明
├── docs/                        # 项目文档
│   ├── reports/                 # 报告类
│   │   ├── 结题报告.docx
│   │   ├── 系统填报报告.docx
│   │   └── 评审报告.md
│   ├── slides/                  # 答辩PPT
│   │   ├── 结题项目_最终版.pptx
│   │   └── 备答说明.md
│   ├── figures/                 # 图表
│   │   ├── 方法架构图.mmd       # Mermaid源码
│   │   ├── 实验结果.png
│   │   └── 分类样例.png
│   └── 公式与参数说明.md
├── application/                 # 申请材料
│   ├── 立项申请.doc
│   ├── 中期检查报告.pdf
│   ├── 季度报告.pdf
│   └── 专利/
│       ├── 受理通知书.pdf
│       └── 交底材料书.docx
└── references/                  # 参考文献说明
    └── 文献核心观点.md
```

## 🚀 快速使用

### 环境依赖
- Python 3.8+
- PyTorch 1.10+
- timm (ConvNeXt-V2)
- albumentations
- matplotlib, scipy, pandas, scikit-learn

```bash
pip install torch timm albumentations matplotlib scipy pandas scikit-learn
```

### 运行VisDA训练
```bash
python code/core/VisDA_train.py
```

### 运行DomainNet训练
```bash
python code/core/DomainNet_train.py
```

### 评估与可视化
```bash
python code/evaluation/Baseline_comparison.py
python code/evaluation/CrossDomain_tsne.py
```

## 📈 实验数据来源

- **VisDA-2017**：http://csr.bu.edu/ftp/visda/2017/multi-source/
- **DomainNet**：http://ai.bu.edu/M3SDA/
- **Office-Home**：https://www.hemanthdv.org/officeHomeDataset.html

## 🏆 项目成果

1. **国家发明专利**（已受理）：基于特征解耦与重构辅助的开放集域适应方法及系统
2. **核心指标全部超额达成**：三大数据集上所有指标超过立项目标
3. **完整技术文档**：含结题报告、PPT、评审报告、备答材料

## 📚 参考文献

- Saito et al. **OSBP: Open Set Domain Adaptation by Backpropagation**, ECCV 2018
- Yan et al. **UAN: Learning to Adapt for Open Set Domain Adaptation**, ICCV 2017
- Liu et al. **STA: Separate to Adapt**, CVPR 2019
- Ganin et al. **DANN: Domain-Adversarial Training of Neural Networks**, JMLR 2016
- Bousmalis et al. **Domain Separation Networks**, NeurIPS 2016
- Kingma & Welling. **Auto-Encoding Variational Bayes**, ICLR 2014
- Woo et al. **ConvNeXt V2**, CVPR 2023

## 📝 License

MIT License - 详见 [LICENSE](LICENSE) 文件

## 👥 团队

项目成员：[成员1], [成员2], [成员3]
指导教师：[导师姓名]
学校：[学校名称]
完成时间：2026年5月
