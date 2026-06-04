# 更新日志

## v1.0 (2026-05-28) - 大创结题版本

### 核心功能
- 特征解耦与数据重构的开放集域适应方法实现
- 动态梯度反转层（DGRL）
- 四源融合开放集检测
- 三阶段动态训练策略

### 实验数据集
- VisDA-2017（Synthetic → Real）
- DomainNet（Real → Clipart）
- Office-Home（四域泛化）

### 性能指标
| 指标 | VisDA | DomainNet |
|------|:---:|:---:|
| 已知类准确率 | 83.82% | 85.90% |
| AUROC | 86.19% | 96.37% |
| H-score | 80.72% | 91.90% |

### 项目成果
- 国家发明专利已受理（申请号：202610279677.6）
- 所有立项目标超额完成

### 文件结构
- `code/core/` - 核心模型与训练
- `code/evaluation/` - 评估脚本
- `docs/reports/` - 报告文档
- `docs/slides/` - 答辩PPT与备答
- `docs/figures/` - 图表
- `application/` - 申请材料
- `references/` - 参考文献说明

---

*大创项目结题存档版本*
