# GitHub 上传操作指南

## 📌 步骤一：在 GitHub 网页上创建空仓库

1. **登录 GitHub**：访问 https://github.com/MisakaMonshin

2. **创建新仓库**：
   - 点击右上角 `+` → `New repository`
   - Repository name: `OSDA-Project`
   - Description: `基于特征解耦与数据重构的开放集域适应方法研究 - 大创结题项目存档`
   - 选择 **Public**（公开）或 **Private**（私有）
   - **不要**勾选 `Add a README file`、`Add .gitignore`、`Choose a license`（我们已经有了）
   - 点击 `Create repository`

3. **仓库创建成功后**，GitHub 会跳转到新页面，URL类似：
   `https://github.com/MisakaMonshin/OSDA-Project`

---

## 📌 步骤二：把本地仓库推送到 GitHub

打开终端，依次执行以下命令：

```bash
# 1. 进入本地仓库目录
cd /root/autodl-tmp/OSDA_Project/github_upload

# 2. 确认remote已设置
git remote -v

# 应该显示：
# origin  https://github.com/MisakaMonshin/OSDA-Project.git (fetch)
# origin  https://github.com/MisakaMonshin/OSDA-Project.git (push)

# 3. 推送到 GitHub
git push -u origin main
```

### 第一次推送会要求身份验证

会提示你输入 GitHub 用户名和密码。**但** GitHub 已经不支持密码登录了，你需要使用 **Personal Access Token (PAT)**：

#### 创建 Token 的步骤：

1. 访问 https://github.com/settings/tokens
2. 点击 `Generate new token` → `Generate new token (classic)`
3. 填写：
   - Note: `OSDA-Project-Upload`
   - Expiration: 30 days 或更长
   - 勾选 `repo` 权限
4. 点击 `Generate token`
5. **复制生成的 token**（只显示一次！）

#### 使用 Token 推送：

```bash
# 推送时使用token作为密码
git push -u origin main
# Username: MisakaMonshin
# Password: ghp_xxxxxxxxxxxxxxxxxxxx (粘贴你的token)
```

#### 或者更简单的方法（推荐）：

```bash
# 把token直接写进remote URL（这样不需要每次输入）
git remote set-url origin https://MisakaMonshin:ghp_你的token@github.com/MisakaMonshin/OSDA-Project.git
git push -u origin main
```

---

## 📌 步骤三：验证上传成功

访问 https://github.com/MisakaMonshin/OSDA-Project

你应该能看到：
- ✅ 28个文件已上传
- ✅ README.md 正确显示（项目说明）
- ✅ 文件结构完整（code/、docs/、application/、references/）

---

## 🔧 如果遇到常见错误

### 错误1：`Permission denied (publickey)`
**原因**：没有配置SSH密钥
**解决**：
```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "your_email@example.com"
# 复制公钥
cat ~/.ssh/id_ed25519.pub
# 把公钥添加到 GitHub: https://github.com/settings/keys
```

### 错误2：`Repository not found`
**原因**：仓库URL错误或仓库不存在
**解决**：
1. 确认已经在 GitHub 网页上创建了仓库
2. 检查用户名和仓库名拼写是否正确

### 错误3：`failed to push some refs`
**原因**：远程仓库有本地没有的提交
**解决**：
```bash
git pull --rebase origin main
git push -u origin main
```

---

## 📊 项目仓库预览

上传成功后，仓库结构应该是这样：

```
https://github.com/MisakaMonshin/OSDA-Project
├── README.md                 # 项目说明（首页显示）
├── LICENSE                   # MIT许可证
├── .gitignore                # Git忽略规则
├── requirements.txt          # Python依赖
├── CHANGELOG.md              # 更新日志
├── code/                     # 源代码
│   ├── README.md
│   ├── core/                 # 核心训练代码
│   └── evaluation/           # 评估脚本
├── docs/                     # 文档资料
│   ├── reports/              # 结题报告
│   ├── slides/               # 答辩PPT
│   └── figures/              # 图表
├── application/              # 申请材料
│   └── patent/               # 专利文件
└── references/               # 参考文献
```

---

## 💡 附加建议

1. **添加 Topics**（在仓库页面右侧）：
   - `open-set-domain-adaptation`
   - `computer-vision`
   - `deep-learning`
   - `pytorch`
   - `feature-disentanglement`

2. **添加 About 描述**：
   > 大创结题项目：基于特征解耦与数据重构的开放集域适应方法研究
   > Dynamic Gradient Reversal + VAE Reconstruction + Four-Source Fusion

3. **启用 Pages**（可选）：
   - Settings → Pages → 选 main 分支
   - 可以把项目文档做成网页展示

---

## 📞 常见问题

**Q: 仓库应该是公开还是私有？**
A: 公开可以让更多人参考（推荐用于学习分享），私有更适合内部存档。

**Q: 文件大小超过限制怎么办？**
A: GitHub单文件限制100MB。我们的项目总大小16MB，完全没问题。

**Q: 怎样把仓库变成Private之后再公开？**
A: Settings → General → Danger Zone → Change repository visibility。
