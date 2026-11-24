# GitHub 部署与本地运行

源码仓库为 `Tao0503/QuantFlow`，公开演示地址为 `https://tao0503.github.io/QuantFlow/`。GitHub Pages 发布 `pages/` 中的静态文件。公开演示提供标的切换、历史汇总对比和研究流程。实际回测、联合调参和研究 Agent 由本地 Python 服务计算。[GitHub Pages 说明](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

## 本地运行

安装 Python 3.12 和 Node.js 22，在克隆的仓库中执行以下命令。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m demo.bootstrap_public
(cd web && npm ci && npm run build)
./start_demo.sh
```

打开 [本地工作台](http://127.0.0.1:8766/)。初始化会生成带标签的合成行情和新的独立研究结果。行情输入、研究产物和 SQLite 工作区保存在个人电脑中。合成行情用于操作演示和测试，历史研究结果见公开演示页。

## 首次发布

仓库维护者在本机通过 [GitHub CLI 官方登录流程](https://cli.github.com/manual/gh_auth_login)登录。创建仓库及推送工作流需要相应权限。

```bash
gh auth login --hostname github.com --git-protocol https --web --scopes workflow
gh api user --jq .login
```

确认命令返回 `Tao0503`，在已经提交本地代码的仓库中执行以下命令。若目标仓库已有提交，先检查远程历史。

```bash
gh repo create Tao0503/QuantFlow --public \
  --description 'ETF strategy research, genetic optimization, and backtesting'
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
  push -u origin main
gh api --method POST repos/Tao0503/QuantFlow/pages -f build_type=workflow
gh workflow run pages.yml --repo Tao0503/QuantFlow --ref main
```

这里假定 `origin` 已配置为 `https://github.com/Tao0503/QuantFlow.git`。也可在仓库的 **Settings → Pages → Source** 选择 **GitHub Actions**，再从 Actions 页面运行 **Publish QuantFlow preview**。后续推送到 `main` 会触发新的展示站部署。[GitHub Pages 发布源说明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

## 发布内容

仓库包含源码、测试、文档和 `pages/` 展示站。原始及缓存行情、完整账本、冻结运行目录、用户 SQLite、虚拟环境、依赖与本机生成文件由 `.gitignore` 排除。研究逻辑的复现来自代码和运行协议，原始历史结果与公开克隆生成的合成结果应分别解读。
