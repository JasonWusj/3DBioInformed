# UNet 复现实验代码改动说明

本文档记录当前本地 `dev` 分支相对 `origin/dev` 的主要代码改动、运行方式和已验证结果。

当前本地提交：

```text
98e0eda Add UNet ablation experiment support
```

远端 `origin/dev` 仍停留在：

```text
986d3f5 Track full paper experiment matrix
```

也就是说，本文档描述的改动目前已经在本地 `dev` 提交，但尚未成功推送到 GitHub。

## 改了什么

### 1. 补齐 UNet 范围内的论文实验配置

新增目录：

```text
configs/unet_ablations/
```

其中包含 UNet 的 biophysics 和 baseline 消融配置，覆盖：

- Sine/ReLU density estimator activation
- with/without boundary condition
- 2 模态和 3 模态输入
- 25%、50%、75% 训练集比例
- Dice+CE、Focal、Jaccard segmentation loss
- Fig. 2 需要的 with/without biophysics 对照

主实验配置也补了显式字段：

```yaml
model:
  density_estimator:
    activation: sine

loss:
  segmentation_loss: dice
```

### 2. 更新论文实验矩阵

更新文件：

```text
configs/paper_experiment_matrix.yaml
```

主要变化：

- 继续保留 Table 1 的完整方法清单。
- UNet baseline 和 UNet Biophy 标记为已实现。
- R2-UNet、nn-UNet、UNETR、SegResNet、SegResNetVAE 仍标记为 missing，因为用户当前要求先只做 UNet。
- 新增 `unet_ablations`，记录所有已实现的 UNet 消融配置。
- Fig. 2 的消融维度标记为 `unet_configs_implemented_results_pending`，表示配置已补齐，但完整长训练和数值汇总尚未执行。

### 3. 训练代码支持 UNet 消融开关

更新文件：

```text
src/train3d.py
src/losses.py
src/model.py
```

新增能力：

- `data.train_fraction`：支持训练集比例消融。
- `loss.segmentation_loss`：支持 `dice`、`dice_ce`、`focal`、`jaccard`。
- `model.density_estimator.activation`：支持 `sine` 和 `relu`。
- biophysics loss 中的 segmentation loss 可注入，不再硬编码为 Dice。
- 训练日志字段从 `train_dice` 改为 `train_seg_loss`，避免 Focal/Jaccard 消融时日志名误导。
- 验证阶段仍固定使用真正的 `DiceLoss` 计算 `val_dice_loss`，并保留 `val_mean_dice` 作为模型选择指标。

### 4. 新增 UNet 实验矩阵运行入口

新增文件：

```text
run_unet_experiment_matrix.py
run_unet_experiment_matrix.sh
```

推荐使用 Python 入口：

```bash
python run_unet_experiment_matrix.py
```

原因是当前 Windows 环境中 WSL bash 和 Git Bash 都存在 Win32 权限错误，Python 入口不依赖 bash。

Python runner 会从：

```text
configs/paper_experiment_matrix.yaml
```

读取当前已实现的 UNet 主实验和消融配置，然后按顺序执行：

```text
src/train3d.py
src/evaluate.py
```

### 5. 增加回归测试

新增/更新测试：

```text
tests/test_unet_ablation_support.py
tests/test_paper_experiment_matrix.py
tests/test_paper_reproduction_config.py
```

覆盖内容：

- UNet 消融配置文件存在。
- 所有消融配置仍使用 `unet3d` backbone。
- baseline/biophysics 配置的 `use_biophysics` 与文件名一致。
- 训练代码支持 activation、segmentation loss、train fraction。
- `paper_experiment_matrix.yaml` 中 UNet 消融配置均指向存在的文件。
- Python 和 shell runner 包含训练、评估和 dry-run 入口。

## 如何运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或：

```bash
pip install -r requirement.txt
```

### 2. 快速代码验证

运行单元测试：

```bash
python -B -m pytest -q
```

运行随机张量 smoke test：

```bash
python -B smoke_test.py
```

检查 UNet 实验矩阵将要执行哪些命令，不启动训练：

```bash
python -B run_unet_experiment_matrix.py --dry-run --skip-preprocess
```

### 3. 运行主实验

论文 biophysics UNet：

```bash
python src/train3d.py --config configs/paper3d_unet.yaml
python src/evaluate.py --config configs/paper3d_unet.yaml --checkpoint outputs/paper3d_unet/final_model.pth --split test --mode auto
```

标准 UNet baseline：

```bash
python src/train3d.py --config configs/baseline3d_unet.yaml
python src/evaluate.py --config configs/baseline3d_unet.yaml --checkpoint outputs/baseline3d_unet/final_model.pth --split test --mode auto
```

### 4. 运行完整 UNet 矩阵

按当前已实现的 UNet 主实验和消融矩阵顺序运行：

```bash
python run_unet_experiment_matrix.py
```

常用参数：

```bash
python run_unet_experiment_matrix.py --split test --mode auto
python run_unet_experiment_matrix.py --split both --mode auto
python run_unet_experiment_matrix.py --skip-preprocess
python run_unet_experiment_matrix.py --python D:\Anaconda\python.exe
```

默认行为：

- 先运行 `src/preprocess3d.py --config configs/paper3d_unet.yaml`
- 逐个训练 UNet 主实验和消融配置
- 每个配置训练结束后评估 `final_model.pth`
- 日志写入 `logs/`

## 当前验证结果

本地已执行并通过：

```text
python -B -m pytest -q
15 passed
```

```text
python -B smoke_test.py
smoke_test passed
```

```text
python -B run_unet_experiment_matrix.py --dry-run --skip-preprocess
成功列出完整 UNet 主实验和消融训练/评估队列
```

Python 语法检查也已通过：

```text
syntax ok
```

## 当前未完成项和限制

### 1. 尚未执行完整长训练

目前完成的是代码、配置和运行入口层面的复现准备。由于完整训练需要 BraTS 2023 数据和较长 GPU 时间，尚未得到新的论文数值复现实验结果。

因此当前状态是：

```text
UNet 配置和运行入口已实现
UNet 长训练结果和数值汇总待执行
非 UNet backbone 按用户要求暂未实现
```

### 2. 非 UNet backbone 暂未实现

以下模型仍未实现：

- R2-UNet
- nn-UNet
- UNETR
- SegResNet
- SegResNetVAE

这是符合当前用户范围的：先在 UNet 上实现。

### 3. 尚未成功推送

多次推送失败，失败点在 GitHub `git-receive-pack` 上传阶段。

已确认：

- 本地认证可进入 push 流程。
- `git ls-remote origin refs/heads/dev` 可读取远端。
- 本地 `dev` 领先 `origin/dev` 1 个提交。
- 上传包很小，trace 中 `Content-Length: 11616`。
- HTTP/2 和 HTTP/1.1 push 均被连接重置。

当前状态：

```text
本地 dev:    98e0eda Add UNet ablation experiment support
origin/dev: 986d3f5 Track full paper experiment matrix
```

## 结论

本轮代码改动已经把论文复现范围中当前指定的 UNet 主实验和 UNet Fig. 2 消融配置补齐，并提供了可审查、可执行的矩阵 runner。代码级验证通过，但完整论文数值复现还需要实际运行长训练；GitHub 推送仍受当前网络/远端 receive-pack 连接重置阻断。
