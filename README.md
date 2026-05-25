# 3D 生物物理约束脑肿瘤分割复现

本目录是论文 **Biophysics Informed Pathological Regularisation for Brain Tumour Segmentation** 的 3D 复现工程骨架，独立于已有的 2D 复现代码。

当前实现覆盖论文复现规划中的第 1-8 步：3D BraTS 数据读取、肿瘤中心裁剪、3D UNet、SIREN 肿瘤细胞密度估计器、3D PDE/边界条件正则、训练、滑动窗口推理、TTA 和 case-level Dice/HD95 评估。第 9 步多模型实验矩阵和第 10 步数值验收标准暂未实现。

## 复现假设

- 主复现目标是论文中的 `3D UNet + biophysics-informed regularisation` 主实验。
- BraTS 输出区域按多标签实现为 `background`、`TC`、`WT`、`ET`。这是因为 TC/WT/ET 是嵌套区域，不是互斥 softmax 类别。
- 配置中的 `patch_size` 和 `roi_size` 采用 PyTorch 3D 顺序：`[D, H, W]`。
- 训练和离线预处理使用 `gt_tumor_center` 裁剪来贴近论文中 “centred on the tumour” 的描述；raw 评估默认使用完整体积加滑动窗口，不使用标签中心裁剪。
- `data.split_seed` 固定病例级 7:1:2 划分；多 run 只偏移训练随机种子，避免不同 run 使用不同 test split。
- 论文 appendix 和官方源码不在本仓库中，因此未公开的增强细节只做了随机三轴翻转。

## 目录结构

```text
3d_bioinformed/
  README.md
  requirement.txt
  requirements.txt
  smoke_test.py
  configs/
    paper3d_unet.yaml
    baseline3d_unet.yaml
  src/
    __init__.py
    data.py
    preprocess3d.py
    model.py
    losses.py
    train3d.py
    train3d_baseline.py
    evaluate.py
    infer_eval3d.py
```

核心文件说明：

- `configs/paper3d_unet.yaml`：论文主实验配置。
- `configs/baseline3d_unet.yaml`：标准 3D UNet 对照实验配置，不使用 PDE/BC 生物物理约束。
- `configs/paper_experiment_matrix.yaml`：论文 Table 1 和 Fig. 2 消融的机器可读复现实验矩阵，明确已实现与缺失项。
- `src/data.py`：BraTS 3D NIfTI 读取、z-score 标准化、肿瘤中心裁剪、TC/WT/ET 区域 mask 构造。
- `src/preprocess3d.py`：一次性把 NIfTI 转成离线预处理后的 3D `.npy` patch，减少训练时 CPU I/O 和解压开销。
- `src/model.py`：3D UNet、3D SIREN 密度估计器和完整分割模型。
- `src/losses.py`：Dice loss、3D Fisher-KPP PDE loss、六个面的 Neumann 边界条件 loss。
- `src/train3d.py`：训练入口，保存训练日志、最佳模型、最终模型和周期 checkpoint。
- `src/train3d_baseline.py`：标准 3D UNet baseline 训练入口，只使用 Dice loss。
- `src/evaluate.py`：推荐评估入口，支持预处理 patch 评估和 raw NIfTI 滑动窗口 + flip TTA 评估，输出 case-level Dice/HD95。
- `src/infer_eval3d.py`：旧评估入口，保留用于兼容。
- `smoke_test.py`：随机张量快速验证，不依赖真实数据。

## 数据格式

期望 BraTS 2023 数据按如下方式放置：

```text
data/BraTS2023/
  BraTS-GLI-00000-000/
    BraTS-GLI-00000-000-t1n.nii.gz
    BraTS-GLI-00000-000-t1c.nii.gz
    BraTS-GLI-00000-000-t2w.nii.gz
    BraTS-GLI-00000-000-t2f.nii.gz
    BraTS-GLI-00000-000-seg.nii.gz
```

默认使用 4 个模态：

- `t1n`
- `t1c`
- `t2w`
- `t2f`

标签按 BraTS 原始标注转换为区域 mask：

- `TC = label 1 + label 3`
- `WT = label 1 + label 2 + label 3`
- `ET = label 3`
- `background = label 0`

## 环境

本工程只列出依赖，不自动创建环境。推荐使用标准文件名 `requirements.txt` 安装：

```bash
pip install -r requirements.txt
```

仓库中也保留了 `requirement.txt` 作为兼容文件，因此下面命令同样可用：

```bash
pip install -r requirement.txt
```

不要运行 `pip install requirement.txt`，那会把 `requirement.txt` 当成一个包名。也不要在 zsh/bash 中直接逐行执行 `torch>=2.1.0` 这类内容，`>` 会被 shell 当成重定向符号。

论文报告的主要环境和训练设置：

- MONAI 1.3.0
- PyTorch 2.1.0
- Nvidia A10 24GB
- AMP 混合精度训练
- batch size 1
- 175 epochs
- Ranger 2020 optimizer
- 初始学习率 `3e-4`
- cosine decay

## 快速验证

`smoke_test.py` 使用随机数据验证模型前向、密度估计器、时间导数自动微分、3D PDE loss 和边界条件 loss。

```bash
python smoke_test.py
```

通过时会输出类似：

```text
smoke_test passed
{'dice': ..., 'pde': ..., 'bc': ..., 'total': ...}
```

## 训练

### 推荐：先生成 3D `.npy` patch

直接从 `.nii.gz` 训练会在每个 epoch 重复做解压、整例 z-score 和裁剪，CPU 很容易成为瓶颈。默认配置已启用：

```yaml
data:
  use_preprocessed: true
  preprocessed_dir: ./data/preprocessed3d_patches
```

因此首次训练前建议先运行离线预处理：

```bash
python src/preprocess3d.py --config configs/paper3d_unet.yaml
```

该步骤不会改变训练样本内容，只是把同样的归一化、肿瘤中心裁剪和 TC/WT/ET mask 构造提前保存成 `.npy`。训练阶段仍会在线执行随机三轴翻转增强。

预处理完成后，训练不再依赖原始 `data/BraTS2023` 目录；病例列表会从 `data/preprocessed3d_patches/metadata.npy` 读取。

如果你确实想每次从原始 NIfTI 读取，把配置改成：

```yaml
data:
  use_preprocessed: false
```

### 单独训练

训练论文方法：

```bash
python src/train3d.py --config configs/paper3d_unet.yaml
```

训练标准 3D UNet baseline：

```bash
python src/train3d_baseline.py
```

等价于：

```bash
python src/train3d.py --config configs/baseline3d_unet.yaml
```

Linux 下也可以一键启动两组对比训练：

```bash
bash run_train_compare.sh
```

默认会先运行一次 `src/preprocess3d.py` 准备 `.npy` patch，然后按顺序运行 baseline 和论文方法，避免单张 GPU 上两个 3D 训练任务同时占满显存。多 GPU 或确认资源足够时可以并行启动：

```bash
bash run_train_compare.sh parallel
```

如果需要指定 Python 解释器：

```bash
PYTHON_BIN=/path/to/python bash run_train_compare.sh
```

如果 `.npy` patch 已经准备好，想跳过预处理：

```bash
AUTO_PREPROCESS=0 bash run_train_compare.sh
```

### 数据加载加速配置

默认 DataLoader 配置会启用 worker 预取和持久 worker：

```yaml
data:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4
```

这些设置不改变实验样本和模型计算，只减少 GPU 等 CPU 数据的时间。若 CPU 内存紧张，可以把 `prefetch_factor` 降到 `2`，或减少 `num_workers`。

预处理后的 `.npy` 大约需要几十 GB 磁盘空间，取决于病例数和 patch 大小。当前实现为了不改变实验数值，图像保存为 `float32`，标签保存为二值 `uint8`。

默认配置见：

```text
configs/paper3d_unet.yaml
```

完整论文实验矩阵见：

```text
configs/paper_experiment_matrix.yaml
```

训练输出保存在 `output_dir` 指定目录，默认是：

```text
outputs/paper3d_unet/
```

训练脚本会保存：

- `best_model.pth`
- `final_model.pth`
- `checkpoint_epoch*.pth`
- `training_log.csv`
- `config_used.yaml`

注意：论文说明测试使用 final model，因此评估脚本默认使用 `final_model.pth`。`best_model.pth` 仍然保留，便于调试和比较。

## 评估

评估论文方法：

```bash
python src/evaluate.py --config configs/paper3d_unet.yaml --checkpoint outputs/paper3d_unet/final_model.pth --split test --mode auto
```

评估标准 baseline：

```bash
python src/evaluate.py --config configs/baseline3d_unet.yaml --checkpoint outputs/baseline3d_unet/final_model.pth --split test --mode auto
```

默认评估 test split。也可以评估 validation split：

```bash
python src/evaluate.py --config configs/paper3d_unet.yaml --split val
```

评估内容：

- 使用 MONAI `sliding_window_inference`
- 使用三轴 flip TTA
- raw NIfTI 模式默认在完整体积上评估；预处理模式评估离线保存的 `128x128x128` patch
- 输出 TC、WT、ET 的 case-level Dice
- 输出 TC、WT、ET 的 HD95，单位为毫米
- 保存逐 case 指标到 `test_metrics.csv` 或 `val_metrics.csv`

预测区域会在评估前强制嵌套：

```text
ET subset TC subset WT
```

## 当前实现边界

已实现：

1. 主实验配置：`3D UNet + biophysics-informed regularisation`
2. BraTS 2023 3D 数据读取和病例级 `7:1:2` 划分
3. `128x128x128` 肿瘤中心 patch 裁剪
4. 3D UNet backbone
5. 带时间输入的 3D SIREN 肿瘤细胞密度估计器
6. 3D Laplacian PDE loss 和六面 Neumann BC loss
7. 论文风格训练循环：AMP、Ranger、cosine、batch size 1、175 epochs
8. 滑动窗口推理、TTA 和 case-level Dice/HD95

暂未实现：

- R2-UNet、nn-UNet、UNETR、SegResNet、SegResNetVAE 的模型/配置实现
- Sine/ReLU、with/without BC、缺失模态、训练集比例、不同 segmentation loss 的系统消融执行
- 与论文表格数值接近程度的验收实验

## 重要注意事项

- 当前代码是 3D 复现工程，不是 2D 代码的直接替换。
- 真实无标签测试时不应使用 `gt_tumor_center`，否则裁剪中心依赖标签。
- 如果要严格比较论文数值，需要固定随机种子、数据划分、裁剪策略、增强策略、模型配置和最终模型选择方式。
- 论文未公开的 appendix 细节会影响完全数值复现，需要在实验日志中明确记录本工程采用的替代实现。
