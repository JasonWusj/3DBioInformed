# 3D 生物物理约束脑肿瘤分割复现

本目录是论文 **Biophysics Informed Pathological Regularisation for Brain Tumour Segmentation** 的 3D 复现工程骨架，独立于已有的 2D 复现代码。

当前实现覆盖论文复现规划中的第 1-8 步：3D BraTS 数据读取、肿瘤中心裁剪、3D UNet、SIREN 肿瘤细胞密度估计器、3D PDE/边界条件正则、训练、滑动窗口推理、TTA 和 case-level Dice/HD95 评估。第 9 步多模型实验矩阵和第 10 步数值验收标准暂未实现。

## 复现假设

- 主复现目标是论文中的 `3D UNet + biophysics-informed regularisation` 主实验。
- BraTS 输出区域按多标签实现为 `background`、`TC`、`WT`、`ET`。这是因为 TC/WT/ET 是嵌套区域，不是互斥 softmax 类别。
- 配置中的 `patch_size` 和 `roi_size` 采用 PyTorch 3D 顺序：`[D, H, W]`。
- `gt_tumor_center` 裁剪用于贴近论文中 “centred on the tumour” 的描述；它依赖标签定位裁剪中心，适合复现实验，但真实无标签推理时需要换成前景中心、脑区中心或检测模型提供的中心。
- 论文 appendix 和官方源码不在本仓库中，因此未公开的增强细节只做了随机三轴翻转。

## 目录结构

```text
3d_bioinformed/
  README.md
  requirement.txt
  smoke_test.py
  configs/
    paper3d_unet.yaml
    baseline3d_unet.yaml
  src/
    __init__.py
    data.py
    model.py
    losses.py
    train3d.py
    train3d_baseline.py
    infer_eval3d.py
```

核心文件说明：

- `configs/paper3d_unet.yaml`：论文主实验配置。
- `configs/baseline3d_unet.yaml`：标准 3D UNet 对照实验配置，不使用 PDE/BC 生物物理约束。
- `src/data.py`：BraTS 3D NIfTI 读取、z-score 标准化、肿瘤中心裁剪、TC/WT/ET 区域 mask 构造。
- `src/model.py`：3D UNet、3D SIREN 密度估计器和完整分割模型。
- `src/losses.py`：Dice loss、3D Fisher-KPP PDE loss、六个面的 Neumann 边界条件 loss。
- `src/train3d.py`：训练入口，保存训练日志、最佳模型、最终模型和周期 checkpoint。
- `src/train3d_baseline.py`：标准 3D UNet baseline 训练入口，只使用 Dice loss。
- `src/infer_eval3d.py`：滑动窗口推理、flip TTA、case-level Dice/HD95 评估。
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

- `TC = label 1 + label 4`
- `WT = label 1 + label 2 + label 4`
- `ET = label 4`
- `background = label 0`

## 环境

本工程只列出依赖，不自动创建环境：

```bash
pip install -r requirement.txt
```

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

默认按顺序运行 baseline 和论文方法，避免单张 GPU 上两个 3D 训练任务同时占满显存。多 GPU 或确认资源足够时可以并行启动：

```bash
bash run_train_compare.sh parallel
```

如果需要指定 Python 解释器：

```bash
PYTHON_BIN=/path/to/python bash run_train_compare.sh
```

默认配置见：

```text
configs/paper3d_unet.yaml
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
python src/infer_eval3d.py --config configs/paper3d_unet.yaml --checkpoint outputs/paper3d_unet/final_model.pth
```

评估标准 baseline：

```bash
python src/infer_eval3d.py --config configs/baseline3d_unet.yaml --checkpoint outputs/baseline3d_unet/final_model.pth
```

默认评估 test split。也可以评估 validation split：

```bash
python src/infer_eval3d.py --config configs/paper3d_unet.yaml --split val
```

评估内容：

- 使用 MONAI `sliding_window_inference`
- 使用三轴 flip TTA
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

- R2-UNet、nn-UNet、UNETR、SegResNet、SegResNetVAE 的完整实验矩阵
- Sine/ReLU、with/without BC、缺失模态、训练集比例、不同 segmentation loss 的系统消融
- 与论文表格数值接近程度的验收实验

## 重要注意事项

- 当前代码是 3D 复现工程，不是 2D 代码的直接替换。
- 真实无标签测试时不应使用 `gt_tumor_center`，否则裁剪中心依赖标签。
- 如果要严格比较论文数值，需要固定随机种子、数据划分、裁剪策略、增强策略、模型配置和最终模型选择方式。
- 论文未公开的 appendix 细节会影响完全数值复现，需要在实验日志中明确记录本工程采用的替代实现。
