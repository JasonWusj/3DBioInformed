# better 分支方法改进说明

## 改进方向

这个分支聚焦于方法层面的改进，而不是服务器路径适配或实验脚本整理。目标仍然是当前项目中的 UNet 版 paper3d 复现。

原始 paper3d 目标函数使用 Dice 监督分割结果，并加入 biophysics PDE / BC 正则。这个设计保留了肿瘤生长动力学先验，但还没有显式利用两个对 BraTS 分割很重要的结构信息：

- ET 应该包含在 TC 内。
- TC 应该包含在 WT 内。
- 肿瘤边界是 3D 不平衡分割中误差高、价值也高的区域。
- 肿瘤 mask 不只是独立 voxel label，而可以看作连续空间概率场在 voxel 网格上的采样。

## 已实现改动

`configs/better3d_unet.yaml` 保留 paper3d 的 UNet + density estimator，同时增加 Gaussian implicit segmentation refinement branch：

```yaml
model:
  gaussian_refinement:
    enabled: true
    num_gaussians: 48
    min_sigma: 0.03
    max_sigma: 0.5
    alpha_init: 0.2

loss:
  segmentation_loss: structure_aware_dice_ce
  lambda_hierarchy: 0.2
  lambda_boundary: 0.5
  boundary_width: 1
  lambda_gaussian_sigma: 1.0e-4
  lambda_gaussian_amplitude: 1.0e-3
```

模型结构是：

```text
voxel_logits = UNet(x)
gaussian_logits = sum_i a_i * G_i(x)
final_logits = voxel_logits + alpha * gaussian_logits
```

Gaussian 分支从 UNet bottleneck features 预测前景类别的 Gaussian 参数。它不会替换 voxel segmentation head，而是给原始 logits 增加一个连续空间 refinement 项。

分割监督项为：

```text
Dice + boundary-weighted BCE + lambda_hierarchy * hierarchy penalty
```

层级一致性惩罚为：

```text
mean(ReLU(P(ET) - P(TC))) + mean(ReLU(P(TC) - P(WT)))
```

这个约束会惩罚 ET 出现在 TC 外、TC 出现在 WT 外的无效预测。

边界加权 BCE 使用 3D max-pooling 对 target mask 做膨胀/腐蚀，构造窄边界带，并提高这些边界 voxel 的 BCE 权重。

Gaussian 参数正则用于避免退化解：

```text
lambda_gaussian_sigma * mean(1 / sigma^2) + lambda_gaussian_amplitude * mean(|a|)
```

其中 `sigma` 正则抑制过尖的 Gaussian，`amplitude` 正则抑制过大的振幅。

## 相关论文依据

- Kervadec et al., "Boundary loss for highly unbalanced segmentation": https://arxiv.org/abs/1812.07032
- Shit et al., "clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation": https://arxiv.org/abs/2003.07311

这里不是直接复现上述论文，而是把它们的边界监督、拓扑/结构约束思想迁移到当前 BraTS 区域 mask 设定中。同时，Gaussian continuous field 的设计借鉴了 GSR 中“用连续函数表示待重建物理场”的思想。

## 运行方式

```bash
python src/train3d.py --config configs/better3d_unet.yaml
python src/evaluate.py --config configs/better3d_unet.yaml --checkpoint outputs/better3d_unet/best_model.pth
```

如果服务器上的预处理数据挂载在绝对路径，例如 `/data/preprocessed3d_patches`，可以修改 `configs/better3d_unet.yaml` 中的 `data.preprocessed_dir`，或通过你的实验 runner 覆盖该路径。

## 预期优势与风险

预期优势：

- 减少不满足 BraTS 区域层级关系的预测。
- 增强 TC / WT / ET 边界附近的学习信号。
- Gaussian refinement branch 用连续空间函数修正 logits，有助于减少碎片化前景岛。
- 与当前 biophysics loss 兼容，因为它保留原始 UNet voxel logits，只在其上叠加连续表示 refinement。

潜在风险：

- 对肿瘤边界附近的标签噪声更敏感。
- Gaussian field 在输出网格上采样，会比 plain UNet 增加显存和计算量。
- 新增权重需要通过消融实验确认，不能直接当作最终论文结论。
- 当前实现假设通道顺序保持为 `[background, TC, WT, ET]`。
