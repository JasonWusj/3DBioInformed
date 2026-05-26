# Better UNet 方法改进文档

## 动机

原始 paper3d 实现使用 Dice 分割监督，并加入 biophysics PDE 和边界条件正则。这个设计保留了物理先验，但分割项本身没有显式编码 BraTS 肿瘤区域中的两个重要性质：

- 区域层级：ET 应该包含在 TC 内，TC 应该包含在 WT 内。
- 边界难点：Dice 主要优化区域重叠，但在高度不平衡的 3D 肿瘤分割中，少量边界错误会明显影响视觉质量、临床可用性和 HD95。
- 连续表示：肿瘤 mask 可以看作连续空间概率场在 voxel 网格上的采样，而不是互相独立的 voxel 分类结果。

因此，better UNet 实验在保持原始 UNet backbone 和 biophysics branch 的基础上，加入了 Gaussian implicit segmentation refinement branch 和结构感知分割损失。

根据当前实验观察，取消 paper3d 中的 density/PDE 路径后效果不好，甚至可能低于 baseline。因此本分支把 density 视为必须保留的核心组件，并恢复 `DensityCouplingLoss`：用肿瘤区域 mask 约束 `u_hat`，让 density estimator 学到和分割目标一致的隐式肿瘤密度场。

## 方法

新配置文件：

```bash
python src/train3d.py --config configs/better3d_unet.yaml
python src/evaluate.py --config configs/better3d_unet.yaml --checkpoint outputs/better3d_unet/best_model.pth
```

模型保留 UNet voxel head，并额外加入连续 Gaussian field：

```text
z_voxel = UNet(x)
z_gaussian = sum_i a_i * G_i(x)
z_final = z_voxel + alpha * z_gaussian
```

Gaussian 参数由 UNet bottleneck features 预测。这个分支是 refinement path，不是 voxel logits 的替代品。

新的分割损失为：

```text
L = Dice + boundary_weighted_BCE + lambda_hierarchy * hierarchy_penalty
```

其中 `hierarchy_penalty` 是可微的区域嵌套约束：

```text
mean(ReLU(P(ET) - P(TC))) + mean(ReLU(P(TC) - P(WT)))
```

`boundary_weighted_BCE` 使用 target mask 的 3D max-pooling 膨胀/腐蚀构造边界带，并提高边界 voxel 附近的 BCE 权重。

训练 loop 还加入了一个轻量 Gaussian 参数正则：

```text
lambda_gaussian_sigma * mean(1 / sigma^2) + lambda_gaussian_amplitude * mean(|a|)
```

这个正则用于抑制过尖 Gaussian 和失控的 amplitude。

## 文献依据

- Kervadec et al., "Boundary loss for highly unbalanced segmentation"：说明在不平衡医学图像分割中，仅使用区域重叠损失不够，边界相关监督可以提供互补信号。https://arxiv.org/abs/1812.07032
- Shit et al., "clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"：说明拓扑/形状约束可以补充 voxel overlap loss。https://arxiv.org/abs/2003.07311
- BraTS 区域定义天然具有嵌套结构：ET inside TC inside WT。
- GSR 的启发是把待重建对象表示成连续空间函数。因此在分割中，可以把 mask 看作隐式空间概率场，而不是只看作离散 voxel label。

## 预期效果

相对于当前 paper3d UNet，潜在优势包括：

- 减少 ET 出现在 TC 外、TC 出现在 WT 外的无效预测。
- 对肿瘤边界施加更强监督，尤其适用于 Dice 对小 ET / TC 区域不敏感的情况。
- Gaussian continuous field 可以鼓励局部空间连续性，减少碎片化前景岛。
- 与现有 biophysics-informed objective 兼容，因为它只在原始 voxel logits 上叠加 refinement。

潜在风险包括：

- BCE 和边界强调可能对粗糙或不一致标签更敏感。
- Gaussian field 采样会增加显存和计算开销。
- 层级约束假设通道顺序始终是 `[background, TC, WT, ET]`。
- `lambda_hierarchy`、`lambda_boundary`、`boundary_width`、`lambda_gaussian_sigma` 和 `lambda_gaussian_amplitude` 都需要后续消融验证，不能直接视为通用最优参数。
