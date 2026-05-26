import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, skip_bg=True):
        super().__init__()
        self.smooth = smooth
        self.skip_bg = skip_bg

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        if self.skip_bg:
            probs = probs[:, 1:]
            target = target[:, 1:]
        dims = tuple(range(2, logits.ndim))
        intersection = (probs * target).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class DiceWithBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        return self.dice(logits, target) + self.bce(logits[:, 1:], target[:, 1:])


class HierarchyConsistencyLoss(nn.Module):
    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        tc = probs[:, 1]
        wt = probs[:, 2]
        et = probs[:, 3]
        return F.relu(et - tc).mean() + F.relu(tc - wt).mean()


class StructureAwareDiceBCELoss(nn.Module):
    def __init__(self, lambda_hierarchy=0.2, lambda_boundary=0.5, boundary_width=1):
        super().__init__()
        self.dice = DiceLoss()
        self.hierarchy = HierarchyConsistencyLoss()
        self.lambda_hierarchy = float(lambda_hierarchy)
        self.lambda_boundary = float(lambda_boundary)
        self.boundary_width = int(boundary_width)

    def _boundary_weights(self, target):
        target = target[:, 1:].float()
        kernel_size = 2 * self.boundary_width + 1
        dilation = F.max_pool3d(target, kernel_size=kernel_size, stride=1, padding=self.boundary_width)
        erosion = 1.0 - F.max_pool3d(1.0 - target, kernel_size=kernel_size, stride=1, padding=self.boundary_width)
        boundary = (dilation - erosion).clamp(0.0, 1.0)
        return 1.0 + self.lambda_boundary * boundary

    def forward(self, logits, target):
        dice = self.dice(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits[:, 1:], target[:, 1:], reduction="none")
        weighted_bce = (bce * self._boundary_weights(target)).mean()
        hierarchy = self.hierarchy(logits, target)
        return dice + weighted_bce + self.lambda_hierarchy * hierarchy


class GaussianRefinementRegularization(nn.Module):
    def __init__(self, lambda_sigma=1.0e-4, lambda_amplitude=1.0e-3):
        super().__init__()
        self.lambda_sigma = float(lambda_sigma)
        self.lambda_amplitude = float(lambda_amplitude)

    def forward(self, gaussian_aux):
        sigma = gaussian_aux["sigma"]
        amplitude = gaussian_aux["amplitude"]
        sigma_penalty = (1.0 / (sigma.square() + 1.0e-6)).mean()
        amplitude_penalty = amplitude.abs().mean()
        return self.lambda_sigma * sigma_penalty + self.lambda_amplitude * amplitude_penalty


class DensityCouplingLoss(nn.Module):
    def __init__(self, region_channel=2, smooth=1.0):
        super().__init__()
        self.region_channel = int(region_channel)
        self.smooth = float(smooth)

    def forward(self, u_hat, target):
        density_target = target[:, self.region_channel : self.region_channel + 1].float()
        if density_target.shape[2:] != u_hat.shape[2:]:
            density_target = F.interpolate(density_target, size=u_hat.shape[2:], mode="trilinear", align_corners=False)
        density_target = density_target.clamp(0.0, 1.0)
        u_hat = u_hat.clamp(1.0e-6, 1.0 - 1.0e-6)

        dims = tuple(range(2, u_hat.ndim))
        intersection = (u_hat * density_target).sum(dim=dims)
        denominator = u_hat.sum(dim=dims) + density_target.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        bce = F.binary_cross_entropy(u_hat, density_target)
        return dice + bce


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        logits = logits[:, 1:]
        target = target[:, 1:]
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(target > 0.5, self.alpha, 1.0 - self.alpha)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


class JaccardLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)[:, 1:]
        target = target[:, 1:]
        dims = tuple(range(2, logits.ndim))
        intersection = (probs * target).sum(dim=dims)
        union = probs.sum(dim=dims) + target.sum(dim=dims) - intersection
        score = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - score.mean()


class PDELoss3D(nn.Module):
    def __init__(self, d_range=(0.02, 1.5), rho_range=(0.002, 0.2), sample_parameters="voxel"):
        super().__init__()
        self.d_range = tuple(d_range)
        self.rho_range = tuple(rho_range)
        self.sample_parameters = sample_parameters

        kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
        kernel[0, 0, 1, 1, 1] = -6.0
        kernel[0, 0, 0, 1, 1] = 1.0
        kernel[0, 0, 2, 1, 1] = 1.0
        kernel[0, 0, 1, 0, 1] = 1.0
        kernel[0, 0, 1, 2, 1] = 1.0
        kernel[0, 0, 1, 1, 0] = 1.0
        kernel[0, 0, 1, 1, 2] = 1.0
        self.register_buffer("laplacian_kernel", kernel)

    def _sample_parameter(self, shape, value_range, device, dtype):
        if self.sample_parameters == "batch":
            shape = (shape[0], 1, 1, 1, 1)
        return torch.empty(shape, device=device, dtype=dtype).uniform_(*value_range)

    @staticmethod
    def time_derivative(u_hat, t_tensor):
        u_flat = u_hat.flatten(2).permute(0, 2, 1)
        du_dt = torch.autograd.grad(
            outputs=u_flat,
            inputs=t_tensor,
            grad_outputs=torch.ones_like(u_flat),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return du_dt.permute(0, 2, 1).reshape_as(u_hat)

    def forward(self, u_hat, t_tensor):
        d = self._sample_parameter(u_hat.shape, self.d_range, u_hat.device, u_hat.dtype)
        rho = self._sample_parameter(u_hat.shape, self.rho_range, u_hat.device, u_hat.dtype)

        kernel = self.laplacian_kernel.to(dtype=u_hat.dtype)
        laplacian_u = F.conv3d(u_hat, kernel, padding=1)
        diffusion = d * laplacian_u
        reaction = rho * u_hat * (1.0 - u_hat)
        du_dt = self.time_derivative(u_hat, t_tensor)
        residual = du_dt - diffusion - reaction
        return (residual ** 2).mean()


class BoundaryConditionLoss3D(nn.Module):
    def forward(self, u_hat):
        grad_d0 = u_hat[:, :, 1, :, :] - u_hat[:, :, 0, :, :]
        grad_d1 = u_hat[:, :, -1, :, :] - u_hat[:, :, -2, :, :]
        grad_h0 = u_hat[:, :, :, 1, :] - u_hat[:, :, :, 0, :]
        grad_h1 = u_hat[:, :, :, -1, :] - u_hat[:, :, :, -2, :]
        grad_w0 = u_hat[:, :, :, :, 1] - u_hat[:, :, :, :, 0]
        grad_w1 = u_hat[:, :, :, :, -1] - u_hat[:, :, :, :, -2]

        return (
            grad_d0.square().mean()
            + grad_d1.square().mean()
            + grad_h0.square().mean()
            + grad_h1.square().mean()
            + grad_w0.square().mean()
            + grad_w1.square().mean()
        ) / 6.0


class BiophysicsInformedLoss3D(nn.Module):
    def __init__(
        self,
        lambda_pde=1.0,
        lambda_bc=1.0,
        d_range=(0.02, 1.5),
        rho_range=(0.002, 0.2),
        sample_parameters="voxel",
        segmentation_loss=None,
        lambda_density=0.0,
        density_region_channel=2,
    ):
        super().__init__()
        self.segmentation_loss = segmentation_loss if segmentation_loss is not None else DiceLoss()
        self.pde_loss = PDELoss3D(d_range, rho_range, sample_parameters)
        self.bc_loss = BoundaryConditionLoss3D()
        self.density_loss = DensityCouplingLoss(region_channel=density_region_channel)
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_density = lambda_density

    def forward(self, logits, target, u_hat, t_tensor):
        seg = self.segmentation_loss(logits, target)
        pde = self.pde_loss(u_hat, t_tensor)
        bc = self.bc_loss(u_hat)
        density = self.density_loss(u_hat, target)
        total = seg + self.lambda_pde * pde + self.lambda_bc * bc + self.lambda_density * density
        return total, {
            "seg": float(seg.detach().cpu()),
            "pde": float(pde.detach().cpu()),
            "bc": float(bc.detach().cpu()),
            "density": float(density.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
