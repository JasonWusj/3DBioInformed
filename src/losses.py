import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        dims = tuple(range(2, logits.ndim))
        intersection = (probs * target).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


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
    ):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.pde_loss = PDELoss3D(d_range, rho_range, sample_parameters)
        self.bc_loss = BoundaryConditionLoss3D()
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc

    def forward(self, logits, target, u_hat, t_tensor):
        dice = self.dice_loss(logits, target)
        pde = self.pde_loss(u_hat, t_tensor)
        bc = self.bc_loss(u_hat)
        total = dice + self.lambda_pde * pde + self.lambda_bc * bc
        return total, {
            "dice": float(dice.detach().cpu()),
            "pde": float(pde.detach().cpu()),
            "bc": float(bc.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
