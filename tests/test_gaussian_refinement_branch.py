import unittest
from pathlib import Path

import torch
import yaml

from src.losses import BiophysicsInformedLoss3D, GaussianRefinementRegularization
from src.model import BiophysicsSegModel3D, GaussianSegRefinementHead3D


ROOT = Path(__file__).resolve().parents[1]


def tiny_cfg():
    return {
        "data": {
            "num_channels": 4,
            "num_classes": 4,
        },
        "model": {
            "features": [4, 8],
            "density_estimator": {
                "hidden_dim": 16,
                "num_layers": 2,
                "feature_size": [4, 4, 4],
            },
            "gaussian_refinement": {
                "enabled": True,
                "num_gaussians": 5,
                "hidden_dim": 16,
                "min_sigma": 0.05,
                "max_sigma": 0.6,
                "alpha_init": 0.25,
            },
        },
    }


class GaussianRefinementBranchTest(unittest.TestCase):
    def test_gaussian_head_refines_logits_without_replacing_voxel_logits(self):
        head = GaussianSegRefinementHead3D(
            in_channels=16,
            num_classes=4,
            num_gaussians=5,
            hidden_dim=16,
            min_sigma=0.05,
            max_sigma=0.6,
            alpha_init=0.25,
        )
        base_logits = torch.zeros(2, 4, 8, 8, 8)
        features = torch.randn(2, 16, 2, 2, 2)

        refined_logits, aux = head(base_logits, features)

        self.assertEqual(refined_logits.shape, base_logits.shape)
        self.assertEqual(aux["mu"].shape, (2, 3, 5, 3))
        self.assertEqual(aux["sigma"].shape, (2, 3, 5, 3))
        self.assertEqual(aux["amplitude"].shape, (2, 3, 5))
        self.assertGreaterEqual(float(aux["sigma"].detach().min()), 0.05)
        self.assertLessEqual(float(aux["sigma"].detach().max()), 0.6)
        self.assertFalse(torch.allclose(refined_logits[:, 1:], base_logits[:, 1:]))

    def test_biophysics_model_returns_gaussian_aux_when_enabled(self):
        model = BiophysicsSegModel3D(tiny_cfg())
        image = torch.randn(1, 4, 16, 16, 16)

        logits, u_hat, t_tensor, gaussian_aux = model(image, return_density=True)
        inference_logits = model(image, return_density=False)

        self.assertEqual(logits.shape, (1, 4, 16, 16, 16))
        self.assertEqual(inference_logits.shape, logits.shape)
        self.assertEqual(u_hat.shape, (1, 1, 4, 4, 4))
        self.assertIn("sigma", gaussian_aux)
        self.assertTrue(t_tensor.requires_grad)

    def test_gaussian_refinement_branch_participates_in_backprop(self):
        model = BiophysicsSegModel3D(tiny_cfg())
        image = torch.randn(1, 4, 16, 16, 16)
        target = torch.zeros(1, 4, 16, 16, 16)
        target[:, 2, 5:11, 5:11, 5:11] = 1.0
        target[:, 1, 6:10, 6:10, 6:10] = 1.0
        target[:, 3, 7:9, 7:9, 7:9] = 1.0
        target[:, 0] = 1.0 - target[:, 2]

        logits, u_hat, t_tensor, gaussian_aux = model(image, return_density=True)
        loss, _ = BiophysicsInformedLoss3D()(logits, target, u_hat, t_tensor)
        loss = loss + GaussianRefinementRegularization()(gaussian_aux)
        loss.backward()

        scale_grad = model.gaussian_refinement.refinement_scale.grad
        self.assertIsNotNone(scale_grad)
        self.assertTrue(torch.isfinite(scale_grad))

    def test_gaussian_regularization_returns_finite_scalar(self):
        head = GaussianSegRefinementHead3D(in_channels=16, num_classes=4, num_gaussians=3, hidden_dim=8)
        base_logits = torch.zeros(1, 4, 4, 4, 4)
        features = torch.randn(1, 16, 2, 2, 2)
        _, aux = head(base_logits, features)

        value = GaussianRefinementRegularization(lambda_sigma=1.0e-4, lambda_amplitude=1.0e-3)(aux)

        self.assertEqual(value.ndim, 0)
        self.assertTrue(torch.isfinite(value))

    def test_better_config_enables_gaussian_refinement(self):
        with open(ROOT / "configs" / "better3d_unet.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertTrue(cfg["model"]["gaussian_refinement"]["enabled"])
        self.assertEqual(cfg["loss"]["segmentation_loss"], "structure_aware_dice_ce")
        self.assertIn("lambda_gaussian_sigma", cfg["loss"])


if __name__ == "__main__":
    unittest.main()
