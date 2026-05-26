import unittest
from pathlib import Path

import torch
import yaml

from src.losses import BiophysicsInformedLoss3D, DensityCouplingLoss


ROOT = Path(__file__).resolve().parents[1]


class DensityCouplingLossTest(unittest.TestCase):
    def test_density_coupling_loss_is_lower_when_density_matches_wt(self):
        target = torch.zeros(1, 4, 8, 8, 8)
        target[:, 2, 2:6, 2:6, 2:6] = 1.0
        target[:, 1, 3:5, 3:5, 3:5] = 1.0
        target[:, 3, 3:5, 3:5, 3:5] = 1.0

        matched_density = torch.zeros(1, 1, 4, 4, 4)
        matched_density[:, :, 1:3, 1:3, 1:3] = 1.0
        wrong_density = 1.0 - matched_density

        loss_fn = DensityCouplingLoss(region_channel=2)

        self.assertLess(loss_fn(matched_density, target), loss_fn(wrong_density, target))

    def test_biophysics_loss_reports_density_component(self):
        logits = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        target = torch.zeros(1, 4, 8, 8, 8)
        target[:, 2, 2:6, 2:6, 2:6] = 1.0
        target[:, 1, 3:5, 3:5, 3:5] = 1.0
        target[:, 3, 3:5, 3:5, 3:5] = 1.0
        t_tensor = torch.ones(1, 64, 1, requires_grad=True)
        u_hat = torch.sigmoid(t_tensor.permute(0, 2, 1).reshape(1, 1, 4, 4, 4))

        criterion = BiophysicsInformedLoss3D(lambda_density=0.3)
        total, parts = criterion(logits, target, u_hat, t_tensor)

        self.assertIn("density", parts)
        self.assertGreater(parts["density"], 0.0)
        self.assertAlmostEqual(
            float(total.detach()),
            parts["seg"] + parts["pde"] + parts["bc"] + 0.3 * parts["density"],
            places=4,
        )

    def test_better_config_enables_density_coupling(self):
        with open(ROOT / "configs" / "better3d_unet.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertTrue(cfg["loss"]["use_biophysics"])
        self.assertGreater(cfg["loss"]["lambda_density"], 0.0)
        self.assertEqual(cfg["loss"]["density_region_channel"], 2)


if __name__ == "__main__":
    unittest.main()
