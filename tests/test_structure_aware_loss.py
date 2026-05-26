import unittest
from pathlib import Path

import torch
import yaml

from src.losses import HierarchyConsistencyLoss, StructureAwareDiceBCELoss


ROOT = Path(__file__).resolve().parents[1]


class StructureAwareLossTest(unittest.TestCase):
    def test_hierarchy_loss_zero_when_regions_are_nested(self):
        logits = torch.full((1, 4, 2, 2, 2), -8.0)
        logits[:, 2] = 4.0  # WT
        logits[:, 1] = 2.0  # TC
        logits[:, 3] = 0.0  # ET

        loss = HierarchyConsistencyLoss()(logits, torch.zeros_like(logits))

        self.assertLess(float(loss), 1.0e-6)

    def test_hierarchy_loss_positive_when_et_exceeds_tc(self):
        logits = torch.full((1, 4, 2, 2, 2), -8.0)
        logits[:, 2] = 4.0  # WT
        logits[:, 1] = 0.0  # TC
        logits[:, 3] = 4.0  # ET

        loss = HierarchyConsistencyLoss()(logits, torch.zeros_like(logits))

        self.assertGreater(float(loss), 0.1)

    def test_builds_structure_aware_segmentation_loss(self):
        train_code = (ROOT / "src" / "train3d.py").read_text(encoding="utf-8")

        self.assertIn("structure_aware_dice_ce", train_code)
        self.assertIn("StructureAwareDiceBCELoss", train_code)

    def test_structure_aware_loss_returns_finite_scalar(self):
        loss = StructureAwareDiceBCELoss(lambda_hierarchy=0.2, lambda_boundary=0.5, boundary_width=1)
        logits = torch.randn(1, 4, 4, 4, 4)
        target = torch.zeros_like(logits)
        target[:, 2, 1:3, 1:3, 1:3] = 1.0
        target[:, 1, 1:3, 1:3, 1:3] = 1.0
        target[:, 3, 1:2, 1:2, 1:2] = 1.0

        value = loss(logits, target)

        self.assertEqual(value.ndim, 0)
        self.assertTrue(torch.isfinite(value))

    def test_better_config_uses_unet_and_structure_aware_loss(self):
        with open(ROOT / "configs" / "better3d_unet.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg["model"]["backbone"], "unet3d")
        self.assertTrue(cfg["loss"]["use_biophysics"])
        self.assertEqual(cfg["loss"]["segmentation_loss"], "structure_aware_dice_ce")


if __name__ == "__main__":
    unittest.main()
