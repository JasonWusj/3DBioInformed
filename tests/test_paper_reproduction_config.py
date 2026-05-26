import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PaperReproductionConfigTest(unittest.TestCase):
    def test_paper_and_baseline_configs_match_reported_unet_setup(self):
        paper = load_yaml("configs/paper3d_unet.yaml")
        baseline = load_yaml("configs/baseline3d_unet.yaml")

        for cfg in (paper, baseline):
            self.assertEqual(cfg["data"]["split_seed"], 42)
            self.assertEqual(cfg["data"]["train_ratio"], 0.7)
            self.assertEqual(cfg["data"]["val_ratio"], 0.1)
            self.assertEqual(cfg["data"]["test_ratio"], 0.2)
            self.assertEqual(cfg["data"]["patch_size"], [128, 128, 128])
            self.assertEqual(cfg["data"]["modalities"], ["-t1n", "-t1c", "-t2w", "-t2f"])
            self.assertEqual(cfg["training"]["epochs"], 175)
            self.assertEqual(cfg["training"]["batch_size"], 1)
            self.assertEqual(float(cfg["training"]["lr"]), 3.0e-4)
            self.assertEqual(cfg["training"]["optimizer"], "ranger")
            self.assertEqual(cfg["training"]["scheduler"], "cosine")
            self.assertTrue(cfg["training"]["amp"])
            self.assertEqual(cfg["model"]["density_estimator"]["feature_size"], [16, 16, 16])
            self.assertEqual(cfg["inference"]["eval_crop_mode"], "none")

        self.assertTrue(paper["loss"]["use_biophysics"])
        self.assertEqual(paper["loss"]["segmentation_loss"], "dice")
        self.assertEqual(paper["loss"]["lambda_pde"], 1.0)
        self.assertEqual(paper["loss"]["lambda_bc"], 1.0)
        self.assertEqual(paper["loss"]["lambda_density"], 0.1)
        self.assertEqual(paper["loss"]["density_region_channel"], 2)
        self.assertEqual(paper["loss"]["d_range"], [0.02, 1.5])
        self.assertEqual(paper["loss"]["rho_range"], [0.002, 0.2])
        self.assertEqual(paper["loss"]["sample_parameters"], "voxel")

        self.assertFalse(baseline["loss"]["use_biophysics"])
        self.assertEqual(baseline["loss"]["segmentation_loss"], "dice")
        self.assertEqual(baseline["loss"]["lambda_pde"], 0.0)
        self.assertEqual(baseline["loss"]["lambda_bc"], 0.0)

    def test_training_code_keeps_paper_default_dice_and_density_coupling(self):
        train_code = (ROOT / "src" / "train3d.py").read_text(encoding="utf-8")
        loss_code = (ROOT / "src" / "losses.py").read_text(encoding="utf-8")
        paper = load_yaml("configs/paper3d_unet.yaml")

        self.assertEqual(paper["loss"]["segmentation_loss"], "dice")
        self.assertIn("lambda_density", train_code)
        self.assertIn("DensityCouplingLoss", loss_code)
        self.assertIn("build_segmentation_loss", train_code)
        self.assertIn('loss_cfg.get("segmentation_loss", "dice")', train_code)

    def test_batch_eval_defaults_to_final_model_for_paper_testing(self):
        script = (ROOT / "run_five_train_eval.sh").read_text(encoding="utf-8")
        self.assertIn('CHECKPOINT_NAME="${CHECKPOINT_NAME:-final_model.pth}"', script)

    def test_batch_eval_offsets_seed_for_each_run(self):
        script = (ROOT / "run_five_train_eval.sh").read_text(encoding="utf-8")
        self.assertIn('"${RUN_ID}"', script)
        self.assertIn('cfg["seed"] = int(cfg.get("seed", 42)) + run_id - 1', script)

    def test_data_split_seed_is_independent_from_run_seed(self):
        train_code = (ROOT / "src" / "train3d.py").read_text(encoding="utf-8")
        eval_code = (ROOT / "src" / "evaluate.py").read_text(encoding="utf-8")
        legacy_eval_code = (ROOT / "src" / "infer_eval3d.py").read_text(encoding="utf-8")

        expected = 'seed=data_cfg.get("split_seed", cfg["seed"])'
        self.assertIn(expected, train_code)
        self.assertIn(expected, eval_code)
        self.assertIn('seed=cfg["data"].get("split_seed", cfg["seed"])', legacy_eval_code)

    def test_raw_eval_uses_uncropped_volume_for_sliding_window(self):
        eval_code = (ROOT / "src" / "evaluate.py").read_text(encoding="utf-8")
        legacy_eval_code = (ROOT / "src" / "infer_eval3d.py").read_text(encoding="utf-8")
        self.assertIn('crop_mode=infer_cfg.get("eval_crop_mode", "none")', eval_code)
        self.assertIn('crop_mode=infer_cfg.get("eval_crop_mode", "none")', legacy_eval_code)


if __name__ == "__main__":
    unittest.main()
