import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class UNetAblationSupportTest(unittest.TestCase):
    def test_unet_ablation_configs_exist(self):
        expected = {
            "unet_biophy_relu.yaml",
            "unet_biophy_no_bc.yaml",
            "unet_biophy_relu_no_bc.yaml",
            "unet_biophy_modalities_t1n_t2f.yaml",
            "unet_biophy_modalities_t1n_t2w_t2f.yaml",
            "unet_biophy_train_fraction_25.yaml",
            "unet_biophy_train_fraction_50.yaml",
            "unet_biophy_train_fraction_75.yaml",
            "unet_biophy_loss_dice_ce.yaml",
            "unet_biophy_loss_focal.yaml",
            "unet_biophy_loss_jaccard.yaml",
            "unet_baseline_modalities_t1n_t2f.yaml",
            "unet_baseline_modalities_t1n_t2w_t2f.yaml",
            "unet_baseline_train_fraction_25.yaml",
            "unet_baseline_train_fraction_50.yaml",
            "unet_baseline_train_fraction_75.yaml",
            "unet_baseline_loss_dice_ce.yaml",
            "unet_baseline_loss_focal.yaml",
            "unet_baseline_loss_jaccard.yaml",
        }
        actual = {path.name for path in (ROOT / "configs" / "unet_ablations").glob("*.yaml")}
        self.assertTrue(expected.issubset(actual))

    def test_unet_ablation_configs_keep_unet_backbone(self):
        for path in (ROOT / "configs" / "unet_ablations").glob("*.yaml"):
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.assertEqual(cfg["model"]["backbone"], "unet3d")
            self.assertEqual(cfg["loss"]["use_biophysics"], path.name.startswith("unet_biophy_"))

    def test_code_supports_unet_ablation_knobs(self):
        model_code = (ROOT / "src" / "model.py").read_text(encoding="utf-8")
        train_code = (ROOT / "src" / "train3d.py").read_text(encoding="utf-8")
        data_code = (ROOT / "src" / "data.py").read_text(encoding="utf-8")
        loss_code = (ROOT / "src" / "losses.py").read_text(encoding="utf-8")

        self.assertIn('activation=model_cfg["density_estimator"].get("activation", "sine")', model_code)
        self.assertIn("build_segmentation_loss", train_code)
        self.assertIn('loss_cfg.get("segmentation_loss", "dice")', train_code)
        self.assertIn('data_cfg.get("train_fraction", 1.0)', train_code)
        self.assertIn("dice_loss = DiceLoss().to(device)", train_code)
        self.assertIn("train_seg_loss", train_code)
        self.assertIn("DiceWithBCELoss", loss_code)
        self.assertIn("FocalLoss", loss_code)
        self.assertIn("JaccardLoss", loss_code)

    def test_matrix_marks_unet_ablations_as_implemented(self):
        with open(ROOT / "configs" / "paper_experiment_matrix.yaml", "r", encoding="utf-8") as f:
            matrix = yaml.safe_load(f)

        self.assertTrue(all(item["status"] == "implemented" for item in matrix["unet_ablations"]))
        self.assertTrue(all(item["config"].startswith("configs/unet_ablations/") for item in matrix["unet_ablations"]))
        self.assertTrue(all((ROOT / item["config"]).exists() for item in matrix["unet_ablations"]))

    def test_unet_matrix_runner_references_existing_configs(self):
        script = (ROOT / "run_unet_experiment_matrix.py").read_text(encoding="utf-8")
        self.assertIn("paper_experiment_matrix.yaml", script)
        self.assertIn("src/train3d.py", script)
        self.assertIn("src/evaluate.py", script)
        self.assertIn("--dry-run", script)

    def test_shell_unet_matrix_runner_references_existing_configs(self):
        script = (ROOT / "run_unet_experiment_matrix.sh").read_text(encoding="utf-8")
        config_lines = [
            line.strip().strip('"')
            for line in script.splitlines()
            if line.strip().startswith('"configs/')
        ]
        self.assertTrue(config_lines)
        self.assertIn("configs/paper3d_unet.yaml", config_lines)
        self.assertIn("configs/baseline3d_unet.yaml", config_lines)
        self.assertTrue(all((ROOT / path).exists() for path in config_lines))
        self.assertIn("DRY_RUN", script)
        self.assertIn("src/evaluate.py", script)


if __name__ == "__main__":
    unittest.main()
