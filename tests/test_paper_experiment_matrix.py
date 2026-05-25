import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "paper_experiment_matrix.yaml"


class PaperExperimentMatrixTest(unittest.TestCase):
    def test_matrix_covers_paper_table_methods(self):
        with open(MATRIX_PATH, "r", encoding="utf-8") as f:
            matrix = yaml.safe_load(f)

        expected = {
            ("UNet", "baseline"),
            ("UNet", "biophysics"),
            ("R2-UNet", "baseline"),
            ("R2-UNet", "biophysics"),
            ("nn-UNet", "baseline"),
            ("nn-UNet", "biophysics"),
            ("UNETR", "baseline"),
            ("UNETR", "biophysics"),
            ("SegResNet", "baseline"),
            ("SegResNet", "biophysics"),
            ("SegResNetVAE", "baseline"),
        }
        actual = {
            (item["backbone"], item["variant"])
            for item in matrix["main_table"]
        }
        self.assertEqual(expected, actual)

    def test_matrix_records_paper_ablation_dimensions(self):
        with open(MATRIX_PATH, "r", encoding="utf-8") as f:
            matrix = yaml.safe_load(f)

        self.assertEqual(
            {
                "activation_and_boundary",
                "modality_subset",
                "training_set_fraction",
                "segmentation_loss",
            },
            set(matrix["ablations"]),
        )

    def test_matrix_keeps_unimplemented_items_explicit(self):
        with open(MATRIX_PATH, "r", encoding="utf-8") as f:
            matrix = yaml.safe_load(f)

        blocked = [
            item for item in matrix["main_table"]
            if item.get("status") != "implemented"
        ]
        self.assertTrue(blocked)
        self.assertTrue(all("missing" in item for item in blocked))


if __name__ == "__main__":
    unittest.main()
