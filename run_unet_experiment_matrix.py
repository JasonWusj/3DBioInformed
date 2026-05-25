import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
MATRIX_PATH = ROOT / "configs" / "paper_experiment_matrix.yaml"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def unet_config_paths():
    matrix = load_yaml(MATRIX_PATH)
    paths = []
    for item in matrix["main_table"]:
        if item.get("backbone") == "UNet" and item.get("status") == "implemented":
            paths.append(item["config"])
    for item in matrix["unet_ablations"]:
        if item.get("status") == "implemented":
            paths.append(item["config"])
    return paths


def command_text(command):
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_command(command, log_path, dry_run):
    print(command_text(command), flush=True)
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def config_output_dir(config_path):
    cfg = load_yaml(ROOT / config_path)
    return ROOT / cfg["output_dir"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the implemented UNet paper experiment matrix.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child commands.")
    parser.add_argument("--split", default="test", choices=["val", "test", "both"], help="Evaluation split.")
    parser.add_argument("--mode", default="auto", choices=["auto", "raw", "preprocessed"], help="Evaluation mode.")
    parser.add_argument("--checkpoint-name", default="final_model.pth", help="Checkpoint filename for evaluation.")
    parser.add_argument("--log-dir", default="logs", help="Directory for train/eval logs.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Do not run src/preprocess3d.py first.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them.")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / args.log_dir

    if not args.skip_preprocess:
        preprocess_command = [
            args.python,
            "src/preprocess3d.py",
            "--config",
            "configs/paper3d_unet.yaml",
        ]
        run_command(preprocess_command, log_dir / f"unet_matrix_preprocess_{timestamp}.log", args.dry_run)
    else:
        print("[preprocess] skipped", flush=True)

    for config_path in unet_config_paths():
        run_name = Path(config_path).stem
        output_dir = config_output_dir(config_path)
        checkpoint_path = output_dir / args.checkpoint_name

        print(f"\n[train] {config_path}", flush=True)
        train_command = [args.python, "src/train3d.py", "--config", config_path]
        run_command(train_command, log_dir / f"{run_name}_train_{timestamp}.log", args.dry_run)

        print(f"[evaluate] {config_path} split={args.split} checkpoint={checkpoint_path}", flush=True)
        if not args.dry_run and not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        eval_command = [
            args.python,
            "src/evaluate.py",
            "--config",
            config_path,
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            args.split,
            "--mode",
            args.mode,
        ]
        run_command(eval_command, log_dir / f"{run_name}_eval_{args.split}_{timestamp}.log", args.dry_run)

    print("\nUNet experiment matrix finished.", flush=True)


if __name__ == "__main__":
    main()
