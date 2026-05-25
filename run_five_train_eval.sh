#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
METHOD="${1:-paper}"
RUNS="${RUNS:-5}"
SPLIT="${SPLIT:-test}"
EVAL_MODE="${EVAL_MODE:-auto}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-final_model.pth}"
LOG_DIR="${LOG_DIR:-logs}"
PREPROCESS="${PREPROCESS:-1}"
PREPROCESS_OVERWRITE="${PREPROCESS_OVERWRITE:-1}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
PAPER_SESSION="${PAPER_SESSION:-paper3d}"
BASELINE_SESSION="${BASELINE_SESSION:-baseline3d}"
PAPER_CUDA_VISIBLE_DEVICES="${PAPER_CUDA_VISIBLE_DEVICES:-0}"
BASELINE_CUDA_VISIBLE_DEVICES="${BASELINE_CUDA_VISIBLE_DEVICES:-1}"
REUSE_TMUX_SESSIONS="${REUSE_TMUX_SESSIONS:-0}"

if [[ -f "src/train3d.py" && -d "configs" ]]; then
  PROJECT_DIR="."
elif [[ -f "src/src/train3d.py" && -d "src/configs" ]]; then
  PROJECT_DIR="src"
else
  echo "ERROR: Cannot find project files. Expected src/train3d.py or src/src/train3d.py."
  exit 1
fi

TRAIN_SCRIPT="${PROJECT_DIR}/src/train3d.py"
PREPROCESS_SCRIPT="${PROJECT_DIR}/src/preprocess3d.py"
EVALUATE_SCRIPT="${PROJECT_DIR}/src/evaluate.py"

run_preprocess() {
  local config_path="$1"

  if [[ "${SKIP_PREPROCESS}" == "1" ]]; then
    echo "Preprocess: skipped by SKIP_PREPROCESS=1"
    return
  fi
  if [[ "${PREPROCESS}" != "1" ]]; then
    echo "Preprocess: disabled by PREPROCESS=${PREPROCESS}"
    return
  fi

  local preprocess_args=(--config "${config_path}")
  if [[ "${PREPROCESS_OVERWRITE}" == "1" ]]; then
    preprocess_args+=(--overwrite)
  fi

  echo
  echo "[preprocess] ${config_path}"
  "${PYTHON_BIN}" "${PREPROCESS_SCRIPT}" "${preprocess_args[@]}" \
    2>&1 | tee "${LOG_DIR}/preprocess_$(date +%Y%m%d_%H%M%S).log"
}

run_in_tmux() {
  local method="$1"
  local session="$2"
  local cuda_devices="$3"

  local cmd
  cmd="cd \"${SCRIPT_DIR}\" && "
  if [[ -n "${cuda_devices}" ]]; then
    cmd+="CUDA_VISIBLE_DEVICES=\"${cuda_devices}\" "
  fi
  cmd+="SKIP_PREPROCESS=1 "
  cmd+="RUNS=\"${RUNS}\" SPLIT=\"${SPLIT}\" EVAL_MODE=\"${EVAL_MODE}\" "
  cmd+="CHECKPOINT_NAME=\"${CHECKPOINT_NAME}\" LOG_DIR=\"${LOG_DIR}\" "
  cmd+="PYTHON_BIN=\"${PYTHON_BIN}\" "
  cmd+="bash \"${SCRIPT_DIR}/${SCRIPT_NAME}\" \"${method}\""

  if tmux has-session -t "${session}" 2>/dev/null; then
    if [[ "${REUSE_TMUX_SESSIONS}" != "1" ]]; then
      echo "ERROR: tmux session already exists: ${session}"
      echo "Set REUSE_TMUX_SESSIONS=1 to send the command to this existing session."
      echo "Attach with: tmux attach -t ${session}"
      exit 1
    fi
    tmux send-keys -t "${session}" "${cmd}" C-m
    echo "Sent ${method} command to existing tmux session '${session}'. Attach with: tmux attach -t ${session}"
  else
    tmux new-session -d -s "${session}" "${cmd}"
    echo "Started tmux session '${session}' for ${method}. Attach with: tmux attach -t ${session}"
  fi
}

if [[ "${METHOD}" == "both" ]]; then
  mkdir -p "${LOG_DIR}"
  run_preprocess "${PROJECT_DIR}/configs/paper3d_unet.yaml"
  run_in_tmux paper "${PAPER_SESSION}" "${PAPER_CUDA_VISIBLE_DEVICES}"
  run_in_tmux baseline "${BASELINE_SESSION}" "${BASELINE_CUDA_VISIBLE_DEVICES}"
  echo
  echo "Paper session:    tmux attach -t ${PAPER_SESSION}    CUDA_VISIBLE_DEVICES=${PAPER_CUDA_VISIBLE_DEVICES}"
  echo "Baseline session: tmux attach -t ${BASELINE_SESSION} CUDA_VISIBLE_DEVICES=${BASELINE_CUDA_VISIBLE_DEVICES}"
  exit 0
fi

case "${METHOD}" in
  paper)
    BASE_CONFIG="${PROJECT_DIR}/configs/paper3d_unet.yaml"
    RUN_PREFIX="paper3d_unet"
    ;;
  baseline)
    BASE_CONFIG="${PROJECT_DIR}/configs/baseline3d_unet.yaml"
    RUN_PREFIX="baseline3d_unet"
    ;;
  *)
    echo "Usage: $0 [paper|baseline|both]"
    echo
    echo "Environment variables:"
    echo "  RUNS=5                         Number of train/eval runs"
    echo "  PYTHON_BIN=python              Python executable"
    echo "  SPLIT=test                     Evaluation split: val, test, or both"
    echo "  EVAL_MODE=auto                 Evaluation mode: preprocessed, raw, or auto"
    echo "  CHECKPOINT_NAME=final_model.pth Checkpoint used for evaluation"
    echo "  LOG_DIR=logs                   Log directory"
    echo "  PREPROCESS=1                   Run preprocessing before train/eval"
    echo "  PREPROCESS_OVERWRITE=1         Regenerate existing preprocessed .npy files"
    echo "  SKIP_PREPROCESS=0              Internal: skip preprocessing in tmux child runs"
    echo "  PAPER_SESSION=paper3d          tmux session name for paper method"
    echo "  BASELINE_SESSION=baseline3d    tmux session name for baseline method"
    echo "  PAPER_CUDA_VISIBLE_DEVICES=0   GPU list for paper tmux session in both mode"
    echo "  BASELINE_CUDA_VISIBLE_DEVICES=1 GPU list for baseline tmux session in both mode"
    echo "  REUSE_TMUX_SESSIONS=0          Set to 1 to send commands to existing tmux sessions"
    exit 2
    ;;
esac

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${PROJECT_DIR}/outputs/${RUN_PREFIX}_5runs_${TIMESTAMP}"
CONFIG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CONFIG_DIR}" "${LOG_DIR}"

echo "Method: ${METHOD}"
echo "Base config: ${BASE_CONFIG}"
echo "Run root: ${RUN_ROOT}"
echo "Runs: ${RUNS}"
echo "Checkpoint for eval: ${CHECKPOINT_NAME}"

run_preprocess "${BASE_CONFIG}"

for RUN_ID in $(seq 1 "${RUNS}"); do
  RUN_NAME="run$(printf "%02d" "${RUN_ID}")"
  RUN_OUTPUT="${RUN_ROOT}/${RUN_NAME}"
  RUN_CONFIG="${CONFIG_DIR}/${RUN_NAME}.yaml"

  "${PYTHON_BIN}" - "${BASE_CONFIG}" "${RUN_CONFIG}" "${RUN_OUTPUT}" "${RUN_ID}" <<'PY'
import sys
import yaml

base_config, run_config, run_output, run_id_arg = sys.argv[1:5]
run_id = int(run_id_arg)
with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["seed"] = int(cfg.get("seed", 42)) + run_id - 1
cfg["output_dir"] = run_output
with open(run_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  echo
  echo "[${METHOD} ${RUN_NAME}] train"
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" --config "${RUN_CONFIG}" \
    2>&1 | tee "${LOG_DIR}/${RUN_PREFIX}_${RUN_NAME}_train_${TIMESTAMP}.log"

  CHECKPOINT_PATH="${RUN_OUTPUT}/${CHECKPOINT_NAME}"
  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "ERROR: Missing checkpoint after training: ${CHECKPOINT_PATH}"
    exit 1
  fi

  echo "[${METHOD} ${RUN_NAME}] evaluate ${SPLIT}"
  "${PYTHON_BIN}" "${EVALUATE_SCRIPT}" \
    --config "${RUN_CONFIG}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --split "${SPLIT}" \
    --mode "${EVAL_MODE}" \
    2>&1 | tee "${LOG_DIR}/${RUN_PREFIX}_${RUN_NAME}_eval_${TIMESTAMP}.log"

  if [[ "${SPLIT}" == "both" ]]; then
    METRIC_SPLITS=(val test)
  else
    METRIC_SPLITS=("${SPLIT}")
  fi

  for METRIC_SPLIT in "${METRIC_SPLITS[@]}"; do
    METRICS_CSV="${RUN_OUTPUT}/${METRIC_SPLIT}_metrics.csv"
    SUMMARY_CSV="${RUN_ROOT}/${METRIC_SPLIT}_summary.csv"
    "${PYTHON_BIN}" - "${METRICS_CSV}" "${SUMMARY_CSV}" "${RUN_NAME}" <<'PY'
import csv
import math
import sys
from pathlib import Path

metrics_csv, summary_csv, run_name = sys.argv[1:4]
regions = ("TC", "WT", "ET")
rows = []
with open(metrics_csv, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

def mean(values):
    return sum(values) / len(values) if values else float("nan")

def std(values):
    if not values:
        return float("nan")
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5

out_path = Path(summary_csv)
write_header = not out_path.exists()
with open(out_path, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run",
            "region",
            "dice_mean",
            "dice_std",
            "hd95_mean",
            "hd95_std",
            "n_cases",
            "n_valid_hd95",
        ],
    )
    if write_header:
        writer.writeheader()
    for region in regions:
        dice = [float(row[f"{region}_dice"]) for row in rows]
        hd95 = [float(row[f"{region}_hd95"]) for row in rows]
        hd95 = [v for v in hd95 if not math.isnan(v)]
        writer.writerow(
            {
                "run": run_name,
                "region": region,
                "dice_mean": f"{mean(dice):.6f}",
                "dice_std": f"{std(dice):.6f}",
                "hd95_mean": f"{mean(hd95):.6f}",
                "hd95_std": f"{std(hd95):.6f}",
                "n_cases": len(rows),
                "n_valid_hd95": len(hd95),
            }
        )
PY
  done
done

echo
echo "Finished ${RUNS} ${METHOD} train/eval runs."
echo "Outputs: ${RUN_ROOT}"
