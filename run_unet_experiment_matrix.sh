#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-test}"
EVAL_MODE="${EVAL_MODE:-auto}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-final_model.pth}"
LOG_DIR="${LOG_DIR:-logs}"
AUTO_PREPROCESS="${AUTO_PREPROCESS:-1}"
DRY_RUN="${DRY_RUN:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

CONFIGS=(
  "configs/baseline3d_unet.yaml"
  "configs/paper3d_unet.yaml"
  "configs/unet_ablations/unet_biophy_relu.yaml"
  "configs/unet_ablations/unet_biophy_no_bc.yaml"
  "configs/unet_ablations/unet_biophy_relu_no_bc.yaml"
  "configs/unet_ablations/unet_biophy_modalities_t1n_t2f.yaml"
  "configs/unet_ablations/unet_baseline_modalities_t1n_t2f.yaml"
  "configs/unet_ablations/unet_biophy_modalities_t1n_t2w_t2f.yaml"
  "configs/unet_ablations/unet_baseline_modalities_t1n_t2w_t2f.yaml"
  "configs/unet_ablations/unet_biophy_train_fraction_25.yaml"
  "configs/unet_ablations/unet_baseline_train_fraction_25.yaml"
  "configs/unet_ablations/unet_biophy_train_fraction_50.yaml"
  "configs/unet_ablations/unet_baseline_train_fraction_50.yaml"
  "configs/unet_ablations/unet_biophy_train_fraction_75.yaml"
  "configs/unet_ablations/unet_baseline_train_fraction_75.yaml"
  "configs/unet_ablations/unet_biophy_loss_dice_ce.yaml"
  "configs/unet_ablations/unet_baseline_loss_dice_ce.yaml"
  "configs/unet_ablations/unet_biophy_loss_focal.yaml"
  "configs/unet_ablations/unet_baseline_loss_focal.yaml"
  "configs/unet_ablations/unet_biophy_loss_jaccard.yaml"
  "configs/unet_ablations/unet_baseline_loss_jaccard.yaml"
)

mkdir -p "${LOG_DIR}"

if [[ "${AUTO_PREPROCESS}" == "1" ]]; then
  echo "[preprocess] configs/paper3d_unet.yaml"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" src/preprocess3d.py --config configs/paper3d_unet.yaml \
      2>&1 | tee "${LOG_DIR}/unet_matrix_preprocess_${TIMESTAMP}.log"
  fi
else
  echo "[preprocess] skipped because AUTO_PREPROCESS=${AUTO_PREPROCESS}"
fi

for CONFIG in "${CONFIGS[@]}"; do
  RUN_NAME="$(basename "${CONFIG}" .yaml)"
  OUTPUT_DIR="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["output_dir"])
PY
)"
  CHECKPOINT_PATH="${OUTPUT_DIR}/${CHECKPOINT_NAME}"

  echo
  echo "[train] ${CONFIG}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" src/train3d.py --config "${CONFIG}" \
      2>&1 | tee "${LOG_DIR}/${RUN_NAME}_train_${TIMESTAMP}.log"
  fi

  echo "[evaluate] ${CONFIG} split=${SPLIT} checkpoint=${CHECKPOINT_PATH}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
      echo "ERROR: Missing checkpoint: ${CHECKPOINT_PATH}"
      exit 1
    fi
    "${PYTHON_BIN}" src/evaluate.py \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT_PATH}" \
      --split "${SPLIT}" \
      --mode "${EVAL_MODE}" \
      2>&1 | tee "${LOG_DIR}/${RUN_NAME}_eval_${SPLIT}_${TIMESTAMP}.log"
  fi
done

echo
echo "UNet experiment matrix finished."
