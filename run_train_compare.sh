#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-sequential}"
LOG_DIR="${LOG_DIR:-logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

BASELINE_CONFIG="configs/baseline3d_unet.yaml"
BIO_CONFIG="configs/paper3d_unet.yaml"

mkdir -p "${LOG_DIR}"

run_baseline() {
  echo "[baseline] start: ${BASELINE_CONFIG}"
  "${PYTHON_BIN}" src/train3d.py --config "${BASELINE_CONFIG}" 2>&1 | tee "${LOG_DIR}/baseline3d_unet_${TIMESTAMP}.log"
}

run_biophysics() {
  echo "[biophysics] start: ${BIO_CONFIG}"
  "${PYTHON_BIN}" src/train3d.py --config "${BIO_CONFIG}" 2>&1 | tee "${LOG_DIR}/paper3d_unet_${TIMESTAMP}.log"
}

case "${MODE}" in
  sequential)
    run_baseline
    run_biophysics
    ;;
  parallel)
    run_baseline &
    BASELINE_PID=$!
    run_biophysics &
    BIO_PID=$!
    wait "${BASELINE_PID}"
    wait "${BIO_PID}"
    ;;
  *)
    echo "Usage: $0 [sequential|parallel]"
    echo
    echo "Environment variables:"
    echo "  PYTHON_BIN=/path/to/python  Python executable, default: python"
    echo "  LOG_DIR=logs                Log directory, default: logs"
    exit 2
    ;;
esac

echo "All requested training jobs finished."
