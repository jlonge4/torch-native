#!/usr/bin/env bash
# Deploy patches + test scripts to trn2-2 and run them in sequence.
#
# Usage: ./tests/deploy_and_test.sh [trn2-2-host]
# Default host: trn2-2

set -euo pipefail

HOST="${1:-trn2-2}"
REMOTE_DIR="/home/ubuntu/runway-ml"
VENV="$HOME/moduscope-deps-20260423-202725/ms_venv"
CKPT="/home/ubuntu/Wan2.2/Wan2.2-TI2V-5B"
WAN_PKG="/home/ubuntu/Wan2.2"

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Syncing to $HOST:$REMOTE_DIR ==="
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
    "$LOCAL_ROOT/wan22_neuron_patches/" \
    "$LOCAL_ROOT/tests/" \
    "$HOST:$REMOTE_DIR/"

echo ""
echo "=== Applying patches on $HOST ==="
ssh "$HOST" "bash $REMOTE_DIR/apply_patches.sh"

echo ""
echo "=== [1/3] VAE test (CPU, random weights) ==="
ssh "$HOST" "source $VENV/bin/activate && \
    cd $REMOTE_DIR && \
    python test_vae.py --device cpu"

echo ""
echo "=== [2/3] T5 encoder test (Neuron) ==="
ssh "$HOST" "source $VENV/bin/activate && \
    cd $REMOTE_DIR && \
    python test_t5.py --checkpoint-dir $CKPT"

echo ""
echo "=== [3/3] Single denoising step test (Neuron) ==="
ssh "$HOST" "source $VENV/bin/activate && \
    WAN_PKG=$WAN_PKG \
    cd $REMOTE_DIR && \
    python test_pipeline_single_step.py --checkpoint-dir $CKPT"

echo ""
echo "All tests complete."
