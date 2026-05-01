#!/usr/bin/env bash
#
# Sets up an H100 (or any recent CUDA) box to run scripts/repro_5230_sparse_mask.py
# end-to-end from a clean Ubuntu image (Lambda / RunPod / Vast base images all work).
#
# Usage:
#   bash scripts/setup_h100.sh
#
# Assumes:
#   - CUDA drivers already present (every cloud GPU image ships them).
#   - Python 3.10+ available as `python3`.
#   - git and curl present.
#
# Does NOT pin torch/transformers to the reporter's exact versions because the
# isolated kernel repro doesn't load Gemma-4 weights — we only need the CE
# code paths from unsloth_zoo + cut_cross_entropy to import cleanly.

set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:shivam2199/unsloth-zoo.git}"
REPO_URL_HTTPS="${REPO_URL_HTTPS:-https://github.com/shivam2199/unsloth-zoo.git}"
BRANCH="${BRANCH:-fix/fused-ce-sparse-mask-gemma4}"
WORK_DIR="${WORK_DIR:-$HOME/unsloth-zoo}"
VENV_DIR="${VENV_DIR:-$HOME/.venv-unsloth-5230}"

echo "=== nvidia-smi sanity check ==="
nvidia-smi || { echo "No GPU detected"; exit 1; }

echo
echo "=== system packages ==="
# Minimal CUDA containers (RunPod/Vast) often lack apt but already have
# python3, git, gcc. Only try to install if apt-get exists AND something's
# actually missing.
need_pkgs=()
command -v python3 >/dev/null || need_pkgs+=(python3-venv python3-pip)
command -v git     >/dev/null || need_pkgs+=(git)
command -v gcc     >/dev/null || need_pkgs+=(build-essential)

if [ ${#need_pkgs[@]} -eq 0 ]; then
    echo "python3, git, gcc already present — skipping apt"
elif command -v apt-get >/dev/null; then
    SUDO=""
    [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq "${need_pkgs[@]}"
else
    echo "MISSING: ${need_pkgs[*]} and no apt-get available."
    echo "Install them via your image's package manager and re-run."
    exit 1
fi

# venv module may be absent even when python3 is present (e.g. slim images).
if ! python3 -c "import venv" 2>/dev/null; then
    echo "python3-venv not available; using plain virtualenv instead"
    python3 -m pip install --user virtualenv
    USE_VIRTUALENV=1
fi

echo
echo "=== fresh venv at $VENV_DIR ==="
if [ "${USE_VIRTUALENV:-0}" = "1" ]; then
    python3 -m virtualenv "$VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

echo
echo "=== torch (CUDA 12.1 wheels — works on H100) ==="
# Latest stable torch auto-pulls a matching triton. Matching the reporter's
# torch==2.8.0+cu126 exactly isn't needed for the kernel repro; any 2.4+ works.
pip install --index-url https://download.pytorch.org/whl/cu121 torch

echo
echo "=== core deps for the repro ==="
pip install \
    transformers \
    cut-cross-entropy \
    numpy

echo
echo "=== clone fork + switch to fix branch ==="
if [ ! -d "$WORK_DIR" ]; then
    # Try SSH first (if user has keys set up); fall back to HTTPS.
    if ! git clone "$REPO_URL" "$WORK_DIR" 2>/dev/null; then
        echo "SSH clone failed, trying HTTPS..."
        git clone "$REPO_URL_HTTPS" "$WORK_DIR"
    fi
fi
cd "$WORK_DIR"
git fetch origin
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
git pull origin "$BRANCH" --rebase || true

echo
echo "=== editable install of unsloth_zoo ==="
pip install -e .

echo
echo "=== final environment ==="
python - <<'PY'
import torch, transformers, sys
print(f"python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__}")
print(f"cuda avail   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device       {torch.cuda.get_device_name(0)}")
    print(f"capability   {torch.cuda.get_device_capability(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"vram         {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
print(f"transformers {transformers.__version__}")
try:
    import cut_cross_entropy
    print(f"cut_cross_entropy OK")
except Exception as e:
    print(f"cut_cross_entropy FAIL: {e}")
try:
    import unsloth_zoo
    from unsloth_zoo.fused_losses import unsloth_fused_ce_loss
    from unsloth_zoo.loss_utils import fused_linear_cross_entropy
    print("unsloth_zoo imports OK")
except Exception as e:
    print(f"unsloth_zoo FAIL: {e}")
PY

echo
echo "=== running repro ==="
cd "$WORK_DIR"
python scripts/repro_5230_sparse_mask.py 2>&1 | tee scripts/repro_5230_output.log

echo
echo "=== done. log at $WORK_DIR/scripts/repro_5230_output.log ==="
