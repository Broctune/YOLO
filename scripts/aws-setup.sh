#!/bin/bash
# Run this ON the EC2 GPU instance to set up the training environment.
# Usage: bash scripts/aws-setup.sh
#
# This script is idempotent — safe to re-run after spot interruptions or restarts.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

REPO_URL="https://github.com/Broctune/YOLO.git"
WORK_DIR="/home/ubuntu/YOLO"
DATASET_DIR="/home/ubuntu/datasets/SKU-110K"
VENV_DIR="${WORK_DIR}/.venv"

echo "=== YOLO GPU Training Setup ==="

# ── 1. System prerequisites ──────────────────────────────────────────────
echo ""
echo "[1/6] System prerequisites..."
sudo apt-get update -qq
sudo apt-get install -y -qq unzip > /dev/null 2>&1
pip install --quiet gdown uv

# ── 2. Clone / update repo ───────────────────────────────────────────────
echo "[2/6] Setting up repository..."
if [[ -d "${WORK_DIR}/.git" ]]; then
  echo "  Repo exists, pulling latest..."
  cd "${WORK_DIR}"
  git pull
else
  echo "  Cloning from ${REPO_URL}..."
  git clone "${REPO_URL}" "${WORK_DIR}"
  cd "${WORK_DIR}"
fi

# ── 3. Python environment ────────────────────────────────────────────────
echo "[3/6] Setting up Python environment..."
if [[ ! -d "${VENV_DIR}" ]]; then
  uv venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
uv pip install -r requirements-dev.txt

# ── 4. Verify GPU ────────────────────────────────────────────────────────
echo "[4/6] Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── 5. Download & prepare SKU-110K dataset ────────────────────────────────
echo "[5/6] Preparing SKU-110K dataset..."
if [[ -d "${DATASET_DIR}/images/train" ]] && [[ $(ls "${DATASET_DIR}/images/train/" 2>/dev/null | head -5 | wc -l) -gt 0 ]]; then
  echo "  Dataset already exists at ${DATASET_DIR}, skipping download."
else
  echo "  Downloading SKU-110K dataset..."
  mkdir -p "${DATASET_DIR}"
  cd /tmp

  # SKU-110K is hosted on Google Drive. The dataset contains:
  # - SKU110K_fixed.tar.gz: images
  # - SKU110K_YOLO.tar.gz: YOLO-format labels (if available)
  #
  # If gdown fails due to rate limits, manually download and upload to S3:
  #   aws s3 cp s3://your-bucket/SKU110K_fixed.tar.gz /tmp/
  #
  # The Google Drive file IDs may change. Check the SKU-110K GitHub page for current links.
  # Original paper repo: https://github.com/eg4000/SKU110K_CVPR19

  echo "  NOTE: SKU-110K download requires Google Drive file IDs."
  echo "  If automatic download fails, see the comments in this script for alternatives."
  echo ""

  # Attempt download — update these IDs if they change
  if [[ ! -f "SKU110K_fixed.tar.gz" ]]; then
    echo "  Downloading images..."
    gdown "1iq93lCdhaPUN0fWbLieMtzfB1850pKwd" -O SKU110K_fixed.tar.gz || {
      echo "  ERROR: gdown failed. Try one of these alternatives:"
      echo "    1. Upload to S3:  aws s3 cp SKU110K_fixed.tar.gz s3://your-bucket/"
      echo "       Then on EC2:  aws s3 cp s3://your-bucket/SKU110K_fixed.tar.gz /tmp/"
      echo "    2. Manual download from Google Drive and scp to instance"
      exit 1
    }
  fi

  echo "  Extracting..."
  tar xzf SKU110K_fixed.tar.gz

  # Organize into YOLO directory structure
  # SKU-110K comes with images and annotations; we need to organize into train/val splits
  mkdir -p "${DATASET_DIR}/images/train" "${DATASET_DIR}/images/val"
  mkdir -p "${DATASET_DIR}/labels/train" "${DATASET_DIR}/labels/val"

  # The extracted structure varies by version. Common layout:
  # SKU110K_fixed/images/ — all images
  # SKU110K_fixed/annotations/ — CSV annotations
  #
  # If YOLO-format labels are available (SKU110K_YOLO.tar.gz), use those.
  # Otherwise, convert from CSV annotations.
  echo ""
  echo "  Dataset extracted. You may need to:"
  echo "    1. Move images into ${DATASET_DIR}/images/{train,val}/"
  echo "    2. Move/convert labels into ${DATASET_DIR}/labels/{train,val}/"
  echo "    3. Run label conversion: python scripts/convert_sku110k_labels.py ${DATASET_DIR}/labels/train"
  echo "                             python scripts/convert_sku110k_labels.py ${DATASET_DIR}/labels/val"
  echo ""
  echo "  See the project README or DaveNotes.txt for the exact dataset structure."

  # Clean up
  # rm -f /tmp/SKU110K_fixed.tar.gz  # uncomment to save disk space
fi

# ── 6. W&B login ─────────────────────────────────────────────────────────
echo "[6/6] Weights & Biases..."
if wandb status 2>/dev/null | grep -q "Logged in"; then
  echo "  Already logged in to W&B."
else
  echo "  Please log in to Weights & Biases:"
  echo "    wandb login"
  echo "  (API key is in LastPass under dave@siftr.com)"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start training:"
echo "  cd ${WORK_DIR}"
echo "  source .venv/bin/activate"
echo "  python yolo/lazy.py \\"
echo "    task=train dataset=sku110k dataset.path=${DATASET_DIR} \\"
echo "    model=v9-t task.epoch=100 task.data.batch_size=32 task.data.cpu_num=4 \\"
echo "    accelerator=gpu device=1 weight=True name=v9t-sku110k-aws use_wandb=True"
echo ""
echo "To resume from checkpoint:"
echo "  python yolo/lazy.py \\"
echo "    task=train dataset=sku110k dataset.path=${DATASET_DIR} \\"
echo "    model=v9-t task.epoch=200 task.data.batch_size=32 task.data.cpu_num=4 \\"
echo "    accelerator=gpu device=1 weight=True name=v9t-sku110k-aws use_wandb=True \\"
echo "    '+resume=\"/home/ubuntu/YOLO/runs/train/v9t-sku110k/checkpoints/last.ckpt\"'"
