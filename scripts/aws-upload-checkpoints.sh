#!/bin/bash
# Upload local checkpoints and weights to the EC2 training instance.
# Run this FROM your local Mac.
# Usage: ./scripts/aws-upload-checkpoints.sh <instance-public-ip>
#
# Uploads:
#   - runs/train/v9t-sku110k/checkpoints/ (~1GB) — resume training from here
#   - weights/sku-110-best.pt (15MB) — best trained model so far

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <instance-public-ip> [ssh-key-path]"
  echo "Example: $0 54.200.1.2"
  echo "Example: $0 54.200.1.2 ~/.ssh/my-key.pem"
  exit 1
fi

REMOTE_IP="$1"
SSH_KEY="${2:-$HOME/Documents/Siftr/siftr-ec2-keypair.pem}"
REMOTE_USER="ubuntu"
REMOTE_BASE="/home/${REMOTE_USER}/YOLO"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=== Uploading checkpoints to ${REMOTE_IP} ==="

# Create remote directories
ssh -i "${SSH_KEY}" "${REMOTE_USER}@${REMOTE_IP}" \
  "mkdir -p ${REMOTE_BASE}/runs/train/v9t-sku110k/checkpoints ${REMOTE_BASE}/weights"

# Upload checkpoints
if [[ -d "${PROJECT_DIR}/runs/train/v9t-sku110k/checkpoints" ]]; then
  echo "Uploading checkpoints..."
  scp -i "${SSH_KEY}" \
    "${PROJECT_DIR}/runs/train/v9t-sku110k/checkpoints/"* \
    "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_BASE}/runs/train/v9t-sku110k/checkpoints/"
  echo "  Done."
else
  echo "  No checkpoints found at ${PROJECT_DIR}/runs/train/v9t-sku110k/checkpoints/"
fi

# Upload best weights
if [[ -f "${PROJECT_DIR}/weights/sku-110-best.pt" ]]; then
  echo "Uploading weights/sku-110-best.pt..."
  scp -i "${SSH_KEY}" \
    "${PROJECT_DIR}/weights/sku-110-best.pt" \
    "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_BASE}/weights/"
  echo "  Done."
else
  echo "  No weights found at ${PROJECT_DIR}/weights/sku-110-best.pt"
fi

echo ""
echo "=== Upload Complete ==="
echo "SSH: ssh -i ${SSH_KEY} ${REMOTE_USER}@${REMOTE_IP}"
