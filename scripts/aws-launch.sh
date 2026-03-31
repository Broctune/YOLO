#!/bin/bash
# Launch a GPU training instance in the Siftr Development VPC.
# Usage:
#   ./scripts/aws-launch.sh              # spot instance (default, ~65% cheaper)
#   ./scripts/aws-launch.sh on-demand    # on-demand instance (no interruptions)
#   ./scripts/aws-launch.sh spot         # explicit spot
#
# Prerequisites:
#   - AWS CLI configured with --profile siftr
#   - vCPU quota approved for G instances (L-DB2E81BA on-demand, L-3819A6DF spot)

set -euo pipefail

PROFILE="siftr"
INSTANCE_TYPE="g5.xlarge"
AMI_ID="ami-09a372bc8e4d71425"  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)
KEY_NAME="siftr-ec2-keypair"
SECURITY_GROUP="sg-086f704f9ed537d39"  # siftr-yolo-training-sg
SUBNET="subnet-04de75c70b91390e7"      # DMZ Subnet A (us-west-2a, has IGW)
VOLUME_SIZE=150  # GB, gp3 root volume
INSTANCE_NAME="yolo-gpu-training"
MARKET_TYPE="${1:-spot}"

echo "=== YOLO GPU Training Instance Launcher ==="
echo "Instance type: ${INSTANCE_TYPE}"
echo "Market type:   ${MARKET_TYPE}"
echo "AMI:           ${AMI_ID}"
echo ""

# Check for existing running/stopped instance with this name
EXISTING=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=${INSTANCE_NAME}" "Name=instance-state-name,Values=running,stopped,pending" \
  --query 'Reservations[0].Instances[0].[InstanceId,State.Name,PublicIpAddress]' \
  --output text \
  --profile "${PROFILE}" 2>/dev/null || echo "None")

if [[ "${EXISTING}" != "None" && "${EXISTING}" != "null" && -n "${EXISTING}" ]]; then
  read -r EXISTING_ID EXISTING_STATE EXISTING_IP <<< "${EXISTING}"
  if [[ "${EXISTING_ID}" != "None" && "${EXISTING_ID}" != "null" ]]; then
    echo "Found existing instance: ${EXISTING_ID} (${EXISTING_STATE})"
    if [[ "${EXISTING_STATE}" == "stopped" ]]; then
      echo "Starting stopped instance..."
      aws ec2 start-instances --instance-ids "${EXISTING_ID}" --profile "${PROFILE}" > /dev/null
      echo "Waiting for instance to start..."
      aws ec2 wait instance-running --instance-ids "${EXISTING_ID}" --profile "${PROFILE}"
      PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "${EXISTING_ID}" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text --profile "${PROFILE}")
      echo ""
      echo "Instance started!"
      echo "  SSH: ssh -i ~/Documents/Siftr/siftr-ec2-keypair.pem ubuntu@${PUBLIC_IP}"
      exit 0
    elif [[ "${EXISTING_STATE}" == "running" ]]; then
      echo "Instance is already running."
      echo "  SSH: ssh -i ~/Documents/Siftr/siftr-ec2-keypair.pem ubuntu@${EXISTING_IP}"
      exit 0
    fi
  fi
fi

# Build launch command
LAUNCH_ARGS=(
  --image-id "${AMI_ID}"
  --instance-type "${INSTANCE_TYPE}"
  --key-name "${KEY_NAME}"
  --security-group-ids "${SECURITY_GROUP}"
  --subnet-id "${SUBNET}"
  --associate-public-ip-address
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":false}}]"
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]"
  --profile "${PROFILE}"
)

if [[ "${MARKET_TYPE}" == "spot" ]]; then
  LAUNCH_ARGS+=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}')
  echo "Using spot pricing (persistent, stops on interruption)"
else
  echo "Using on-demand pricing"
fi

echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances "${LAUNCH_ARGS[@]}" \
  --query 'Instances[0].InstanceId' --output text)

echo "Instance ID: ${INSTANCE_ID}"
echo "Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --profile "${PROFILE}"

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text --profile "${PROFILE}")

echo ""
echo "=== Instance Ready ==="
echo "  Instance ID: ${INSTANCE_ID}"
echo "  Public IP:   ${PUBLIC_IP}"
echo "  SSH:         ssh -i ~/Documents/Siftr/siftr-ec2-keypair.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "Next steps:"
echo "  1. SSH into the instance"
echo "  2. Run the setup script: bash scripts/aws-setup.sh"
echo "  3. Upload checkpoints:   bash scripts/aws-upload-checkpoints.sh ${PUBLIC_IP}"
