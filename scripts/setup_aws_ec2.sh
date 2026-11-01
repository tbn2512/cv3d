#!/usr/bin/env bash
# ==============================================================================
# CV3D AWS EC2 GPU Instance Bootstrap Script
# Tested on: Ubuntu 22.04 LTS (Deep Learning AMI or standard Ubuntu 22.04)
# Target Instances: g4dn.xlarge (NVIDIA T4) or g5.xlarge (NVIDIA A10G)
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo "🚀 [1/5] Checking GPU & NVIDIA Drivers"
echo "=========================================================="
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ nvidia-smi not found. Installing NVIDIA headless drivers..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    echo "NVIDIA driver installed. A reboot may be required if modules don't load."
else
    nvidia-smi
fi

echo "=========================================================="
echo "📦 [2/5] Installing System Dependencies & COLMAP with CUDA"
echo "=========================================================="
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake git curl wget tar unzip pigz \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx xvfb \
    python3-pip python3-dev python3-venv colmap

echo "COLMAP version:"
colmap -h | head -n 3

echo "=========================================================="
echo "🐍 [3/5] Setting up Python Virtual Environment"
echo "=========================================================="
mkdir -p ~/cv3d
cd ~/cv3d

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1/12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=========================================================="
echo "⚡ [4/5] Installing Nerfstudio & CV Packages"
echo "=========================================================="
pip install ninja
pip install nerfstudio rembg onnxruntime-gpu pandas opencv-python tqdm

echo "=========================================================="
echo "📁 [5/5] Creating Directory Layout"
echo "=========================================================="
mkdir -p ~/cv3d/data ~/cv3d/processed ~/cv3d/scripts ~/cv3d/outputs

echo "=========================================================="
echo "✅ Environment Setup Complete!"
echo "Activate environment anytime with: source ~/cv3d/venv/bin/activate"
echo "=========================================================="
