#!/bin/bash
# ==============================================================
#  Jetson CUDA + Nemotron-3B 4-bit environment setup
#  Tested on JetPack 6.1 (CUDA 12.6, Ubuntu 20.04/22.04 SBSA)
# ==============================================================

set -e
echo "🧱 Setting up Nemotron-3B CUDA environment..."

# ---------- 1. System check ----------
echo "🔍 Checking Python and CUDA versions..."
python3 --version || sudo apt install python3 -y
nvcc --version || echo "⚠️ CUDA not found in PATH, continuing..."


# ---------- 4. cuSPARSELt setup ----------
echo "⚙️ Installing cuSPARSELt (needed by bitsandbytes)..."
UBUNTU_VERSION=$(lsb_release -rs)
if [[ "$UBUNTU_VERSION" == "20.04" ]]; then
    CUDA_REPO="ubuntu2004"
else
    CUDA_REPO="ubuntu2204"
fi
wget -q https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/sbsa/cuda-repo-${CUDA_REPO}-sbsa-local_12.6.77-1_arm64.deb
sudo dpkg -i cuda-repo-${CUDA_REPO}-sbsa-local_12.6.77-1_arm64.deb
sudo cp /var/cuda-repo-${CUDA_REPO}-sbsa-local/cuda-*-keyring.gpg /usr/share/keyrings/ || true
sudo apt update
sudo apt install -y libcusparseLt-dev
sudo ldconfig
echo "✅ cuSPARSELt installed."

# ---------- 5. Verify bitsandbytes ----------
python3 - <<'PY'
import torch, bitsandbytes as bnb
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
bnb.utils.check_cuda_setup()
PY

# ---------- 6. Create .env template ----------
echo "🧾 Creating .env file..."
cat > .env <<'EOF'
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN_HERE
HF_MODEL=nvidia/omni-embed-nemotron-3b
EOF
echo "⚠️ Edit .env and insert your Hugging Face token before running main.py."

# ---------- 7. Final message ----------
echo "🎉 Setup complete!"
echo "Activate environment:  source ~/Downloads/venv/bin/activate"
echo "Run server:           python3 main.py"
echo "Or:                   uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo "Then open:            http://127.0.0.1:8000/docs"
