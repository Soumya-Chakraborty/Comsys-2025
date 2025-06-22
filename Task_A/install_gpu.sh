#!/bin/bash

# GPU-Enabled Installation Script for Gender Classification System
# This script installs PyTorch with CUDA support and all dependencies

set -e  # Exit on any error

echo "🚀 GPU-Enabled Installation Script for Gender Classification System"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_warning "This script is optimized for Linux. Proceed with caution on other systems."
fi

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_success "Python version $python_version is compatible"
else
    print_error "Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Check if NVIDIA GPU is available
print_status "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read gpu; do
        echo "  - $gpu MB"
    done
    CUDA_AVAILABLE=true
else
    print_warning "NVIDIA GPU not detected. Installing CPU-only version."
    CUDA_AVAILABLE=false
fi

# Create virtual environment if it doesn't exist
print_status "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Using existing virtual environment"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with appropriate CUDA support
print_status "Installing PyTorch..."
if [ "$CUDA_AVAILABLE" = true ]; then
    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        print_status "CUDA version detected: $cuda_version"

        # Install PyTorch with CUDA support
        if [[ "$cuda_version" == "12."* ]]; then
            print_status "Installing PyTorch with CUDA 12.x support..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$cuda_version" == "11."* ]]; then
            print_status "Installing PyTorch with CUDA 11.x support..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            print_warning "Unknown CUDA version. Installing latest PyTorch with CUDA support..."
            pip install torch torchvision torchaudio
        fi
    else
        print_status "Installing PyTorch with default CUDA support..."
        pip install torch torchvision torchaudio
    fi
else
    print_status "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core dependencies
print_status "Installing core dependencies..."
pip install timm>=0.9.0
pip install transformers>=4.30.0

# Install computer vision dependencies
print_status "Installing computer vision dependencies..."
pip install opencv-python>=4.5.0
pip install Pillow>=9.0.0
pip install albumentations>=1.3.0
pip install scikit-image>=0.19.0
pip install imageio>=2.22.0

# Install scientific computing
print_status "Installing scientific computing libraries..."
pip install numpy>=1.21.0
pip install scipy>=1.9.0
pip install pandas>=1.5.0
pip install scikit-learn>=1.2.0

# Install visualization
print_status "Installing visualization libraries..."
pip install matplotlib>=3.6.0
pip install seaborn>=0.11.0
pip install plotly>=5.0.0

# Install progress and logging
print_status "Installing progress and logging libraries..."
pip install tqdm>=4.64.0
pip install rich>=13.0.0

# Install metrics and evaluation
print_status "Installing evaluation libraries..."
pip install torchmetrics>=0.11.0

# Install GPU optimization libraries (if CUDA available)
if [ "$CUDA_AVAILABLE" = true ]; then
    print_status "Installing GPU optimization libraries..."

    # Install mixed precision training support
    pip install accelerate>=0.20.0

    # Try to install additional GPU optimization tools (optional)
    print_status "Installing optional GPU optimization tools..."
    pip install bitsandbytes>=0.41.0 || print_warning "bitsandbytes installation failed (optional)"

    # Install memory monitoring
    pip install py3nvml>=0.2.7 || print_warning "py3nvml installation failed (optional)"

    # Install pruning tools
    pip install torch-pruning>=1.2.0 || print_warning "torch-pruning installation failed (optional)"
fi

# Install utilities
print_status "Installing utilities..."
pip install psutil>=5.9.0
pip install click>=8.1.0
pip install pyyaml>=6.0

# Install development tools (optional)
print_status "Installing development tools..."
pip install jupyter>=1.0.0 || print_warning "Jupyter installation failed (optional)"
pip install ipywidgets>=8.0.0 || print_warning "ipywidgets installation failed (optional)"

# Verify installation
print_status "Verifying installation..."
python3 -c "
import torch
import torchvision
import timm
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from PIL import Image
import tqdm

print('✅ All core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('Running on CPU')
"

# Test GPU functionality if available
if [ "$CUDA_AVAILABLE" = true ]; then
    print_status "Testing GPU functionality..."
    python3 -c "
import torch
if torch.cuda.is_available():
    # Test basic GPU operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print('✅ GPU tensor operations working')

    # Test mixed precision
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    print('✅ Mixed precision support available')

    # Memory info
    allocated = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU Memory: {allocated:.2f}GB / {total:.1f}GB')
else:
    print('⚠️  CUDA not available for testing')
"
fi

# Create directories
print_status "Creating output directories..."
mkdir -p output/{models,plots,logs,results}
mkdir -p prediction_results

# Set permissions
chmod +x *.py

print_success "Installation completed successfully!"
echo ""
echo "🎉 Your GPU-enabled environment is ready!"
echo ""
echo "📋 Next steps:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Test the setup: python3 test_gpu_setup.py"
echo "  3. Start training: python3 train_complete_system.py"
echo ""
echo "📊 System Summary:"
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "  ✅ GPU acceleration enabled"
    echo "  ✅ Mixed precision training available"
    echo "  ✅ CUDA optimizations installed"
else
    echo "  ⚠️  CPU-only mode (GPU not available)"
fi
echo "  ✅ All dependencies installed"
echo "  ✅ Output directories created"
echo ""
echo "💡 Tips:"
echo "  • Run 'python3 test_gpu_setup.py' to verify GPU setup"
echo "  • Use 'nvidia-smi' to monitor GPU usage during training"
echo "  • Check 'output/' directory for training results"
echo "  • Read README.md for detailed usage instructions"
echo ""
print_success "Happy training! 🚀"
