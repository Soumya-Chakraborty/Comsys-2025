# GPU Quick Start Guide - Gender Classification System

🚀 **Get your GPU-accelerated gender classification system running in minutes!**

## 🔧 Prerequisites

- **Linux/Windows/macOS** with Python 3.8+
- **NVIDIA GPU** with CUDA support (recommended)
- **8GB+ GPU memory** (16GB+ recommended for full ensemble)
- **16GB+ system RAM**

## ⚡ Quick Installation

### Option 1: Automated GPU Setup (Linux/macOS)
```bash
# Make installation script executable and run
chmod +x install_gpu.sh
./install_gpu.sh

# Activate environment
source venv/bin/activate

# Test GPU setup
python3 test_gpu_setup.py
```

### Option 2: Manual Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install PyTorch with CUDA (adjust for your CUDA version)
# For CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Verify GPU Setup

```bash
# Quick GPU test
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
"

# Comprehensive GPU test
python3 test_gpu_setup.py
```

## 🚀 Quick Training

### 1. Single Model Training (Fastest)
```bash
# Train CNN-Transformer hybrid model
python3 gender_classifier.py

# Expected time: 30-45 minutes
# Expected accuracy: 85-90%
```

### 2. Complete System Training
```bash
# Train everything: single model + ensemble + optimization
python3 train_complete_system.py

# Expected time: 2-3 hours
# Expected accuracy: 90-95%
```

### 3. Custom Training
```bash
# Custom parameters
python3 train_complete_system.py \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --output-dir my_results

# Skip certain components
python3 train_complete_system.py \
    --skip-ensemble \
    --epochs 50
```

## 🎯 Making Predictions

### Single Image
```bash
# Predict single image
python3 demo_predictions.py \
    --model output/models/best_single_model.pth \
    --image path/to/your/image.jpg

# Interactive mode
python3 demo_predictions.py \
    --model output/models/best_single_model.pth \
    --interactive
```

### Batch Processing
```bash
# Process entire folder
python3 demo_predictions.py \
    --model output/models/best_ensemble_model.pth \
    --batch path/to/image/folder \
    --output results.csv
```

## 📊 Expected Performance

| Model Type | Training Time | Accuracy | GPU Memory |
|------------|---------------|----------|------------|
| Single CNN-Transformer | 30-45 min | 85-90% | 4-6GB |
| Ensemble (4 models) | 2-3 hours | 90-95% | 8-12GB |
| Optimized Student | 1 hour | 85-90% | 2-4GB |

## 🔧 GPU Optimization Tips

### Memory Management
```bash
# For low GPU memory (4-6GB)
python3 train_complete_system.py --batch-size 16 --skip-ensemble

# For high GPU memory (12GB+)
python3 train_complete_system.py --batch-size 64
```

### Monitor GPU Usage
```bash
# Monitor during training
watch -n 1 nvidia-smi

# Or use built-in monitoring
python3 -c "
from gpu_optimized_trainer import GPUManager
gpu = GPUManager()
print(gpu.get_gpu_memory_info())
"
```

## 🐛 Common Issues & Solutions

### 1. CUDA Out of Memory
```bash
# Reduce batch size
python3 train_complete_system.py --batch-size 8

# Enable gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Clear cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

### 2. Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Optimize data loading
python3 train_complete_system.py --config config_template.json
# Edit config: set "num_workers": 8
```

### 3. Model Loading Issues
```python
# Check available models
import os
print(os.listdir('output/models/'))

# Load with CPU fallback
model = torch.load('model.pth', map_location='cpu')
```

## 📈 Performance Monitoring

### During Training
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Memory usage tracking
python3 -c "
import torch
import time
while True:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f'GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached')
    time.sleep(5)
"
```

### Training Logs
```bash
# View training progress
tail -f output/logs/training.log

# Plot training curves
python3 -c "
from gender_classifier import GenderClassifier
classifier = GenderClassifier()
classifier.load_best_model('output/models/best_model.pth')
classifier.plot_training_curves('training_curves.png')
"
```

## 🎯 Optimal Configuration

### For RTX 4090 (24GB)
```json
{
  "data": {"batch_size": 64, "num_workers": 12},
  "training": {"num_epochs": 100, "learning_rate": 1e-4},
  "models": {
    "train_single": true,
    "train_ensemble": true,
    "apply_distillation": true,
    "apply_quantization": true
  }
}
```

### For RTX 3080 (10GB)
```json
{
  "data": {"batch_size": 32, "num_workers": 8},
  "training": {"num_epochs": 50, "learning_rate": 1e-4},
  "models": {
    "train_single": true,
    "train_ensemble": false,
    "apply_distillation": true,
    "apply_quantization": true
  }
}
```

### For RTX 3060 (8GB)
```json
{
  "data": {"batch_size": 16, "num_workers": 6},
  "training": {"num_epochs": 50, "learning_rate": 5e-5},
  "models": {
    "train_single": true,
    "train_ensemble": false,
    "apply_distillation": false,
    "apply_quantization": true
  }
}
```

## 🚀 Advanced Features

### Mixed Precision Training
```python
# Automatically enabled for CUDA
# Provides ~30% speedup with minimal accuracy loss
print("Mixed precision:", torch.cuda.is_available())
```

### Multi-GPU Training
```bash
# Use DataParallel for multiple GPUs
python3 train_complete_system.py --multi-gpu

# Or use DistributedDataParallel
torchrun --nproc_per_node=2 train_complete_system.py
```

### Model Optimization
```bash
# Create deployment model
python3 -c "
from ensemble_classifier import AdvancedEnsembleClassifier
classifier = AdvancedEnsembleClassifier()
classifier.create_deployment_model()
"
```

## 📞 Support

### Check System Status
```bash
# Complete system check
python3 -c "
import torch
import torchvision
import timm
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA:', torch.version.cuda if torch.cuda.is_available() else 'Not available')
print('✅ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')
print('✅ Memory:', f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A')
"
```

### Common Commands
```bash
# Check CUDA version
nvcc --version

# Check PyTorch CUDA version
python3 -c "import torch; print(torch.version.cuda)"

# Clear all GPU memory
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Reset GPU
sudo nvidia-smi --gpu-reset
```

## 🎉 Success Indicators

You'll know everything is working when you see:

1. **✅ GPU Detection**: `CUDA available: True`
2. **✅ Memory Usage**: GPU memory utilization during training
3. **✅ Mixed Precision**: `Mixed Precision: True` in training logs
4. **✅ Fast Training**: Epochs complete in 1-2 minutes (vs 10-15 on CPU)
5. **✅ High Accuracy**: >85% validation accuracy

## 📚 Next Steps

1. **Optimize for your GPU**: Adjust batch size based on your GPU memory
2. **Experiment with models**: Try different architectures in the ensemble
3. **Deploy models**: Use quantized models for production
4. **Monitor bias**: Check fairness metrics regularly
5. **Scale up**: Add more data or try larger models

---

**🚀 Ready to train? Run `python3 train_complete_system.py` and watch your GPU work its magic!**