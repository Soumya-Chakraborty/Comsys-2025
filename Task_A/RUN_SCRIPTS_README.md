# Task A - Gender Classification System Runner Scripts

This directory contains automated scripts to run the complete Task A gender classification system with minimal manual setup.

## 🚀 Quick Start

### Linux/macOS
```bash
# Make script executable (if not already)
chmod +x run_task_a.sh

# Run with default settings
./run_task_a.sh

# Quick training (20 epochs, larger batch size)
./run_task_a.sh --quick

# Custom parameters
./run_task_a.sh --epochs 100 --batch-size 64 --lr 1e-3
```

### Windows
```cmd
# Run with default settings
run_task_a.bat

# Quick training (20 epochs, larger batch size)
run_task_a.bat --quick

# Custom parameters
run_task_a.bat --epochs 100 --batch-size 64 --lr 1e-3
```

## 📋 What These Scripts Do

Both scripts perform a complete end-to-end setup and training:

### 1. **System Checks**
- ✅ Python version (3.8+ required)
- ✅ GPU availability (CUDA detection)
- ✅ Disk space (minimum 5GB recommended)
- ✅ System memory (adjusts batch size if needed)

### 2. **Environment Setup**
- 🔧 Creates Python virtual environment
- 📦 Installs PyTorch with CUDA support (if available)
- 📦 Installs all required dependencies from `requirements.txt`
- ✨ Verifies installation with comprehensive tests

### 3. **Training Process**
- 🏋️ Trains CNN-Transformer hybrid model
- 🎯 Trains ensemble models (EfficientNet, ResNet, ViT, ConvNeXt)
- ⚡ Applies model optimization (quantization, distillation)
- 📊 Generates performance metrics and visualizations

### 4. **Results & Demo**
- 📈 Shows training results summary
- 🎮 Runs prediction demo
- 💾 Saves all models and evaluation results

## 🎛️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs N` | Number of training epochs | 50 |
| `--batch-size N` | Training batch size | 32 |
| `--lr, --learning-rate` | Learning rate | 1e-4 |
| `--output-dir DIR` | Output directory | output |
| `--skip-ensemble` | Skip ensemble training | false |
| `--skip-optimization` | Skip model optimization | false |
| `--gpu-only` | Exit if no GPU available | false |
| `--quick` | Quick mode (20 epochs, batch 64) | false |
| `--config FILE` | Use custom config file | - |
| `--help, -h` | Show help message | - |

## 🔧 Configuration Examples

### Quick Training (Development)
```bash
./run_task_a.sh --quick
# - 20 epochs
# - Batch size 64
# - ~30-45 minutes on GPU
```

### Production Training
```bash
./run_task_a.sh --epochs 100 --batch-size 32
# - 100 epochs
# - Full ensemble training
# - ~3-4 hours on GPU
```

### Memory-Constrained Systems
```bash
./run_task_a.sh --batch-size 16 --skip-ensemble
# - Smaller batch size
# - Single model only
# - Works with 4GB GPU memory
```

### GPU-Only Training
```bash
./run_task_a.sh --gpu-only --epochs 200
# - Exits if no GPU detected
# - Extended training for best results
```

## 📁 Output Structure

After running, you'll find:

```
output/
├── models/
│   ├── best_single_model.pth          # CNN-Transformer hybrid
│   ├── best_ensemble_model.pth        # Ensemble model
│   ├── deployment_model.pth           # Optimized for production
│   └── ...
├── plots/
│   ├── training_curves.png            # Loss/accuracy curves
│   ├── confusion_matrix.png           # Performance analysis
│   └── model_comparison.png           # Model comparisons
├── logs/
│   ├── training.log                   # Detailed training log
│   └── evaluation.log                 # Evaluation metrics
└── evaluation_results.json            # Complete results summary
```

## 🎯 Expected Performance

| Hardware | Training Time | Expected Accuracy |
|----------|---------------|-------------------|
| RTX 4090 (24GB) | 45 min - 2h | 90-95% |
| RTX 3080 (10GB) | 1h - 3h | 88-93% |
| RTX 3060 (8GB) | 1.5h - 4h | 85-90% |
| CPU Only | 8h - 24h | 82-88% |

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solution: Reduce batch size
./run_task_a.sh --batch-size 16
```

**2. Python Version Error**
```bash
# Solution: Install Python 3.8+
# Ubuntu/Debian: sudo apt install python3.8
# macOS: brew install python@3.8
# Windows: Download from python.org
```

**3. Missing Dataset**
```bash
# Ensure your directory structure is:
Task_A/
├── train/
│   ├── female/
│   └── male/
└── val/
    ├── female/
    └── male/
```

**4. Dependency Installation Failed**
```bash
# Solution: Update pip and try again
pip install --upgrade pip
./run_task_a.sh
```

**5. Training Interrupted**
```bash
# The script saves checkpoints, you can resume by re-running
./run_task_a.sh --output-dir output_resume
```

### Performance Optimization

**For High-End GPUs (16GB+)**
```bash
./run_task_a.sh --batch-size 64 --epochs 100
```

**For Mid-Range GPUs (8-12GB)**
```bash
./run_task_a.sh --batch-size 32 --epochs 50
```

**For Entry-Level GPUs (4-6GB)**
```bash
./run_task_a.sh --batch-size 16 --skip-ensemble --epochs 30
```

## 📊 Monitoring Training

### Real-time GPU Monitoring
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

### Training Progress
```bash
# View training logs
tail -f output/logs/training.log

# View setup log
tail -f setup_and_training.log
```

## 🔍 After Training

### Test Your Models
```bash
# Interactive prediction demo
python demo_predictions.py --model output/models/best_single_model.pth --interactive

# Batch processing
python demo_predictions.py --model output/models/best_ensemble_model.pth --batch test_images/ --output results.csv
```

### Model Analysis
```bash
# View detailed results
cat output/evaluation_results.json

# Check model sizes
ls -lh output/models/
```

## 🎉 Success Indicators

You'll know everything worked when you see:

- ✅ "Training completed successfully!"
- ✅ Models saved in `output/models/`
- ✅ Accuracy > 85% (single model) or > 90% (ensemble)
- ✅ Training curves showing convergence
- ✅ Demo runs without errors

## 🤝 Support

If you encounter issues:

1. **Check the logs**: `setup_and_training.log` contains detailed information
2. **Verify requirements**: Ensure your system meets the minimum requirements
3. **Try quick mode**: Use `--quick` flag for faster testing
4. **Reduce complexity**: Use `--skip-ensemble` for simpler training

## 📚 Next Steps

1. **Experiment with parameters**: Try different epochs, batch sizes, learning rates
2. **Analyze results**: Study the confusion matrices and training curves
3. **Deploy models**: Use the optimized models for production
4. **Extend dataset**: Add more data for better performance

---

**🚀 Ready to start? Choose your command and let the magic happen!**

```bash
# Linux/macOS
./run_task_a.sh --quick

# Windows
run_task_a.bat --quick
```
