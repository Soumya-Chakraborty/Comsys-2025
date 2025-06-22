# Comprehensive Gender Classification System - Usage Guide

**Author:** Soumya Chakraborty  
**Version:** 2.0.0  
**Email:** soumya.chakraborty@example.com  
**License:** MIT  

## 🚀 Overview

This comprehensive gender classification system implements state-of-the-art deep learning techniques for binary gender classification from facial images. The system combines CNN and Transformer architectures with advanced optimization techniques.

### 📐 Mathematical Foundations

The system is built on rigorous mathematical principles:

#### 1. CNN Feature Extraction
```
f_cnn(x) = φ(W_conv * x + b_conv)
```
Where:
- `x`: Input image tensor
- `W_conv`: Convolution weight matrix
- `b_conv`: Bias term
- `φ`: Activation function (ReLU, etc.)

#### 2. Transformer Attention Mechanism
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```
Where:
- `Q`: Query matrix [batch_size, seq_len, d_k]
- `K`: Key matrix [batch_size, seq_len, d_k]
- `V`: Value matrix [batch_size, seq_len, d_v]
- `d_k`: Dimension of key vectors (for scaling)

#### 3. Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

#### 4. Ensemble Fusion
```
f_ensemble(x) = Σ(α_i * f_i(x)) where Σα_i = 1
```
Where:
- `α_i`: Learned attention weights for model i
- `f_i(x)`: Output of individual model i

#### 5. Focal Loss for Class Imbalance
```
FL(p_t) = -α_t(1-p_t)^γ log(p_t)
```
Where:
- `p_t`: Predicted probability for the true class
- `α_t`: Weighting factor for class t
- `γ`: Focusing parameter (reduces loss for well-classified examples)

#### 6. Knowledge Distillation
```
L = αL_CE(y, student_logits) + (1-α)τ²L_KD(soft_teacher, soft_student)
```
Where:
- `L_CE`: Cross-entropy loss with hard targets
- `L_KD`: KL divergence loss with soft targets
- `τ`: Temperature parameter for softening distributions
- `α`: Weighting factor between hard and soft targets

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ GPU memory (16GB+ for ensemble training)

### Quick Installation
```bash
# Clone and navigate to the project
cd ComsysHackathon/Task_A

# Automated installation (Linux/macOS)
chmod +x install_gpu.sh
./install_gpu.sh

# Manual installation
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### Verification
```bash
# Quick GPU test
python3 verify_gpu.py

# Comprehensive system test
python3 test_gpu_setup.py

# Simple demo
python3 demo_simple_training.py
```

## 📊 Dataset Structure

The system expects the following directory structure:

```
Task_A/
├── train/
│   ├── female/    # Female training images
│   └── male/      # Male training images
└── val/
    ├── female/    # Female validation images
    └── male/      # Male validation images
```

**Current Dataset Statistics:**
- Training: 303 female, 1623 male (5.4:1 imbalance)
- Validation: 79 female, 343 male (4.3:1 imbalance)

## 🎯 Command-Line Usage

### Basic Training Commands

#### 1. Train with Custom Parameters
```bash
# Train with custom epochs, batch size, and learning rate
python train_complete_system.py --epochs 100 --batch-size 64 --lr 1e-3

# Mathematical significance:
# - Higher epochs: More gradient descent iterations
# - Larger batch size: More stable gradients, requires more GPU memory
# - Learning rate: Step size in gradient descent optimization
```

#### 2. Train Only Specific Components
```bash
# Skip ensemble training (faster, less memory)
python train_complete_system.py --skip-ensemble --output-dir custom_results

# Skip single model training (only ensemble)
python train_complete_system.py --skip-single --epochs 50

# Skip all optimization phases
python train_complete_system.py --skip-optimization
```

#### 3. Use Configuration File
```bash
# Use predefined configuration
python train_complete_system.py --config config_template.json

# Override specific parameters
python train_complete_system.py --config config_template.json --epochs 150 --lr 5e-5
```

### Advanced Training Options

#### Model Architecture Selection
```bash
# Train CNN-Transformer hybrid (default)
python train_complete_system.py --architecture hybrid

# Train ensemble model
python train_complete_system.py --architecture ensemble

# Custom backbone architectures
python train_complete_system.py \
    --cnn-backbone efficientnet_b4 \
    --transformer-backbone vit_large_patch16_224
```

#### Loss Function Options
```bash
# Use Focal Loss (recommended for imbalanced data)
python train_complete_system.py --loss focal

# Use standard Cross-Entropy
python train_complete_system.py --loss cross_entropy

# Use Label Smoothing
python train_complete_system.py --loss label_smoothing
```

#### Optimization Strategies
```bash
# Different optimizers
python train_complete_system.py --optimizer adamw    # Default, best performance
python train_complete_system.py --optimizer adam     # Alternative

# Learning rate schedulers
python train_complete_system.py --scheduler cosine_warm_restarts  # Default
python train_complete_system.py --scheduler cosine               # Simple cosine
python train_complete_system.py --scheduler step                 # Step decay
```

#### Advanced Parameters
```bash
# Fine-tune training hyperparameters
python train_complete_system.py \
    --dropout 0.4 \
    --weight-decay 1e-5 \
    --grad-clip 0.5 \
    --no-mixed-precision

# Data processing options
python train_complete_system.py \
    --no-face-detection \
    --train-dir custom_train \
    --val-dir custom_val
```

## 🏗️ Model Architectures

### 1. CNN-Transformer Hybrid

**Architecture Overview:**
```
Input Image [3×224×224]
    ↓
CNN Backbone (EfficientNet-B3)
    ↓
CNN Features [1536-d]
    ↓
Vision Transformer (ViT)
    ↓
Transformer Features [768-d]
    ↓
Attention Fusion Layer
    ↓
Classification Head
    ↓
Output [2 classes]
```

**Mathematical Flow:**
1. **CNN Feature Extraction**: `f_cnn = EfficientNet(x)`
2. **Transformer Feature Extraction**: `f_vit = ViT(x)`
3. **Feature Fusion**: `f_fused = AttentionFusion(f_cnn ⊕ f_vit)`
4. **Classification**: `y = Classifier(f_fused)`

### 2. Ensemble Architecture

**Four Base Models:**
- EfficientNet-B3
- ResNet-50
- Vision Transformer (ViT)
- ConvNeXt

**Fusion Methods:**
1. **Weighted Average**: `f_ensemble = Σ(w_i * f_i(x))`
2. **Meta-Learning**: Additional network learns optimal combination
3. **Attention-Based**: Dynamic weighting based on input

## 📈 Training Pipeline

### Phase 1: Data Preparation
- Advanced data augmentation
- Face detection and cropping
- Class imbalance handling with weighted sampling
- Mathematical normalization: `x_norm = (x - μ) / σ`

### Phase 2: Model Training
- Mixed precision training (30% speedup)
- Gradient clipping for stability
- Learning rate scheduling
- Early stopping with patience

### Phase 3: Optimization
- Knowledge distillation for model compression
- Dynamic quantization (4x size reduction)
- Model pruning for faster inference

### Phase 4: Evaluation
- Comprehensive metrics calculation
- Bias and fairness analysis
- Confusion matrix generation
- Performance visualization

## 🔧 Configuration Options

### Sample Configuration File
```json
{
  "data": {
    "train_dir": "Task_A/train",
    "val_dir": "Task_A/val",
    "batch_size": 32,
    "num_workers": 4,
    "image_size": 224,
    "use_face_detection": true,
    "augmentation_prob": 0.8
  },
  "training": {
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "dropout_rate": 0.3,
    "gradient_clip_value": 1.0,
    "early_stopping_patience": 15
  },
  "model": {
    "architecture": "hybrid",
    "num_classes": 2,
    "cnn_backbone": "efficientnet_b3",
    "transformer_backbone": "vit_base_patch16_224"
  },
  "loss": {
    "type": "focal",
    "focal_alpha": 1.0,
    "focal_gamma": 2.0
  },
  "optimization": {
    "use_mixed_precision": true,
    "optimizer": "adamw",
    "scheduler": "cosine_warm_restarts"
  }
}
```

## 🚀 Performance Optimization

### GPU Memory Optimization
```bash
# For limited GPU memory (4-6GB)
python train_complete_system.py --batch-size 8 --skip-ensemble

# For high-end GPUs (16GB+)
python train_complete_system.py --batch-size 128 --architecture ensemble

# Memory monitoring during training
nvidia-smi -l 1
```

### Training Speed Optimization
```bash
# Enable all optimizations
python train_complete_system.py \
    --batch-size 64 \
    --optimizer adamw \
    --scheduler cosine_warm_restarts

# Multi-GPU training (if available)
CUDA_VISIBLE_DEVICES=0,1 python train_complete_system.py --batch-size 128
```

## 📊 Evaluation and Metrics

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: TP / (TP + FP) per class
- **Recall**: TP / (TP + FN) per class
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Fairness Metrics
- **Demographic Parity**: Equal prediction rates across groups
- **Equalized Odds**: Equal true positive rates across groups
- **Accuracy Difference**: Performance gap between demographic groups

### Mathematical Formulation
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

## 🎯 Making Predictions

### Single Image Prediction
```bash
# Predict single image
python demo_predictions.py \
    --model output/models/best_single_model.pth \
    --image path/to/image.jpg

# Interactive mode
python demo_predictions.py \
    --model output/models/best_model.pth \
    --interactive
```

### Batch Processing
```bash
# Process entire folder
python demo_predictions.py \
    --model output/models/best_ensemble_model.pth \
    --batch path/to/image/folder \
    --output results.csv
```

### Python API Usage
```python
from comprehensive_gender_classifier import ComprehensiveGenderClassifier

# Initialize classifier
classifier = ComprehensiveGenderClassifier()
classifier.load_model('path/to/model.pth')

# Make prediction
gender, confidence, probabilities = classifier.predict_single_image('image.jpg')
print(f"Predicted: {gender} (Confidence: {confidence:.2%})")
```

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solutions:
python train_complete_system.py --batch-size 8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

#### 2. Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Optimize data loading
python train_complete_system.py --config optimized_config.json
```

#### 3. Poor Performance
```bash
# Increase training epochs
python train_complete_system.py --epochs 100

# Use ensemble model
python train_complete_system.py --architecture ensemble

# Adjust learning rate
python train_complete_system.py --lr 5e-5
```

## 📈 Expected Results

### Performance Benchmarks
| Model Type | Training Time | Accuracy | GPU Memory |
|------------|---------------|----------|------------|
| Single CNN-Transformer | 30-45 min | 85-90% | 4-6GB |
| Ensemble (4 models) | 2-3 hours | 90-95% | 8-12GB |
| Optimized Student | 1 hour | 85-90% | 2-4GB |

### Mathematical Convergence
- **Training Loss**: Should decrease exponentially following `L(t) = L_0 * e^(-λt)`
- **Validation Accuracy**: Should increase logarithmically `A(t) = A_max * (1 - e^(-μt))`
- **Learning Rate**: Follows cosine annealing `lr(t) = lr_min + (lr_max - lr_min) * (1 + cos(πt/T)) / 2`

## 🎓 Advanced Usage

### Custom Loss Functions
```python
# Implement custom loss in configuration
{
  "loss": {
    "type": "custom",
    "class_weights": [0.8, 0.2],
    "label_smoothing": 0.1,
    "focal_gamma": 2.5
  }
}
```

### Custom Architectures
```python
# Extend the system with custom models
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Custom architecture implementation
        
# Register and use in training pipeline
```

### Hyperparameter Optimization
```bash
# Grid search over hyperparameters
for lr in 1e-5 1e-4 1e-3; do
  for bs in 16 32 64; do
    python train_complete_system.py --lr $lr --batch-size $bs --output-dir "results_lr${lr}_bs${bs}"
  done
done
```

## 📚 Research and Citations

### Theoretical Foundations
This implementation is based on several key research papers:

1. **EfficientNet**: Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
2. **Vision Transformer**: Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
3. **Focal Loss**: Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection"
4. **Knowledge Distillation**: Hinton, G., et al. (2015). "Distilling the Knowledge in a Neural Network"

### Mathematical Rigor
All algorithms implemented follow rigorous mathematical principles:
- Gradient descent optimization theory
- Information theory for loss functions
- Statistical learning theory for generalization
- Differential geometry for manifold learning

## 🤝 Contributing

### Code Style
- Follow PEP 8 style guidelines
- Include comprehensive docstrings with mathematical formulations
- Add type hints for all function parameters
- Include unit tests for new functionality

### Mathematical Documentation
- Document all mathematical formulations in LaTeX notation
- Provide theoretical justification for algorithm choices
- Include convergence analysis for optimization procedures

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **Author**: Soumya Chakraborty
- **PyTorch Team**: For the excellent deep learning framework
- **Timm Library**: For pretrained model implementations
- **Research Community**: For advancing the field of computer vision

---

**For technical support or questions, contact: soumya.chakraborty@example.com**

*System designed and implemented by Soumya Chakraborty with mathematical rigor and engineering excellence.*