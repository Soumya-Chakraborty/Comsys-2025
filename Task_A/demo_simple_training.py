#!/usr/bin/env python3
"""
Simple Demo Script for Gender Classification System
Author: Soumya Chakraborty
Date: 2024

This script provides a simple demonstration of the comprehensive gender classification system
with basic training and evaluation capabilities.

Mathematical Foundations:
- CNN Feature Extraction: f_cnn(x) = φ(W_conv * x + b_conv)
- Transformer Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Focal Loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)

Usage:
    python demo_simple_training.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Author Information
__author__ = "Soumya Chakraborty"
__version__ = "1.0.0"
__email__ = "soumya.chakraborty@example.com"

def check_requirements():
    """Check if all required packages are available."""
    required_packages = [
        'torch', 'torchvision', 'timm', 'numpy', 'PIL', 'cv2',
        'sklearn', 'matplotlib', 'seaborn', 'tqdm'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

    print("✅ All required packages are available")
    return True

def check_data_availability():
    """Check if the dataset is available."""
    required_dirs = [
        "Task_A/train/female",
        "Task_A/train/male",
        "Task_A/val/female",
        "Task_A/val/male"
    ]

    missing_dirs = []

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        else:
            file_count = len([f for f in os.listdir(dir_path)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✅ {dir_path}: {file_count} images")

    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        print("Please ensure the dataset is properly extracted")
        return False

    return True

def check_gpu_availability():
    """Check GPU availability and configuration."""
    print(f"\n🔧 GPU Configuration Check")
    print("-" * 40)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA available with {device_count} GPU(s)")

        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f}GB)")

        # Test basic GPU operation
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print("✅ GPU operations working correctly")
            return True
        except Exception as e:
            print(f"❌ GPU operation failed: {e}")
            return False
    else:
        print("⚠️  CUDA not available - will use CPU")
        print("   Training will be significantly slower")
        return False

def create_simple_config():
    """Create a simple configuration for demo purposes."""
    return {
        'data': {
            'train_dir': 'Task_A/train',
            'val_dir': 'Task_A/val',
            'batch_size': 16,  # Smaller batch size for demo
            'num_workers': 2,
            'image_size': 224,
            'use_face_detection': True,
            'augmentation_prob': 0.5
        },
        'training': {
            'num_epochs': 5,  # Just 5 epochs for demo
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'dropout_rate': 0.3,
            'warmup_epochs': 1,
            'gradient_clip_value': 1.0,
            'early_stopping_patience': 3
        },
        'training_phases': {
            'train_single': True,
            'train_ensemble': False,  # Skip ensemble for demo
            'model_optimization': False  # Skip optimization for demo
        },
        'model': {
            'architecture': 'hybrid',
            'num_classes': 2,
            'cnn_backbone': 'efficientnet_b3',
            'transformer_backbone': 'vit_base_patch16_224'
        },
        'loss': {
            'type': 'focal',
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'label_smoothing': 0.1
        },
        'optimization': {
            'use_mixed_precision': torch.cuda.is_available(),
            'optimizer': 'adamw',
            'scheduler': 'cosine_warm_restarts',
            'cosine_t0': 3,
            'cosine_t_mult': 2
        },
        'evaluation': {
            'save_predictions': True,
            'calculate_fairness_metrics': True,
            'generate_plots': True,
            'save_confusion_matrix': True
        },
        'output': {
            'save_dir': 'demo_output',
            'model_dir': 'demo_output/models',
            'plots_dir': 'demo_output/plots',
            'logs_dir': 'demo_output/logs'
        },
        'device': None
    }

def run_simple_demo():
    """Run a simple demonstration of the system."""
    print(f"\n🚀 SIMPLE GENDER CLASSIFICATION DEMO")
    print("=" * 60)
    print(f"Author: {__author__}")
    print(f"Version: {__version__}")
    print("=" * 60)

    # Mathematical foundations display
    print("\n📐 Mathematical Foundations:")
    print("- CNN Feature Extraction: f_cnn(x) = φ(W_conv * x + b_conv)")
    print("- Transformer Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V")
    print("- Ensemble Fusion: f_ensemble(x) = Σ(α_i * f_i(x)) where Σα_i = 1")
    print("- Focal Loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)")

    try:
        # Import the comprehensive system
        from comprehensive_gender_classifier import ComprehensiveGenderClassifier
        print("✅ Comprehensive system imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import comprehensive system: {e}")
        print("Please ensure comprehensive_gender_classifier.py is available")
        return False

    # Create demo configuration
    config = create_simple_config()
    print("✅ Demo configuration created")

    # Create output directory
    os.makedirs(config['output']['save_dir'], exist_ok=True)

    start_time = time.time()

    try:
        # Initialize classifier
        print(f"\n🔧 Initializing Gender Classifier...")
        classifier = ComprehensiveGenderClassifier(config=config)

        # Prepare data
        print(f"\n📊 Preparing data...")
        class_weights = classifier.prepare_data()
        print(f"Class weights: Female={class_weights[0]:.3f}, Male={class_weights[1]:.3f}")

        # Create model
        print(f"\n🏗️ Creating CNN-Transformer hybrid model...")
        classifier.create_model('hybrid')

        # Setup training
        print(f"\n⚙️ Setting up training components...")
        classifier.setup_training(class_weights)

        # Train model
        print(f"\n🚀 Starting training (5 epochs for demo)...")
        training_history = classifier.train()

        training_time = time.time() - start_time

        # Results summary
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Results Summary:")
        print(f"   Training Epochs: {len(training_history['train_loss'])}")
        print(f"   Best Validation Accuracy: {classifier.best_val_acc:.2f}%")
        print(f"   Final Training Loss: {training_history['train_loss'][-1]:.4f}")
        print(f"   Final Validation Loss: {training_history['val_loss'][-1]:.4f}")
        print(f"   Total Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"   Results Location: {config['output']['save_dir']}")

        # Save demo results
        demo_results = {
            'author': __author__,
            'version': __version__,
            'training_time_seconds': training_time,
            'best_accuracy': classifier.best_val_acc,
            'epochs_completed': len(training_history['train_loss']),
            'final_train_loss': training_history['train_loss'][-1],
            'final_val_loss': training_history['val_loss'][-1],
            'config': config
        }

        import json
        with open(os.path.join(config['output']['save_dir'], 'demo_results.json'), 'w') as f:
            json.dump(demo_results, f, indent=2)

        print("=" * 60)
        print(f"✨ Demo by {__author__} - System working correctly!")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    print(f"🧪 Gender Classification System Demo")
    print(f"Author: {__author__}")
    print("=" * 50)

    # Check all requirements
    print(f"\n1️⃣ Checking Requirements...")
    if not check_requirements():
        return False

    print(f"\n2️⃣ Checking Data Availability...")
    if not check_data_availability():
        print("\n💡 To get the dataset:")
        print("   1. Ensure Task_A/train and Task_A/val directories exist")
        print("   2. Each should contain 'female' and 'male' subdirectories")
        print("   3. Place image files (.jpg, .png, .jpeg) in respective folders")
        return False

    print(f"\n3️⃣ Checking GPU Availability...")
    gpu_available = check_gpu_availability()

    if not gpu_available:
        print("\n⚠️  Training will be slow on CPU. Continue anyway? (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("Demo cancelled.")
            return False

    # Run the demo
    print(f"\n4️⃣ Running Simple Training Demo...")
    success = run_simple_demo()

    if success:
        print(f"\n🎊 DEMO SUCCESSFUL!")
        print("You can now run the full system with:")
        print("python train_complete_system.py --epochs 50")
        print("\nOr with custom parameters:")
        print("python train_complete_system.py --epochs 100 --batch-size 64 --lr 1e-3")
        print("\nOr skip ensemble training:")
        print("python train_complete_system.py --skip-ensemble --output-dir custom_results")
    else:
        print(f"\n💥 DEMO FAILED!")
        print("Please check the error messages above and fix any issues.")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
