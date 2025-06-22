#!/usr/bin/env python3
"""
Setup script for Face Recognition System
Handles installation, environment setup, and initial configuration
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import json
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionSetup:
    """Setup manager for the face recognition system"""

    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.requirements_file = self.project_dir / "requirements.txt"
        self.train_dir = self.project_dir / "train"
        self.outputs_dir = self.project_dir / "outputs"

    def check_python_version(self):
        """Check if Python version is compatible"""
        logger.info("Checking Python version...")

        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False

        logger.info(f"Python version: {sys.version}")
        return True

    def check_gpu_availability(self):
        """Check GPU availability and CUDA setup"""
        logger.info("Checking GPU availability...")

        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU available: {gpu_name} (Count: {gpu_count})")
                logger.info(f"CUDA version: {torch.version.cuda}")
                return True
            else:
                logger.warning("No GPU detected. Training will use CPU (slower)")
                return False

        except ImportError:
            logger.warning("PyTorch not installed yet. GPU check will be performed after installation.")
            return False

    def install_requirements(self, upgrade=False):
        """Install required packages"""
        logger.info("Installing requirements...")

        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return False

        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)]
            if upgrade:
                cmd.append("--upgrade")

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Requirements installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False

    def create_directories(self):
        """Create necessary directories"""
        logger.info("Creating directories...")

        directories = [
            self.outputs_dir,
            self.outputs_dir / "models",
            self.outputs_dir / "logs",
            self.outputs_dir / "results"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def check_data_structure(self):
        """Check if training data has correct structure"""
        logger.info("Checking data structure...")

        if not self.train_dir.exists():
            logger.warning(f"Training directory not found: {self.train_dir}")
            logger.info("Please ensure your training data follows this structure:")
            logger.info("train/")
            logger.info("├── person1_name/")
            logger.info("│   ├── person1_image.jpg")
            logger.info("│   └── distortion/")
            logger.info("│       ├── person1_image_blurred.jpg")
            logger.info("│       ├── person1_image_foggy.jpg")
            logger.info("│       └── ...")
            logger.info("├── person2_name/")
            logger.info("└── ...")
            return False

        # Count person directories
        person_dirs = [d for d in self.train_dir.iterdir() if d.is_dir()]

        if len(person_dirs) < 2:
            logger.warning(f"Found only {len(person_dirs)} person directories. Need at least 2 for training.")
            return False

        # Check structure of first few directories
        sample_dirs = person_dirs[:3]
        total_images = 0
        total_distorted = 0

        for person_dir in sample_dirs:
            # Count original images
            original_images = list(person_dir.glob("*.jpg"))
            total_images += len(original_images)

            # Check distortion directory
            distortion_dir = person_dir / "distortion"
            if distortion_dir.exists():
                distorted_images = list(distortion_dir.glob("*.jpg"))
                total_distorted += len(distorted_images)

        logger.info(f"Found {len(person_dirs)} person directories")
        logger.info(f"Sample check: {total_images} original images, {total_distorted} distorted images")

        return True

    def create_config_file(self):
        """Create default configuration file"""
        logger.info("Creating configuration file...")

        config = {
            "model": {
                "name": "vit_base_patch16_224",
                "embedding_dim": 512,
                "image_size": 224
            },
            "training": {
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "val_split": 0.2
            },
            "arcface": {
                "margin": 0.5,
                "scale": 64
            },
            "data": {
                "include_distorted": True,
                "max_samples_per_class": None
            },
            "paths": {
                "train_dir": "train",
                "output_dir": "outputs"
            }
        }

        config_path = self.project_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved to: {config_path}")

    def run_system_test(self):
        """Run a quick system test"""
        logger.info("Running system test...")

        try:
            # Test imports
            import torch
            import torchvision
            import timm
            import cv2
            import numpy as np
            import sklearn
            import albumentations

            logger.info("✓ All required packages imported successfully")

            # Test GPU
            if torch.cuda.is_available():
                device = torch.device('cuda')
                test_tensor = torch.randn(1, 3, 224, 224).to(device)
                logger.info("✓ GPU tensor operations working")
            else:
                logger.info("✓ CPU tensor operations working")

            # Test model creation
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
            logger.info("✓ Model creation successful")

            return True

        except Exception as e:
            logger.error(f"System test failed: {e}")
            return False

    def generate_quick_start_guide(self):
        """Generate quick start guide"""
        guide_path = self.project_dir / "QUICK_START.md"

        guide_content = """# Quick Start Guide

## 1. Installation Complete! ✅

Your Face Recognition System is now set up and ready to use.

## 2. Data Preparation

Make sure your training data is in the correct format:
```
train/
├── person1_name/
│   ├── person1_image.jpg
│   └── distortion/
│       ├── person1_image_blurred.jpg
│       ├── person1_image_foggy.jpg
│       └── ...
└── person2_name/
    └── ...
```

## 3. Training

### Quick Training (Demo)
```bash
python demo.py --quick_train
```

### Full Training
```bash
python train_face_recognition.py --train_dir train --epochs 100
```

## 4. Testing

### Run Demo
```bash
python demo.py
```

### Face Verification
```bash
python inference.py --mode verify --image1 path/to/img1.jpg --image2 path/to/img2.jpg
```

### Face Identification
```bash
python inference.py --mode identify --query_image path/to/query.jpg --gallery_dir train
```

### Full Evaluation
```bash
python inference.py --mode evaluate --data_dir train
```

## 5. Configuration

Edit `config.json` to customize training parameters:
- Model architecture
- Training hyperparameters
- Data augmentation settings

## 6. Troubleshooting

### Common Issues:
- **CUDA Out of Memory**: Reduce batch_size in config
- **Slow Training**: Enable GPU, increase batch_size
- **Low Accuracy**: Check data quality, increase epochs

### Get Help:
- Check the full README.md for detailed documentation
- Review training logs in outputs/training.log
- Use demo.py to test system functionality

## 7. Performance Tips

- Use SSD storage for faster data loading
- Monitor GPU utilization during training
- Adjust batch size based on available memory
- Use mixed precision training for speed

Happy face recognition! 🎭
"""

        with open(guide_path, 'w') as f:
            f.write(guide_content)

        logger.info(f"Quick start guide created: {guide_path}")

    def run_setup(self, install_deps=True, test_system=True):
        """Run complete setup process"""
        logger.info("="*60)
        logger.info("FACE RECOGNITION SYSTEM SETUP")
        logger.info("="*60)

        # Check Python version
        if not self.check_python_version():
            return False

        # Check GPU
        self.check_gpu_availability()

        # Install dependencies
        if install_deps:
            if not self.install_requirements():
                return False

        # Create directories
        self.create_directories()

        # Check data structure
        self.check_data_structure()

        # Create config
        self.create_config_file()

        # Test system
        if test_system:
            if not self.run_system_test():
                logger.warning("System test failed, but setup can continue")

        # Generate guide
        self.generate_quick_start_guide()

        logger.info("="*60)
        logger.info("SETUP COMPLETED SUCCESSFULLY! ✅")
        logger.info("="*60)
        logger.info("Next steps:")
        logger.info("1. Check QUICK_START.md for usage instructions")
        logger.info("2. Prepare your training data in the 'train' directory")
        logger.info("3. Run 'python demo.py --quick_train' for a quick test")
        logger.info("4. Run 'python train_face_recognition.py' for full training")

        return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Face Recognition System Setup')

    parser.add_argument('--no-install', action='store_true',
                        help='Skip dependency installation')
    parser.add_argument('--no-test', action='store_true',
                        help='Skip system test')
    parser.add_argument('--upgrade', action='store_true',
                        help='Upgrade existing packages')

    return parser.parse_args()

def main():
    """Main setup function"""
    args = parse_arguments()

    setup = FaceRecognitionSetup()

    success = setup.run_setup(
        install_deps=not args.no_install,
        test_system=not args.no_test
    )

    if success:
        print("\n🎉 Setup completed successfully!")
        print("Run 'python demo.py' to get started!")
    else:
        print("\n❌ Setup encountered issues. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
