#!/usr/bin/env python3
"""
Setup script for Gender Classification System
Handles environment verification and dependency installation
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")

    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False

    print(f"✅ Python version: {sys.version}")
    return True

def check_cuda_availability():
    """Check CUDA availability"""
    print("\n🔧 Checking CUDA availability...")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️  CUDA not available, will use CPU")
            print("   Training will be significantly slower")
    except ImportError:
        print("⚠️  PyTorch not installed yet, CUDA check will be performed after installation")

    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")

    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print("❌ requirements.txt file not found")
        return False

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def verify_installations():
    """Verify that all required packages are installed"""
    print("\n🔍 Verifying installations...")

    required_packages = [
        "torch",
        "torchvision",
        "timm",
        "numpy",
        "pandas",
        "Pillow",
        "opencv-python",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tqdm"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f"✅ {package}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False

    print("\n✅ All required packages are installed")
    return True

def test_imports():
    """Test importing key modules"""
    print("\n🧪 Testing key imports...")

    test_imports = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("timm", "Timm"),
        ("cv2", "OpenCV"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib.pyplot", "Matplotlib"),
        ("seaborn", "Seaborn"),
    ]

    failed_imports = []

    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)

    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        return False

    print("\n✅ All imports successful")
    return True

def check_dataset():
    """Check if dataset is available"""
    print("\n📁 Checking dataset structure...")

    base_path = Path(__file__).parent
    required_dirs = [
        "train/female",
        "train/male",
        "val/female",
        "val/male"
    ]

    missing_dirs = []

    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists() and full_path.is_dir():
            file_count = len(list(full_path.glob("*")))
            print(f"✅ {dir_path}: {file_count} files")
        else:
            print(f"❌ {dir_path}: not found")
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        print("Please ensure the dataset is properly extracted")
        return False

    print("\n✅ Dataset structure verified")
    return True

def create_directories():
    """Create necessary output directories"""
    print("\n📂 Creating output directories...")

    base_path = Path(__file__).parent
    directories = [
        "output",
        "output/models",
        "output/plots",
        "output/logs",
        "prediction_results"
    ]

    for dir_name in directories:
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_name}")

    return True

def run_quick_test():
    """Run a quick functionality test"""
    print("\n🚀 Running quick functionality test...")

    try:
        # Test basic PyTorch operations
        import torch
        import torch.nn as nn

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

        # Test forward pass
        x = torch.randn(1, 10)
        output = model(x)

        print("✅ PyTorch functionality test passed")

        # Test image processing
        from PIL import Image
        import numpy as np

        # Create dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        print("✅ Image processing test passed")

        # Test transforms
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transformed = transform(dummy_image)
        print("✅ Image transforms test passed")

        print("\n✅ All functionality tests passed")
        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n🚀 Next Steps:")
    print("\n1. Train the complete system:")
    print("   python train_complete_system.py")
    print("\n2. Train only single model:")
    print("   python gender_classifier.py")
    print("\n3. Run predictions on single image:")
    print("   python demo_predictions.py --model models/best_model.pth --image path/to/image.jpg")
    print("\n4. Interactive demo:")
    print("   python demo_predictions.py --model models/best_model.pth --interactive")
    print("\n5. Batch processing:")
    print("   python demo_predictions.py --model models/best_model.pth --batch path/to/folder")
    print("\n📚 For more options, see README.md")
    print("="*60)

def main():
    """Main setup function"""
    print("🔧 Gender Classification System Setup")
    print("="*50)

    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)

    # Step 2: Install requirements
    if not install_requirements():
        print("\n❌ Failed to install requirements")
        sys.exit(1)

    # Step 3: Verify installations
    if not verify_installations():
        print("\n❌ Package verification failed")
        sys.exit(1)

    # Step 4: Test imports
    if not test_imports():
        print("\n❌ Import testing failed")
        sys.exit(1)

    # Step 5: Check CUDA
    check_cuda_availability()

    # Step 6: Check dataset
    check_dataset()

    # Step 7: Create directories
    create_directories()

    # Step 8: Run functionality test
    if not run_quick_test():
        print("\n❌ Functionality test failed")
        sys.exit(1)

    # Step 9: Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
