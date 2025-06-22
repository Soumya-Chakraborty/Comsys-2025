#!/usr/bin/env python3
"""
Installation Test Script
Simple script to verify that the face recognition system is properly installed
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

def test_python_version():
    """Test Python version compatibility"""
    print("Testing Python version...")

    if sys.version_info >= (3, 8):
        print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True
    else:
        print(f"✗ Python version too old: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print("  Required: Python 3.8 or higher")
        return False

def test_package_imports():
    """Test importing required packages"""
    print("\nTesting package imports...")

    required_packages = [
        'torch',
        'torchvision',
        'timm',
        'cv2',
        'numpy',
        'pandas',
        'sklearn',
        'PIL',
        'matplotlib',
        'seaborn',
        'tqdm',
        'albumentations',
        'pathlib',
        'json',
        'logging'
    ]

    failed_imports = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package} - {e}")
            failed_imports.append(package)

    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False

    return True

def test_torch_functionality():
    """Test PyTorch functionality"""
    print("\nTesting PyTorch functionality...")

    try:
        import torch

        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)
        print("✓ Basic tensor operations")

        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

            # Test GPU operations
            try:
                x_gpu = x.cuda()
                print("✓ GPU tensor operations")
            except Exception as e:
                print(f"✗ GPU tensor operations failed: {e}")
        else:
            print("⚠ CUDA not available (will use CPU)")

        return True

    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")

    try:
        import timm

        # Test creating a ViT model
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
        print("✓ Vision Transformer model creation")

        # Test forward pass
        import torch
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"✓ Model forward pass: {output.shape}")

        return True

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_image_processing():
    """Test image processing functionality"""
    print("\nTesting image processing...")

    try:
        import cv2
        import numpy as np
        from PIL import Image
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Test OpenCV
        gray = cv2.cvtColor(dummy_image, cv2.COLOR_RGB2GRAY)
        print("✓ OpenCV operations")

        # Test PIL
        pil_image = Image.fromarray(dummy_image)
        print("✓ PIL operations")

        # Test Albumentations
        transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        transformed = transform(image=dummy_image)
        print("✓ Albumentations transforms")

        return True

    except Exception as e:
        print(f"✗ Image processing test failed: {e}")
        return False

def test_file_structure():
    """Test file structure"""
    print("\nTesting file structure...")

    required_files = [
        'train_face_recognition.py',
        'inference.py',
        'demo.py',
        'setup.py',
        'requirements.txt',
        'README.md',
        'config.json'
    ]

    missing_files = []
    current_dir = Path(__file__).parent

    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name}")
            missing_files.append(file_name)

    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        return False

    return True

def test_data_directory():
    """Test data directory structure"""
    print("\nTesting data directory...")

    current_dir = Path(__file__).parent
    train_dir = current_dir / 'train'

    if not train_dir.exists():
        print("⚠ 'train' directory not found")
        print("  This is normal if you haven't added your training data yet")
        return True

    # Count person directories
    person_dirs = [d for d in train_dir.iterdir() if d.is_dir()]

    if len(person_dirs) == 0:
        print("⚠ No person directories found in 'train'")
        return True

    print(f"✓ Found {len(person_dirs)} person directories")

    # Check structure of first directory
    if person_dirs:
        first_person = person_dirs[0]
        images = list(first_person.glob("*.jpg"))
        distortion_dir = first_person / "distortion"

        print(f"✓ Sample person: {first_person.name}")
        print(f"  - Original images: {len(images)}")

        if distortion_dir.exists():
            distorted_images = list(distortion_dir.glob("*.jpg"))
            print(f"  - Distorted images: {len(distorted_images)}")
        else:
            print("  - No distortion directory")

    return True

def test_project_imports():
    """Test importing project modules"""
    print("\nTesting project module imports...")

    try:
        # Add current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))

        # Test importing main modules
        from train_face_recognition import Config, FaceRecognitionTrainer
        print("✓ train_face_recognition module")

        from inference import FaceRecognitionInference
        print("✓ inference module")

        return True

    except Exception as e:
        print(f"✗ Project imports failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Face Recognition System Installation Test")
    print("=" * 50)

    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_package_imports),
        ("PyTorch Functionality", test_torch_functionality),
        ("Model Creation", test_model_creation),
        ("Image Processing", test_image_processing),
        ("File Structure", test_file_structure),
        ("Data Directory", test_data_directory),
        ("Project Imports", test_project_imports)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'─' * 20} {test_name} {'─' * 20}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    if failed == 0:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Add your training data to the 'train' directory")
        print("2. Run 'python demo.py --quick_train' for a quick test")
        print("3. Run 'python train_face_recognition.py' for full training")
        return True
    else:
        print(f"\n⚠ {failed} test(s) failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Run 'pip install -r requirements.txt' to install missing packages")
        print("- Check that you're using Python 3.8 or higher")
        print("- Ensure all required files are present")
        return False

def main():
    """Main function"""
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
