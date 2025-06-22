#!/usr/bin/env python3
"""
Simple GPU Verification Script
Quick test to ensure GPU is properly configured for the gender classification system
"""

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np

def print_header():
    """Print header"""
    print("🔧 GPU Verification for Gender Classification System")
    print("=" * 60)

def check_basic_setup():
    """Check basic PyTorch and CUDA setup"""
    print("\n🧪 Basic Setup Check")
    print("-" * 30)

    # Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {python_version}")

    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {'✅' if cuda_available else '❌'} {cuda_available}")

    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")

        # GPU information
        device_count = torch.cuda.device_count()
        print(f"GPU count: {device_count}")

        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {name} ({memory_gb:.1f}GB)")

        return True
    else:
        print("❌ CUDA not available - will use CPU")
        return False

def test_gpu_operations(device):
    """Test basic GPU operations"""
    print(f"\n🧪 Testing GPU Operations on {device}")
    print("-" * 40)

    try:
        # Create tensors
        print("Creating tensors...")
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)

        # Matrix multiplication
        start_time = time.time()
        z = torch.mm(x, y)
        operation_time = time.time() - start_time

        print(f"✅ Matrix multiplication: {operation_time:.4f}s")

        # Memory info
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3

            print(f"GPU Memory:")
            print(f"  Allocated: {allocated:.2f}GB")
            print(f"  Reserved: {reserved:.2f}GB")
            print(f"  Total: {total:.1f}GB")
            print(f"  Utilization: {(allocated/total)*100:.1f}%")

        return True

    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        return False

def test_mixed_precision(device):
    """Test mixed precision training"""
    print(f"\n🧪 Testing Mixed Precision Training")
    print("-" * 40)

    if device.type != 'cuda':
        print("⚠️  Mixed precision requires CUDA")
        return False

    try:
        # Simple model
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).to(device)

        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        # Test data
        x = torch.randn(32, 100).to(device)
        y = torch.randint(0, 10, (32,)).to(device)

        # Mixed precision forward pass
        optimizer.zero_grad()
        with autocast():
            outputs = model(x)
            loss = criterion(outputs, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print("✅ Mixed precision training works")
        return True

    except Exception as e:
        print(f"❌ Mixed precision test failed: {e}")
        return False

def test_cnn_model(device):
    """Test CNN model for image classification"""
    print(f"\n🧪 Testing CNN Model")
    print("-" * 30)

    try:
        # Simple CNN model
        class TestCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 2)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = TestCNN().to(device)

        # Test forward pass
        dummy_input = torch.randn(8, 3, 224, 224).to(device)

        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        forward_time = time.time() - start_time

        print(f"✅ CNN forward pass: {forward_time:.4f}s")
        print(f"Output shape: {output.shape}")

        # Test training step
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()

        dummy_target = torch.randint(0, 2, (8,)).to(device)

        if device.type == 'cuda':
            with autocast():
                output = model(dummy_input)
                loss = criterion(output, dummy_target)
        else:
            output = model(dummy_input)
            loss = criterion(output, dummy_target)

        loss.backward()
        optimizer.step()

        print(f"✅ CNN training step completed")
        print(f"Loss: {loss.item():.4f}")

        return True

    except Exception as e:
        print(f"❌ CNN model test failed: {e}")
        return False

def find_optimal_batch_size(device):
    """Find optimal batch size for the GPU"""
    print(f"\n🧪 Finding Optimal Batch Size")
    print("-" * 40)

    if device.type != 'cuda':
        print("⚠️  Batch size optimization only for CUDA")
        return 32

    # Simple model for testing
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 2)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimal_batch_size = 1

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        try:
            torch.cuda.empty_cache()

            # Test forward and backward pass
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
            dummy_target = torch.randint(0, 2, (batch_size,)).to(device)

            # Forward pass
            output = model(dummy_input)
            loss = criterion(output, dummy_target)

            # Backward pass
            loss.backward()
            model.zero_grad()

            optimal_batch_size = batch_size
            print(f"  Batch size {batch_size}: ✅")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch size {batch_size}: ❌ (Out of Memory)")
                break
            else:
                print(f"  Batch size {batch_size}: ❌ ({str(e)[:50]}...)")
                break

    print(f"✅ Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

def performance_benchmark(device):
    """Run performance benchmark"""
    print(f"\n🧪 Performance Benchmark")
    print("-" * 30)

    results = {}

    # Matrix multiplication benchmark
    sizes = [1000, 2000]
    for size in sizes:
        try:
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()

            a = torch.randn(size, size).to(device)
            b = torch.randn(size, size).to(device)
            c = torch.mm(a, b)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            operation_time = time.time() - start_time
            gflops = (2 * size**3) / (operation_time * 1e9)

            results[f'matmul_{size}x{size}'] = {
                'time': operation_time,
                'gflops': gflops
            }

            print(f"Matrix {size}x{size}: {operation_time:.4f}s ({gflops:.1f} GFLOPS)")

        except Exception as e:
            print(f"Matrix {size}x{size}: Failed ({e})")

    return results

def main():
    """Main verification function"""
    print_header()

    # Check basic setup
    cuda_available = check_basic_setup()

    # Select device
    if cuda_available:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    print(f"\n🎯 Selected device: {device}")

    # Run tests
    tests_passed = 0
    total_tests = 0

    # Test 1: Basic GPU operations
    total_tests += 1
    if test_gpu_operations(device):
        tests_passed += 1

    # Test 2: Mixed precision (CUDA only)
    if cuda_available:
        total_tests += 1
        if test_mixed_precision(device):
            tests_passed += 1

    # Test 3: CNN model
    total_tests += 1
    if test_cnn_model(device):
        tests_passed += 1

    # Test 4: Find optimal batch size (CUDA only)
    optimal_batch_size = 32
    if cuda_available:
        optimal_batch_size = find_optimal_batch_size(device)

    # Test 5: Performance benchmark
    performance_results = performance_benchmark(device)

    # Final summary
    print(f"\n📊 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Device: {device}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"GPU Memory: {memory_gb:.1f}GB")
        print(f"Mixed Precision: {'✅ Available' if tests_passed >= 2 else '❌ Issues'}")
        print(f"Recommended batch size: {optimal_batch_size}")
    else:
        print("GPU: Not available")
        print("Mode: CPU only")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 30)

    if tests_passed == total_tests:
        print("✅ All tests passed! Your system is ready for training.")
        if cuda_available:
            print(f"• Use batch_size={optimal_batch_size} for optimal performance")
            print("• Mixed precision training will be enabled automatically")
            print("• Expected training time: 30-45 minutes for single model")
        else:
            print("• Use smaller batch sizes for CPU training")
            print("• Expected training time: 3-4 hours for single model")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("• Consider installing CUDA-enabled PyTorch")
        print("• Ensure sufficient GPU memory")
        print("• Update GPU drivers if needed")

    print(f"\n🚀 Next Steps:")
    print("1. Run: python3 train_complete_system.py")
    print("2. Monitor with: nvidia-smi (if using GPU)")
    print("3. Check results in: output/ directory")

    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
