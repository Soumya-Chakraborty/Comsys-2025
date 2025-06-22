#!/usr/bin/env python3
"""
GPU Setup Verification and Testing Script
Tests GPU functionality, memory management, and optimization features
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class GPUTester:
    """Comprehensive GPU testing and verification system"""

    def __init__(self):
        self.device = None
        self.test_results = {}
        self.setup_device()

    def setup_device(self):
        """Setup and detect optimal device"""
        print("🔧 GPU SETUP VERIFICATION")
        print("=" * 60)

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {'✅' if cuda_available else '❌'} {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"GPU Count: {device_count}")

            # List all available GPUs
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {memory_gb:.1f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
                print(f"    Multiprocessors: {props.multi_processor_count}")

            # Select best GPU (most memory)
            best_gpu = 0
            max_memory = 0
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                if props.total_memory > max_memory:
                    max_memory = props.total_memory
                    best_gpu = i

            self.device = torch.device(f'cuda:{best_gpu}')
            torch.cuda.set_device(self.device)

            print(f"\n✅ Selected GPU {best_gpu}: {torch.cuda.get_device_name(best_gpu)}")

            # Set optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        else:
            self.device = torch.device('cpu')
            print("⚠️  Using CPU - GPU not available")

        return self.device

    def test_basic_operations(self):
        """Test basic GPU operations"""
        print(f"\n🧪 Testing Basic GPU Operations")
        print("-" * 40)

        try:
            # Test tensor creation and operations
            print("Creating tensors...")
            x = torch.randn(1000, 1000).to(self.device)
            y = torch.randn(1000, 1000).to(self.device)

            # Matrix multiplication
            start_time = time.time()
            z = torch.mm(x, y)
            operation_time = time.time() - start_time

            print(f"✅ Matrix multiplication: {operation_time:.4f}s")

            # Memory info
            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

            self.test_results['basic_operations'] = {
                'status': 'passed',
                'operation_time': operation_time,
                'memory_allocated': allocated if self.device.type == 'cuda' else 0
            }

        except Exception as e:
            print(f"❌ Basic operations failed: {e}")
            self.test_results['basic_operations'] = {'status': 'failed', 'error': str(e)}

    def test_mixed_precision(self):
        """Test mixed precision training"""
        print(f"\n🧪 Testing Mixed Precision Training")
        print("-" * 40)

        if self.device.type != 'cuda':
            print("⚠️  Mixed precision requires CUDA")
            self.test_results['mixed_precision'] = {'status': 'skipped', 'reason': 'CPU only'}
            return

        try:
            # Create simple model
            model = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            ).to(self.device)

            optimizer = optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            scaler = GradScaler()

            # Test data
            x = torch.randn(32, 100).to(self.device)
            y = torch.randint(0, 10, (32,)).to(self.device)

            # Training step with mixed precision
            start_time = time.time()

            optimizer.zero_grad()
            with autocast():
                outputs = model(x)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            amp_time = time.time() - start_time

            print(f"✅ Mixed precision training: {amp_time:.4f}s")

            # Compare with regular precision
            optimizer.zero_grad()
            start_time = time.time()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            regular_time = time.time() - start_time

            speedup = regular_time / amp_time if amp_time > 0 else 1.0
            print(f"Regular precision: {regular_time:.4f}s")
            print(f"Speedup: {speedup:.2f}x")

            self.test_results['mixed_precision'] = {
                'status': 'passed',
                'amp_time': amp_time,
                'regular_time': regular_time,
                'speedup': speedup
            }

        except Exception as e:
            print(f"❌ Mixed precision test failed: {e}")
            self.test_results['mixed_precision'] = {'status': 'failed', 'error': str(e)}

    def test_memory_management(self):
        """Test GPU memory management"""
        print(f"\n🧪 Testing Memory Management")
        print("-" * 40)

        if self.device.type != 'cuda':
            print("⚠️  Memory management test requires CUDA")
            self.test_results['memory_management'] = {'status': 'skipped', 'reason': 'CPU only'}
            return

        try:
            # Clear cache
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(self.device)

            # Allocate large tensor
            large_tensor = torch.randn(10000, 10000).to(self.device)
            after_allocation = torch.cuda.memory_allocated(self.device)

            memory_used = (after_allocation - initial_memory) / 1024**3
            print(f"Memory allocated: {memory_used:.2f}GB")

            # Clear and check
            del large_tensor
            torch.cuda.empty_cache()
            after_clear = torch.cuda.memory_allocated(self.device)

            memory_freed = (after_allocation - after_clear) / 1024**3
            print(f"Memory freed: {memory_freed:.2f}GB")

            print("✅ Memory management working correctly")

            self.test_results['memory_management'] = {
                'status': 'passed',
                'memory_used_gb': memory_used,
                'memory_freed_gb': memory_freed
            }

        except Exception as e:
            print(f"❌ Memory management test failed: {e}")
            self.test_results['memory_management'] = {'status': 'failed', 'error': str(e)}

    def test_data_loading(self):
        """Test optimized data loading"""
        print(f"\n🧪 Testing Data Loading Performance")
        print("-" * 40)

        try:
            # Create dummy dataset
            data = torch.randn(1000, 3, 224, 224)
            labels = torch.randint(0, 2, (1000,))
            dataset = TensorDataset(data, labels)

            # Test different configurations
            configs = [
                {'batch_size': 32, 'num_workers': 0, 'pin_memory': False},
                {'batch_size': 32, 'num_workers': 4, 'pin_memory': self.device.type == 'cuda'},
            ]

            results = {}

            for i, config in enumerate(configs):
                loader = DataLoader(dataset, **config)

                start_time = time.time()
                for batch_idx, (data_batch, target_batch) in enumerate(loader):
                    data_batch = data_batch.to(self.device, non_blocking=self.device.type == 'cuda')
                    target_batch = target_batch.to(self.device, non_blocking=self.device.type == 'cuda')

                    if batch_idx >= 10:  # Test first 10 batches
                        break

                load_time = time.time() - start_time
                config_name = f"Config {i+1}"
                results[config_name] = load_time

                print(f"{config_name}: {load_time:.4f}s")
                print(f"  Batch size: {config['batch_size']}")
                print(f"  Workers: {config['num_workers']}")
                print(f"  Pin memory: {config['pin_memory']}")

            print("✅ Data loading test completed")

            self.test_results['data_loading'] = {
                'status': 'passed',
                'results': results
            }

        except Exception as e:
            print(f"❌ Data loading test failed: {e}")
            self.test_results['data_loading'] = {'status': 'failed', 'error': str(e)}

    def test_model_training(self):
        """Test actual model training"""
        print(f"\n🧪 Testing Model Training")
        print("-" * 40)

        try:
            # Create a simple CNN model
            class TestCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.AdaptiveAvgPool2d(1)
                    self.fc = nn.Linear(64, 2)
                    self.dropout = nn.Dropout(0.5)

                def forward(self, x):
                    x = torch.relu(self.conv1(x))
                    x = torch.relu(self.conv2(x))
                    x = self.pool(x)
                    x = x.view(x.size(0), -1)
                    x = self.dropout(x)
                    return self.fc(x)

            model = TestCNN().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # Create dummy training data
            train_data = torch.randn(100, 3, 64, 64)
            train_labels = torch.randint(0, 2, (100,))
            train_dataset = TensorDataset(train_data, train_labels)
            train_loader = DataLoader(
                train_dataset,
                batch_size=16,
                shuffle=True,
                pin_memory=self.device.type == 'cuda'
            )

            # Training loop
            model.train()
            total_loss = 0
            start_time = time.time()

            use_amp = self.device.type == 'cuda'
            scaler = GradScaler() if use_amp else None

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device, non_blocking=self.device.type == 'cuda')
                target = target.to(self.device, non_blocking=self.device.type == 'cuda')

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

            training_time = time.time() - start_time
            avg_loss = total_loss / len(train_loader)

            print(f"✅ Training completed")
            print(f"  Time: {training_time:.4f}s")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Mixed precision: {use_amp}")

            self.test_results['model_training'] = {
                'status': 'passed',
                'training_time': training_time,
                'average_loss': avg_loss,
                'mixed_precision': use_amp
            }

        except Exception as e:
            print(f"❌ Model training test failed: {e}")
            self.test_results['model_training'] = {'status': 'failed', 'error': str(e)}

    def benchmark_performance(self):
        """Benchmark GPU performance"""
        print(f"\n🧪 GPU Performance Benchmark")
        print("-" * 40)

        if self.device.type != 'cuda':
            print("⚠️  Performance benchmark requires CUDA")
            self.test_results['benchmark'] = {'status': 'skipped', 'reason': 'CPU only'}
            return

        try:
            # Benchmark different operations
            benchmarks = {}

            # Matrix multiplication benchmark
            sizes = [1000, 2000, 4000]
            for size in sizes:
                torch.cuda.synchronize()
                start_time = time.time()

                a = torch.randn(size, size).to(self.device)
                b = torch.randn(size, size).to(self.device)
                c = torch.mm(a, b)

                torch.cuda.synchronize()
                end_time = time.time()

                operation_time = end_time - start_time
                gflops = (2 * size**3) / (operation_time * 1e9)

                benchmarks[f'matmul_{size}x{size}'] = {
                    'time': operation_time,
                    'gflops': gflops
                }

                print(f"Matrix {size}x{size}: {operation_time:.4f}s ({gflops:.2f} GFLOPS)")

            # Convolution benchmark
            conv_layer = nn.Conv2d(64, 128, 3, padding=1).to(self.device)
            input_tensor = torch.randn(32, 64, 224, 224).to(self.device)

            torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(100):
                output = conv_layer(input_tensor)

            torch.cuda.synchronize()
            conv_time = time.time() - start_time

            benchmarks['convolution'] = {'time': conv_time}
            print(f"Convolution (100 iterations): {conv_time:.4f}s")

            print("✅ Performance benchmark completed")

            self.test_results['benchmark'] = {
                'status': 'passed',
                'results': benchmarks
            }

        except Exception as e:
            print(f"❌ Performance benchmark failed: {e}")
            self.test_results['benchmark'] = {'status': 'failed', 'error': str(e)}

    def test_optimal_batch_size(self):
        """Find optimal batch size"""
        print(f"\n🧪 Finding Optimal Batch Size")
        print("-" * 40)

        if self.device.type != 'cuda':
            print("⚠️  Batch size optimization requires CUDA")
            self.test_results['optimal_batch_size'] = {'status': 'skipped', 'reason': 'CPU only'}
            return

        try:
            # Create test model
            model = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 2)
            ).to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimal_batch_size = 1

            for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                try:
                    torch.cuda.empty_cache()

                    # Test forward pass
                    dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
                    dummy_target = torch.randint(0, 2, (batch_size,)).to(self.device)

                    with torch.no_grad():
                        output = model(dummy_input)

                    # Test backward pass
                    model.train()
                    output = model(dummy_input)
                    loss = criterion(output, dummy_target)
                    loss.backward()
                    model.zero_grad()

                    optimal_batch_size = batch_size
                    print(f"Batch size {batch_size}: ✅")

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"Batch size {batch_size}: ❌ (OOM)")
                        break
                    else:
                        raise e

            print(f"✅ Optimal batch size: {optimal_batch_size}")

            self.test_results['optimal_batch_size'] = {
                'status': 'passed',
                'optimal_batch_size': optimal_batch_size
            }

        except Exception as e:
            print(f"❌ Optimal batch size test failed: {e}")
            self.test_results['optimal_batch_size'] = {'status': 'failed', 'error': str(e)}

    def run_all_tests(self):
        """Run all GPU tests"""
        print("🚀 Starting Comprehensive GPU Testing")
        print("=" * 60)

        test_functions = [
            self.test_basic_operations,
            self.test_mixed_precision,
            self.test_memory_management,
            self.test_data_loading,
            self.test_model_training,
            self.benchmark_performance,
            self.test_optimal_batch_size
        ]

        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                print(f"❌ Test {test_func.__name__} failed with exception: {e}")

        self.print_summary()
        return self.test_results

    def print_summary(self):
        """Print test summary"""
        print(f"\n📊 TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for result in self.test_results.values() if result.get('status') == 'passed')
        failed = sum(1 for result in self.test_results.values() if result.get('status') == 'failed')
        skipped = sum(1 for result in self.test_results.values() if result.get('status') == 'skipped')

        print(f"Total Tests: {len(self.test_results)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")

        print(f"\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_emoji = {'passed': '✅', 'failed': '❌', 'skipped': '⏭️'}.get(result['status'], '❓')
            print(f"  {status_emoji} {test_name}: {result['status']}")

            if result['status'] == 'failed' and 'error' in result:
                print(f"    Error: {result['error']}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 30)

        if self.device.type == 'cuda':
            if 'optimal_batch_size' in self.test_results:
                batch_size = self.test_results['optimal_batch_size'].get('optimal_batch_size', 32)
                print(f"• Use batch size: {batch_size}")

            if 'mixed_precision' in self.test_results and self.test_results['mixed_precision']['status'] == 'passed':
                print("• Enable mixed precision training for better performance")

            print("• Use pin_memory=True and non_blocking=True for data loading")
            print("• Clear GPU cache periodically during long training runs")
            print("• Monitor GPU memory usage to avoid OOM errors")
        else:
            print("• Consider using a GPU for significantly faster training")
            print("• Reduce batch size and model complexity for CPU training")

    def save_results(self, filename="gpu_test_results.json"):
        """Save test results to file"""
        import json

        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        print(f"📁 Test results saved to: {filename}")

def main():
    """Main function"""
    print("🔧 GPU Setup and Performance Testing Tool")
    print("This script will test your GPU setup and optimization")
    print()

    tester = GPUTester()
    results = tester.run_all_tests()

    # Save results
    tester.save_results()

    print(f"\n🎉 GPU testing completed!")
    print("You can now proceed with training the gender classification model.")

    return results

if __name__ == "__main__":
    main()
