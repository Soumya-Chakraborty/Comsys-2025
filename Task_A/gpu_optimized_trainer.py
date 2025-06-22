#!/usr/bin/env python3
"""
GPU-Optimized Gender Classification Training System
Enhanced device management, memory optimization, and GPU acceleration
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import json
import warnings
import gc
import psutil
from pathlib import Path
from datetime import datetime

# Try to import optional GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import py3nvml.py3nvml as nvml
    PY3NVML_AVAILABLE = True
except ImportError:
    PY3NVML_AVAILABLE = False

warnings.filterwarnings('ignore')

class GPUManager:
    """Enhanced GPU management and monitoring"""

    def __init__(self):
        self.device = self.setup_device()
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        self.memory_tracker = []

        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except:
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False

    def setup_device(self):
        """Setup optimal device configuration"""
        if torch.cuda.is_available():
            # Select the best GPU
            device_count = torch.cuda.device_count()
            print(f"🔧 Found {device_count} CUDA device(s)")

            # Print GPU information
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")

            # Select GPU with most memory
            best_gpu = 0
            max_memory = 0
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                if props.total_memory > max_memory:
                    max_memory = props.total_memory
                    best_gpu = i

            device = torch.device(f'cuda:{best_gpu}')
            torch.cuda.set_device(device)

            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            print(f"✅ Using GPU {best_gpu}: {torch.cuda.get_device_name(best_gpu)}")
            print(f"   Memory: {torch.cuda.get_device_properties(best_gpu).total_memory / 1024**3:.1f} GB")

        else:
            device = torch.device('cpu')
            print("⚠️  CUDA not available, using CPU")
            print("   Training will be significantly slower")

        return device

    def get_gpu_memory_info(self):
        """Get current GPU memory usage"""
        if self.device.type != 'cuda':
            return None

        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        cached = torch.cuda.memory_reserved(self.device) / 1024**3
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3

        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'total_gb': total,
            'utilization': (allocated / total) * 100 if total > 0 else 0
        }

    def clear_gpu_cache(self):
        """Clear GPU cache"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    def optimize_memory_usage(self):
        """Optimize GPU memory usage"""
        if self.device.type == 'cuda':
            # Clear cache
            self.clear_gpu_cache()

            # Set memory fraction if needed
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            if total_memory < 8 * 1024**3:  # Less than 8GB
                torch.cuda.set_per_process_memory_fraction(0.8)
                print("🔧 Set GPU memory fraction to 80% for low-memory GPU")

    def monitor_memory(self, step_name=""):
        """Monitor and log memory usage"""
        memory_info = self.get_gpu_memory_info()
        if memory_info:
            self.memory_tracker.append({
                'step': step_name,
                'timestamp': time.time(),
                'allocated_gb': memory_info['allocated_gb'],
                'utilization': memory_info['utilization']
            })

            if memory_info['utilization'] > 90:
                print(f"⚠️  High GPU memory usage: {memory_info['utilization']:.1f}% at {step_name}")

    def get_optimal_batch_size(self, model, input_shape=(3, 224, 224), max_batch_size=128):
        """Find optimal batch size for the model"""
        if self.device.type != 'cuda':
            return 32  # Default for CPU

        print("🔍 Finding optimal batch size...")

        model.eval()
        optimal_batch_size = 1

        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            if batch_size > max_batch_size:
                break

            try:
                # Clear cache before test
                self.clear_gpu_cache()

                # Create dummy batch
                dummy_input = torch.randn(batch_size, *input_shape).to(self.device)

                # Test forward pass
                with torch.no_grad():
                    _ = model(dummy_input)

                # Test backward pass
                model.train()
                dummy_target = torch.randint(0, 2, (batch_size,)).to(self.device)
                output = model(dummy_input)
                loss = nn.CrossEntropyLoss()(output, dummy_target)
                loss.backward()

                optimal_batch_size = batch_size
                print(f"   Batch size {batch_size}: ✅")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   Batch size {batch_size}: ❌ (OOM)")
                    break
                else:
                    raise e
            finally:
                model.zero_grad()
                self.clear_gpu_cache()

        print(f"✅ Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size

class OptimizedDataLoader:
    """GPU-optimized data loader with prefetching"""

    def __init__(self, gpu_manager):
        self.gpu_manager = gpu_manager
        self.device = gpu_manager.device

    def create_optimized_loader(self, dataset, batch_size, shuffle=True, num_workers=None):
        """Create optimized data loader"""

        # Auto-determine optimal number of workers
        if num_workers is None:
            num_workers = min(8, os.cpu_count())
            if self.device.type == 'cuda':
                # For GPU, use more workers for better throughput
                num_workers = min(12, os.cpu_count())

        # Optimize data loader settings
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': self.device.type == 'cuda',
            'persistent_workers': num_workers > 0,
            'prefetch_factor': 2 if num_workers > 0 else 2,
        }

        # Add GPU-specific optimizations
        if self.device.type == 'cuda':
            loader_kwargs.update({
                'pin_memory_device': str(self.device),
                'non_blocking': True,
            })

        print(f"🔧 DataLoader config: batch_size={batch_size}, num_workers={num_workers}")

        return DataLoader(dataset, **loader_kwargs)

class GPUOptimizedModel(nn.Module):
    """Base model class with GPU optimizations"""

    def __init__(self, gpu_manager, enable_checkpointing=True):
        super().__init__()
        self.gpu_manager = gpu_manager
        self.device = gpu_manager.device
        self.enable_checkpointing = enable_checkpointing and self.device.type == 'cuda'

    def forward(self, x):
        """Forward pass with optional gradient checkpointing"""
        if self.enable_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x):
        """Implement this in subclasses"""
        raise NotImplementedError

    def optimize_for_inference(self):
        """Optimize model for inference"""
        self.eval()

        if self.device.type == 'cuda':
            # Enable inference optimizations
            torch.backends.cudnn.benchmark = True

            # Try to use torch.jit compilation
            try:
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                traced_model = torch.jit.trace(self, dummy_input)
                return traced_model
            except:
                print("⚠️  JIT compilation failed, using original model")
                return self

        return self

class GPUOptimizedTrainer:
    """GPU-optimized training system"""

    def __init__(self, model, train_loader, val_loader, config=None):
        self.gpu_manager = GPUManager()
        self.device = self.gpu_manager.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or self.get_default_config()

        # Initialize mixed precision training
        self.use_amp = self.device.type == 'cuda' and self.config.get('use_mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Setup optimizer and scheduler
        self.setup_training_components()

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'gpu_memory': []
        }

        print(f"🚀 GPU-Optimized Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {self.use_amp}")
        print(f"   Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def get_default_config(self):
        """Get default training configuration"""
        return {
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'use_mixed_precision': True,
            'gradient_clip_value': 1.0,
            'warmup_epochs': 5,
            'save_every_n_epochs': 10,
            'early_stopping_patience': 15,
            'scheduler_type': 'cosine_warm_restarts'
        }

    def setup_training_components(self):
        """Setup optimizer, scheduler, and loss function"""

        # Optimizer with different learning rates for different parts
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if any(x in name.lower() for x in ['backbone', 'features', 'encoder']):
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config['learning_rate'] * 0.1},
            {'params': head_params, 'lr': self.config['learning_rate']}
        ], weight_decay=self.config['weight_decay'])

        # Scheduler
        if self.config['scheduler_type'] == 'cosine_warm_restarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=50, eta_min=1e-6
            )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def train_epoch(self):
        """Train for one epoch with GPU optimizations"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Monitor memory before training
        self.gpu_manager.monitor_memory("epoch_start")

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Move data to device with non-blocking transfer
            if self.device.type == 'cuda':
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
            else:
                images, labels = images.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.get('gradient_clip_value'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip_value']
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.config.get('gradient_clip_value'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip_value']
                    )

                self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            # Memory monitoring every 100 batches
            if batch_idx % 100 == 0:
                self.gpu_manager.monitor_memory(f"batch_{batch_idx}")

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validation with memory optimization"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                if self.device.type == 'cuda':
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                else:
                    images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Store for detailed metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

    def train(self, num_epochs=50, save_dir="models"):
        """Complete training loop with GPU optimizations"""

        print(f"🚀 Starting GPU-optimized training for {num_epochs} epochs")

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Optimize memory before training
        self.gpu_manager.optimize_memory_usage()

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Training
            train_loss, train_acc = self.train_epoch()

            # Validation
            val_loss, val_acc, val_preds, val_labels, val_probs = self.validate()

            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rate'].append(current_lr)

            # Record GPU memory usage
            memory_info = self.gpu_manager.get_gpu_memory_info()
            if memory_info:
                self.training_history['gpu_memory'].append(memory_info['utilization'])

            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.2e}")

            if memory_info:
                print(f"GPU Memory: {memory_info['allocated_gb']:.1f}GB / "
                      f"{memory_info['total_gb']:.1f}GB ({memory_info['utilization']:.1f}%)")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(
                    os.path.join(save_dir, "best_model.pth"),
                    epoch, val_acc, is_best=True
                )
                print(f"✅ New best model saved! Accuracy: {val_acc:.2f}%")

            # Regular checkpoint saving
            if (epoch + 1) % self.config.get('save_every_n_epochs', 10) == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
                    epoch, val_acc
                )

            # Memory cleanup
            if (epoch + 1) % 5 == 0:
                self.gpu_manager.clear_gpu_cache()

        total_time = time.time() - start_time

        print(f"\n🎉 Training completed!")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        # Save final model and training history
        self.save_checkpoint(
            os.path.join(save_dir, "final_model.pth"),
            num_epochs - 1, self.best_val_acc, save_history=True
        )

        return self.training_history

    def save_checkpoint(self, path, epoch, val_acc, is_best=False, save_history=False):
        """Save model checkpoint with training state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'val_acc': val_acc,
            'config': self.config,
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if save_history:
            checkpoint['training_history'] = self.training_history

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.current_epoch = checkpoint['epoch']

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        print(f"✅ Checkpoint loaded from epoch {self.current_epoch}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

    def plot_training_curves(self, save_path="training_curves.png"):
        """Plot training curves with GPU memory usage"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.training_history['train_loss']) + 1)

        # Loss curves
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(epochs, self.training_history['train_acc'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.training_history['val_acc'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate
        axes[1, 0].plot(epochs, self.training_history['learning_rate'], 'g-')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

        # GPU memory usage
        if self.training_history['gpu_memory']:
            axes[1, 1].plot(epochs, self.training_history['gpu_memory'], 'm-')
            axes[1, 1].set_title('GPU Memory Utilization')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Memory Usage (%)')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'GPU Memory\nNot Available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('GPU Memory Utilization')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Training curves saved to: {save_path}")

def test_gpu_optimization():
    """Test GPU optimization functionality"""
    print("🧪 Testing GPU Optimization System")
    print("=" * 50)

    # Test GPU manager
    gpu_manager = GPUManager()

    # Test memory monitoring
    memory_info = gpu_manager.get_gpu_memory_info()
    if memory_info:
        print(f"GPU Memory: {memory_info['allocated_gb']:.2f}GB allocated, "
              f"{memory_info['utilization']:.1f}% utilization")

    # Test simple model
    class TestModel(GPUOptimizedModel):
        def __init__(self, gpu_manager):
            super().__init__(gpu_manager)
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 2)

        def _forward_impl(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = TestModel(gpu_manager)

    # Test optimal batch size detection
    optimal_batch_size = gpu_manager.get_optimal_batch_size(model)

    print(f"✅ GPU optimization test completed")
    print(f"Optimal batch size: {optimal_batch_size}")

if __name__ == "__main__":
    test_gpu_optimization()
