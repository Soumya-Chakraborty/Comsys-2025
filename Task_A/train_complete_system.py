#!/usr/bin/env python3
"""
Complete Gender Classification Training System
Author: Soumya Chakraborty
Date: 2024

This script provides a comprehensive command-line interface for training gender classification models
with advanced features including CNN-Transformer hybrid architectures, ensemble methods, and
GPU optimization.

Mathematical Foundations:
- CNN Feature Extraction: f_cnn(x) = φ(W_conv * x + b_conv)
- Transformer Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Ensemble Fusion: f_ensemble(x) = Σ(α_i * f_i(x)) where Σα_i = 1
- Focal Loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
- Knowledge Distillation: L = αL_CE + (1-α)τ²L_KD

Usage Examples:
# Train with custom parameters
python train_complete_system.py --epochs 100 --batch-size 64 --lr 1e-3

# Train only specific components
python train_complete_system.py --skip-ensemble --output-dir custom_results

# Use configuration file
python train_complete_system.py --config config_template.json
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

# Import our comprehensive system
try:
    from comprehensive_gender_classifier import (
        ComprehensiveGenderClassifier,
        CNNTransformerHybrid,
        EnsembleModel,
        GenderDataset,
        MathematicalFoundations,
        set_random_seeds
    )
except ImportError:
    print("Error: Could not import comprehensive_gender_classifier module")
    print("Please ensure the comprehensive_gender_classifier.py file is in the same directory")
    sys.exit(1)

warnings.filterwarnings('ignore')

# Author Information
__author__ = "Soumya Chakraborty"
__version__ = "2.0.0"
__email__ = "soumya.chakraborty@example.com"
__license__ = "MIT"

class AdvancedTrainingSystem:
    """
    Advanced Training System for Gender Classification.
    Author: Soumya Chakraborty

    This class orchestrates the complete training pipeline including:
    1. Data preparation with advanced augmentation
    2. Model architecture selection and configuration
    3. Training with GPU optimization and mixed precision
    4. Ensemble methods and model fusion
    5. Comprehensive evaluation and bias analysis
    6. Model compression and deployment optimization
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the advanced training system.

        Args:
            config: Complete configuration dictionary
        """
        self.config = config
        self.start_time = time.time()
        self.results = {}

        # Setup logging
        self._setup_logging()

        # Initialize random seeds
        set_random_seeds(42)

        # Log system information
        self._log_system_info()

        logging.info(f"Advanced Training System initialized by {__author__}")
        logging.info(f"Version: {__version__}")

    def _setup_logging(self) -> None:
        """Setup comprehensive logging system."""
        log_dir = Path(self.config['output']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Configure logging with both file and console handlers
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        # Log configuration
        logging.info(f"Logging initialized. Log file: {log_file}")

    def _log_system_info(self) -> None:
        """Log comprehensive system information."""
        logging.info("="*80)
        logging.info("SYSTEM INFORMATION")
        logging.info("="*80)
        logging.info(f"Author: {__author__}")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            logging.info(f"CUDA available: True")
            logging.info(f"CUDA version: {torch.version.cuda}")
            logging.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logging.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logging.info("CUDA available: False")

        logging.info("="*80)

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Returns:
            Dictionary containing all results and metrics
        """
        try:
            # Phase 1: Data Preparation
            self._run_data_preparation()

            # Phase 2: Single Model Training
            if self.config['training_phases']['train_single']:
                self._run_single_model_training()

            # Phase 3: Ensemble Training
            if self.config['training_phases']['train_ensemble']:
                self._run_ensemble_training()

            # Phase 4: Model Optimization
            if self.config['training_phases']['model_optimization']:
                self._run_model_optimization()

            # Phase 5: Comprehensive Evaluation
            self._run_comprehensive_evaluation()

            # Phase 6: Generate Reports
            self._generate_comprehensive_reports()

            # Calculate total execution time
            total_time = time.time() - self.start_time
            self.results['execution_time'] = {
                'total_seconds': total_time,
                'total_minutes': total_time / 60,
                'total_hours': total_time / 3600
            }

            logging.info(f"\n🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
            logging.info(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

            return self.results

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _run_data_preparation(self) -> None:
        """Run data preparation phase."""
        logging.info("\n" + "="*60)
        logging.info("PHASE 1: DATA PREPARATION")
        logging.info("="*60)

        # Initialize the classifier for data preparation
        self.classifier = ComprehensiveGenderClassifier(
            config=self.config,
            device=self.config.get('device')
        )

        # Prepare data loaders
        class_weights = self.classifier.prepare_data()

        self.results['data_preparation'] = {
            'class_weights': class_weights.tolist(),
            'train_batches': len(self.classifier.train_loader),
            'val_batches': len(self.classifier.val_loader),
            'batch_size': self.config['data']['batch_size']
        }

        logging.info("✅ Data preparation completed successfully")

    def _run_single_model_training(self) -> None:
        """Run single model training phase."""
        logging.info("\n" + "="*60)
        logging.info("PHASE 2: SINGLE MODEL TRAINING")
        logging.info("="*60)

        start_time = time.time()

        # Create single model
        self.classifier.create_model('hybrid')

        # Setup training
        class_weights = np.array(self.results['data_preparation']['class_weights'])
        self.classifier.setup_training(class_weights)

        # Train model
        training_history = self.classifier.train()

        training_time = time.time() - start_time

        # Store results
        self.results['single_model'] = {
            'architecture': 'hybrid',
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'best_val_accuracy': self.classifier.best_val_acc,
            'training_history': training_history,
            'final_epoch': len(training_history['train_loss'])
        }

        # Save model
        model_path = Path(self.config['output']['model_dir']) / 'best_single_model.pth'
        self._save_model_with_metadata(model_path, 'single_hybrid')

        logging.info(f"✅ Single model training completed")
        logging.info(f"   Training time: {training_time/60:.2f} minutes")
        logging.info(f"   Best accuracy: {self.classifier.best_val_acc:.2f}%")

    def _run_ensemble_training(self) -> None:
        """Run ensemble model training phase."""
        logging.info("\n" + "="*60)
        logging.info("PHASE 3: ENSEMBLE MODEL TRAINING")
        logging.info("="*60)

        start_time = time.time()

        # Create ensemble model
        ensemble_classifier = ComprehensiveGenderClassifier(
            config=self.config,
            device=self.config.get('device')
        )

        # Prepare data for ensemble
        class_weights = ensemble_classifier.prepare_data()

        # Create ensemble model
        ensemble_classifier.create_model('ensemble')

        # Setup training
        ensemble_classifier.setup_training(class_weights)

        # Train ensemble
        ensemble_history = ensemble_classifier.train()

        training_time = time.time() - start_time

        # Store results
        self.results['ensemble_model'] = {
            'architecture': 'ensemble',
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'best_val_accuracy': ensemble_classifier.best_val_acc,
            'training_history': ensemble_history,
            'final_epoch': len(ensemble_history['train_loss'])
        }

        # Save ensemble model
        model_path = Path(self.config['output']['model_dir']) / 'best_ensemble_model.pth'
        self._save_model_with_metadata(model_path, 'ensemble', ensemble_classifier)

        logging.info(f"✅ Ensemble model training completed")
        logging.info(f"   Training time: {training_time/60:.2f} minutes")
        logging.info(f"   Best accuracy: {ensemble_classifier.best_val_acc:.2f}%")

        # Store ensemble classifier for later use
        self.ensemble_classifier = ensemble_classifier

    def _run_model_optimization(self) -> None:
        """Run model optimization phase including distillation and quantization."""
        logging.info("\n" + "="*60)
        logging.info("PHASE 4: MODEL OPTIMIZATION")
        logging.info("="*60)

        optimization_results = {}

        # Knowledge Distillation
        if hasattr(self, 'ensemble_classifier'):
            logging.info("🎓 Applying Knowledge Distillation...")
            distillation_results = self._apply_knowledge_distillation()
            optimization_results['knowledge_distillation'] = distillation_results

        # Model Quantization
        logging.info("🗜️ Applying Model Quantization...")
        quantization_results = self._apply_model_quantization()
        optimization_results['quantization'] = quantization_results

        # Model Pruning
        logging.info("✂️ Applying Model Pruning...")
        pruning_results = self._apply_model_pruning()
        optimization_results['pruning'] = pruning_results

        self.results['model_optimization'] = optimization_results

        logging.info("✅ Model optimization completed")

    def _apply_knowledge_distillation(self) -> Dict[str, Any]:
        """Apply knowledge distillation to create a smaller student model."""
        from torchvision.models import resnet34, ResNet34_Weights

        # Create student model (smaller ResNet)
        student_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        student_model.fc = nn.Linear(student_model.fc.in_features, 2)
        student_model.to(self.ensemble_classifier.device)

        # Setup student training
        student_optimizer = optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        # Distillation parameters
        temperature = 4.0
        alpha = 0.7
        num_epochs = 20

        best_student_acc = 0.0
        distillation_history = {'loss': [], 'accuracy': []}

        # Teacher model (ensemble) in eval mode
        self.ensemble_classifier.model.eval()

        for epoch in range(num_epochs):
            student_model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.ensemble_classifier.train_loader:
                images = images.to(self.ensemble_classifier.device, non_blocking=True)
                labels = labels.to(self.ensemble_classifier.device, non_blocking=True)

                student_optimizer.zero_grad()

                # Teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.ensemble_classifier.model(images)
                    if isinstance(teacher_outputs, tuple):
                        teacher_outputs = teacher_outputs[0]
                    teacher_probs = torch.softmax(teacher_outputs / temperature, dim=1)

                # Student predictions
                student_outputs = student_model(images)
                student_log_probs = torch.log_softmax(student_outputs / temperature, dim=1)

                # Distillation loss
                distill_loss = criterion_kl(student_log_probs, teacher_probs) * (temperature ** 2)

                # Student classification loss
                student_loss = criterion_ce(student_outputs, labels)

                # Combined loss
                total_loss = alpha * distill_loss + (1 - alpha) * student_loss

                total_loss.backward()
                student_optimizer.step()

                epoch_loss += total_loss.item()
                _, predicted = student_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Validation
            student_model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in self.ensemble_classifier.val_loader:
                    images = images.to(self.ensemble_classifier.device, non_blocking=True)
                    labels = labels.to(self.ensemble_classifier.device, non_blocking=True)

                    outputs = student_model(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            train_acc = 100. * correct / total
            val_acc = 100. * val_correct / val_total
            avg_loss = epoch_loss / len(self.ensemble_classifier.train_loader)

            distillation_history['loss'].append(avg_loss)
            distillation_history['accuracy'].append(val_acc)

            if val_acc > best_student_acc:
                best_student_acc = val_acc
                torch.save(student_model.state_dict(),
                          Path(self.config['output']['model_dir']) / 'distilled_student_model.pth')

            logging.info(f"Distillation Epoch {epoch+1}/{num_epochs}: "
                        f"Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        # Calculate compression metrics
        teacher_params = sum(p.numel() for p in self.ensemble_classifier.model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        compression_ratio = teacher_params / student_params

        return {
            'best_accuracy': best_student_acc,
            'teacher_parameters': teacher_params,
            'student_parameters': student_params,
            'compression_ratio': compression_ratio,
            'distillation_history': distillation_history
        }

    def _apply_model_quantization(self) -> Dict[str, Any]:
        """Apply dynamic quantization to the model."""
        if not hasattr(self, 'classifier') or self.classifier.model is None:
            return {'status': 'skipped', 'reason': 'no_model_available'}

        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.classifier.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )

        # Save quantized model
        torch.save(quantized_model.state_dict(),
                  Path(self.config['output']['model_dir']) / 'quantized_model.pth')

        # Calculate size reduction
        original_size = sum(p.numel() * p.element_size() for p in self.classifier.model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        size_reduction = original_size / quantized_size

        return {
            'original_size_mb': original_size / (1024**2),
            'quantized_size_mb': quantized_size / (1024**2),
            'size_reduction_ratio': size_reduction,
            'quantization_applied': True
        }

    def _apply_model_pruning(self) -> Dict[str, Any]:
        """Apply structured pruning to the model."""
        if not hasattr(self, 'classifier') or self.classifier.model is None:
            return {'status': 'skipped', 'reason': 'no_model_available'}

        # Simple magnitude-based pruning
        pruning_ratio = 0.3  # Remove 30% of smallest weights

        pruned_params = 0
        total_params = 0

        for module in self.classifier.model.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                total_params += weight.numel()

                # Calculate threshold for pruning
                weight_magnitude = torch.abs(weight)
                threshold = torch.quantile(weight_magnitude.flatten(), pruning_ratio)

                # Apply pruning mask
                pruning_mask = weight_magnitude > threshold
                pruned_params += (~pruning_mask).sum().item()

                # Zero out pruned weights
                weight.data = weight.data * pruning_mask.float()

        sparsity = pruned_params / total_params

        # Save pruned model
        torch.save(self.classifier.model.state_dict(),
                  Path(self.config['output']['model_dir']) / 'pruned_model.pth')

        return {
            'pruning_ratio': pruning_ratio,
            'actual_sparsity': sparsity,
            'pruned_parameters': pruned_params,
            'total_parameters': total_params
        }

    def _run_comprehensive_evaluation(self) -> None:
        """Run comprehensive evaluation and analysis."""
        logging.info("\n" + "="*60)
        logging.info("PHASE 5: COMPREHENSIVE EVALUATION")
        logging.info("="*60)

        evaluation_results = {}

        # Evaluate single model
        if 'single_model' in self.results:
            logging.info("📊 Evaluating single model...")
            single_eval = self._evaluate_model(self.classifier, 'single_model')
            evaluation_results['single_model'] = single_eval

        # Evaluate ensemble model
        if 'ensemble_model' in self.results and hasattr(self, 'ensemble_classifier'):
            logging.info("📊 Evaluating ensemble model...")
            ensemble_eval = self._evaluate_model(self.ensemble_classifier, 'ensemble_model')
            evaluation_results['ensemble_model'] = ensemble_eval

        # Performance comparison
        if len(evaluation_results) > 1:
            comparison = self._compare_models(evaluation_results)
            evaluation_results['model_comparison'] = comparison

        self.results['evaluation'] = evaluation_results

        logging.info("✅ Comprehensive evaluation completed")

    def _evaluate_model(self, classifier, model_name: str) -> Dict[str, Any]:
        """Evaluate a specific model comprehensively."""
        # Run validation
        val_loss, val_acc, val_preds, val_labels, val_probs = classifier.validate()

        # Calculate detailed metrics
        accuracy = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted')

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            val_labels, val_preds, average=None
        )

        # AUC-ROC
        probs_array = np.array(val_probs)
        auc_roc = roc_auc_score(val_labels, probs_array[:, 1])

        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)

        # Bias analysis
        bias_metrics = self._calculate_bias_metrics(val_labels, val_preds)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'bias_metrics': bias_metrics,
            'validation_loss': val_loss,
            'validation_accuracy': val_acc
        }

    def _calculate_bias_metrics(self, labels: List, preds: List) -> Dict[str, float]:
        """Calculate bias and fairness metrics."""
        female_indices = [i for i, label in enumerate(labels) if label == 0]
        male_indices = [i for i, label in enumerate(labels) if label == 1]

        if len(female_indices) == 0 or len(male_indices) == 0:
            return {'error': 'insufficient_data_for_bias_analysis'}

        # Accuracy per group
        female_correct = sum(1 for i in female_indices if preds[i] == labels[i])
        male_correct = sum(1 for i in male_indices if preds[i] == labels[i])

        female_accuracy = female_correct / len(female_indices)
        male_accuracy = male_correct / len(male_indices)

        # True positive rates
        female_tp_rate = sum(1 for i in female_indices if preds[i] == 0 and labels[i] == 0) / len(female_indices)
        male_tp_rate = sum(1 for i in male_indices if preds[i] == 1 and labels[i] == 1) / len(male_indices)

        return {
            'female_accuracy': female_accuracy,
            'male_accuracy': male_accuracy,
            'accuracy_difference': abs(female_accuracy - male_accuracy),
            'female_tpr': female_tp_rate,
            'male_tpr': male_tp_rate,
            'tpr_difference': abs(female_tp_rate - male_tp_rate),
            'demographic_parity_difference': abs(female_tp_rate - male_tp_rate),
            'fairness_score': 1.0 - abs(female_accuracy - male_accuracy)  # Higher is better
        }

    def _compare_models(self, evaluation_results: Dict) -> Dict[str, Any]:
        """Compare performance of different models."""
        models = list(evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

        comparison = {}
        best_model = None
        best_accuracy = 0.0

        for model_name in models:
            results = evaluation_results[model_name]
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_model = model_name

        comparison['best_model'] = best_model
        comparison['best_accuracy'] = best_accuracy

        # Performance differences
        performance_matrix = {}
        for metric in metrics:
            performance_matrix[metric] = {}
            for model_name in models:
                performance_matrix[metric][model_name] = evaluation_results[model_name][metric]

        comparison['performance_matrix'] = performance_matrix

        return comparison

    def _generate_comprehensive_reports(self) -> None:
        """Generate comprehensive reports and visualizations."""
        logging.info("\n" + "="*60)
        logging.info("PHASE 6: REPORT GENERATION")
        logging.info("="*60)

        # Generate training curves
        self._generate_training_curves()

        # Generate evaluation plots
        self._generate_evaluation_plots()

        # Generate markdown report
        self._generate_markdown_report()

        # Save complete results
        self._save_complete_results()

        logging.info("✅ Report generation completed")

    def _generate_training_curves(self) -> None:
        """Generate training curve visualizations."""
        plots_dir = Path(self.config['output']['plots_dir'])
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot for each trained model
        if 'single_model' in self.results:
            self._plot_training_history(
                self.results['single_model']['training_history'],
                'Single Model Training Curves',
                plots_dir / 'single_model_curves.png'
            )

        if 'ensemble_model' in self.results:
            self._plot_training_history(
                self.results['ensemble_model']['training_history'],
                'Ensemble Model Training Curves',
                plots_dir / 'ensemble_model_curves.png'
            )

    def _plot_training_history(self, history: Dict, title: str, save_path: Path) -> None:
        """Plot training history curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(history['train_loss']) + 1)

        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate
        axes[1, 0].plot(epochs, history['learning_rate'], 'g-')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

        # GPU memory (if available)
        if history.get('gpu_memory'):
            axes[1, 1].plot(epochs, history['gpu_memory'], 'm-')
            axes[1, 1].set_title('GPU Memory Usage')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Memory (GB)')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'GPU Memory\nNot Available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.suptitle(f'{title} - by {__author__}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_evaluation_plots(self) -> None:
        """Generate evaluation visualizations."""
        plots_dir = Path(self.config['output']['plots_dir'])

        if 'evaluation' in self.results:
            eval_results = self.results['evaluation']

            # Model comparison plot
            if 'model_comparison' in eval_results:
                self._plot_model_comparison(eval_results['model_comparison'], plots_dir)

            # Confusion matrices
            for model_name, results in eval_results.items():
                if model_name != 'model_comparison' and 'confusion_matrix' in results:
                    self._plot_confusion_matrix(
                        results['confusion_matrix'],
                        model_name,
                        plots_dir / f'{model_name}_confusion_matrix.png'
                    )

    def _plot_model_comparison(self, comparison: Dict, plots_dir: Path) -> None:
        """Plot model performance comparison."""
        performance_matrix = comparison['performance_matrix']
        metrics = list(performance_matrix.keys())
        models = list(performance_matrix[metrics[0]].keys())

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(metrics))
        width = 0.35

        for i, model_name in enumerate(models):
            values = [performance_matrix[metric][model_name] for metric in metrics]
            ax.bar(x + i * width, values, width, label=model_name.replace('_', ' ').title())

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(f'Model Performance Comparison - by {__author__}')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, cm: List[List[int]], model_name: str, save_path: Path) -> None:
        """Plot confusion matrix."""
        cm_array = np.array(cm)
        classes = ['Female', 'Male']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_markdown_report(self) -> None:
        """Generate comprehensive markdown report."""
        report_path = Path(self.config['output']['save_dir']) / 'comprehensive_report.md'

        with open(report_path, 'w') as f:
            f.write(f"# Comprehensive Gender Classification Report\n\n")
            f.write(f"**Author:** {__author__}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Version:** {__version__}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            total_time = self.results.get('execution_time', {}).get('total_minutes', 0)
            f.write(f"Complete gender classification system training completed in {total_time:.2f} minutes.\n\n")

            # System Configuration
            f.write("## System Configuration\n\n")
            f.write(f"- **Device:** {self.config.get('device', 'auto')}\n")
            f.write(f"- **PyTorch Version:** {torch.__version__}\n")
            f.write(f"- **Batch Size:** {self.config['data']['batch_size']}\n")
            f.write(f"- **Learning Rate:** {self.config['training']['learning_rate']}\n")
            f.write(f"- **Epochs:** {self.config['training']['num_epochs']}\n\n")

            # Results Summary
            if 'single_model' in self.results:
                f.write("### Single Model Results\n\n")
                single = self.results['single_model']
                f.write(f"- **Architecture:** CNN-Transformer Hybrid\n")
                f.write(f"- **Best Accuracy:** {single['best_val_accuracy']:.2f}%\n")
                f.write(f"- **Training Time:** {single['training_time_minutes']:.2f} minutes\n\n")

            if 'ensemble_model' in self.results:
                f.write("### Ensemble Model Results\n\n")
                ensemble = self.results['ensemble_model']
                f.write(f"- **Architecture:** Multi-Model Ensemble\n")
                f.write(f"- **Best Accuracy:** {ensemble['best_val_accuracy']:.2f}%\n")
                f.write(f"- **Training Time:** {ensemble['training_time_minutes']:.2f} minutes\n\n")

            # Mathematical Foundations
            f.write("## Mathematical Foundations\n\n")
            f.write("### CNN Feature Extraction\n")
            f.write("```\nf_cnn(x) = φ(W_conv * x + b_conv)\n```\n\n")
            f.write("### Transformer Attention\n")
            f.write("```\nAttention(Q,K,V) = softmax(QK^T/√d_k)V\n```\n\n")
            f.write("### Ensemble Fusion\n")
            f.write("```\nf_ensemble(x) = Σ(α_i * f_i(x)) where Σα_i = 1\n```\n\n")
            f.write("### Focal Loss\n")
            f.write("```\nFL(p_t) = -α_t(1-p_t)^γ log(p_t)\n```\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Production Deployment:** Use quantized models for faster inference\n")
            f.write("2. **Bias Monitoring:** Regularly assess fairness metrics\n")
            f.write("3. **Data Augmentation:** Continue advanced augmentation strategies\n")
            f.write("4. **Model Updates:** Implement continuous learning pipeline\n\n")

            f.write(f"---\n*Report generated by {__author__} using advanced deep learning techniques*\n")

        logging.info(f"Comprehensive report saved to: {report_path}")

    def _save_complete_results(self) -> None:
        """Save complete results to JSON file."""
        results_path = Path(self.config['output']['save_dir']) / 'complete_results.json'

        # Prepare results for JSON serialization
        serializable_results = self._make_json_serializable(self.results)

        # Add metadata
        complete_results = {
            'metadata': {
                'author': __author__,
                'version': __version__,
                'timestamp': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            },
            'configuration': self.config,
            'results': serializable_results
        }

        with open(results_path, 'w') as f:
            json.dump(complete_results, f, indent=2)

        logging.info(f"Complete results saved to: {results_path}")

    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _save_model_with_metadata(self, path: Path, model_type: str, classifier=None) -> None:
        """Save model with comprehensive metadata."""
        if classifier is None:
            classifier = self.classifier

        checkpoint = {
            'model_state_dict': classifier.model.state_dict(),
            'model_type': model_type,
            'author': __author__,
            'version': __version__,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'best_val_acc': classifier.best_val_acc,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }

        torch.save(checkpoint, path)
        logging.info(f"Model saved: {path}")


def create_config_from_args(args) -> Dict[str, Any]:
    """
    Create comprehensive configuration from command-line arguments.
    Author: Soumya Chakraborty

    Args:
        args: Parsed command-line arguments

    Returns:
        Complete configuration dictionary
    """
    return {
        'data': {
            'train_dir': args.train_dir,
            'val_dir': args.val_dir,
            'batch_size': args.batch_size,
            'num_workers': 4,
            'image_size': 224,
            'use_face_detection': not args.no_face_detection,
            'augmentation_prob': 0.8
        },
        'training': {
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'dropout_rate': args.dropout,
            'warmup_epochs': 5,
            'gradient_clip_value': args.grad_clip,
            'early_stopping_patience': 15
        },
        'training_phases': {
            'train_single': not args.skip_single,
            'train_ensemble': not args.skip_ensemble,
            'model_optimization': not args.skip_optimization
        },
        'model': {
            'architecture': args.architecture if not args.skip_ensemble else 'hybrid',
            'num_classes': 2,
            'cnn_backbone': args.cnn_backbone,
            'transformer_backbone': args.transformer_backbone
        },
        'loss': {
            'type': args.loss,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'label_smoothing': 0.1
        },
        'optimization': {
            'use_mixed_precision': not args.no_mixed_precision,
            'optimizer': args.optimizer,
            'scheduler': args.scheduler,
            'cosine_t0': 10,
            'cosine_t_mult': 2
        },
        'evaluation': {
            'save_predictions': True,
            'calculate_fairness_metrics': not args.no_fairness,
            'generate_plots': not args.no_plots,
            'save_confusion_matrix': True
        },
        'output': {
            'save_dir': args.output_dir,
            'model_dir': f"{args.output_dir}/models",
            'plots_dir': f"{args.output_dir}/plots",
            'logs_dir': f"{args.output_dir}/logs"
        },
        'device': args.device
    }


def main():
    """
    Main function with comprehensive command-line interface.
    Author: Soumya Chakraborty

    Supports the exact command-line usage patterns requested:
    # Train with custom parameters
    python train_complete_system.py --epochs 100 --batch-size 64 --lr 1e-3

    # Train only specific components
    python train_complete_system.py --skip-ensemble --output-dir custom_results

    # Use configuration file
    python train_complete_system.py --config config_template.json
    """
    parser = argparse.ArgumentParser(
        description=f"Comprehensive Gender Classification System by {__author__}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f"""
Examples:
  # Train with custom parameters
  python {sys.argv[0]} --epochs 100 --batch-size 64 --lr 1e-3

  # Train only specific components
  python {sys.argv[0]} --skip-ensemble --output-dir custom_results

  # Use configuration file
  python {sys.argv[0]} --config config_template.json

Mathematical Foundations:
  - CNN Feature Extraction: f_cnn(x) = φ(W_conv * x + b_conv)
  - Transformer Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
  - Ensemble Fusion: f_ensemble(x) = Σ(α_i * f_i(x)) where Σα_i = 1
  - Focal Loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)

Author: {__author__}
Version: {__version__}
        """
    )

    # Basic training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")

    # Data arguments
    parser.add_argument("--train-dir", type=str, default="Task_A/train", help="Training data directory")
    parser.add_argument("--val-dir", type=str, default="Task_A/val", help="Validation data directory")
    parser.add_argument("--no-face-detection", action="store_true", help="Disable face detection")

    # Model arguments
    parser.add_argument("--architecture", type=str, default="hybrid",
                       choices=["hybrid", "ensemble"], help="Model architecture")
    parser.add_argument("--cnn-backbone", type=str, default="efficientnet_b3",
                       help="CNN backbone architecture")
    parser.add_argument("--transformer-backbone", type=str, default="vit_base_patch16_224",
                       help="Transformer backbone architecture")

    # Training control - EXACT FUNCTIONALITY REQUESTED
    parser.add_argument("--skip-single", action="store_true", help="Skip single model training")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble training")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip model optimization")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")

    # Loss and optimization
    parser.add_argument("--loss", type=str, default="focal",
                       choices=["focal", "cross_entropy", "label_smoothing"], help="Loss function")
    parser.add_argument("--optimizer", type=str, default="adamw",
                       choices=["adamw", "adam"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine_warm_restarts",
                       choices=["cosine_warm_restarts", "cosine", "step"], help="Learning rate scheduler")

    # Advanced options
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value")

    # Evaluation options
    parser.add_argument("--no-plots", action="store_true", help="Don't generate training plots")
    parser.add_argument("--no-fairness", action="store_true", help="Skip fairness analysis")

    args = parser.parse_args()

    # Print header
    print("🚀 Comprehensive Gender Classification Training System")
    print("=" * 80)
    print(f"Author: {__author__}")
    print(f"Version: {__version__}")
    print(f"Email: {__email__}")
    print("=" * 80)
    print("\nMathematical Foundations:")
    print("- CNN Feature Extraction: f_cnn(x) = φ(W_conv * x + b_conv)")
    print("- Transformer Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V")
    print("- Ensemble Fusion: f_ensemble(x) = Σ(α_i * f_i(x)) where Σα_i = 1")
    print("- Focal Loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)")
    print("- Knowledge Distillation: L = αL_CE + (1-α)τ²L_KD")
    print("=" * 80)

    # Create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration loaded from: {args.config}")
    else:
        config = create_config_from_args(args)
        print("✅ Configuration created from command-line arguments")

    # Save configuration
    os.makedirs(args.output_dir, exist_ok=True)
    config_save_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Configuration saved to: {config_save_path}")

    print(f"\n📋 Training Configuration Summary:")
    print(f"   Epochs: {config['training']['num_epochs']}")
    print(f"   Batch Size: {config['data']['batch_size']}")
    print(f"   Learning Rate: {config['training']['learning_rate']}")
    print(f"   Architecture: {config['model']['architecture']}")
    print(f"   Output Directory: {config['output']['save_dir']}")
    print(f"   Skip Single: {not config['training_phases']['train_single']}")
    print(f"   Skip Ensemble: {not config['training_phases']['train_ensemble']}")

    try:
        # Initialize and run the advanced training system
        print(f"\n🚀 Initializing Advanced Training System...")
        system = AdvancedTrainingSystem(config)

        # Run complete pipeline
        print(f"\n🎯 Starting Complete Training Pipeline...")
        results = system.run_complete_pipeline()

        # Success summary
        total_time = results.get('execution_time', {}).get('total_minutes', 0)
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📊 Results Summary:")
        if 'single_model' in results:
            print(f"   Single Model Accuracy: {results['single_model']['best_val_accuracy']:.2f}%")
        if 'ensemble_model' in results:
            print(f"   Ensemble Model Accuracy: {results['ensemble_model']['best_val_accuracy']:.2f}%")
        print(f"   Total Training Time: {total_time:.2f} minutes")
        print(f"   Results Location: {config['output']['save_dir']}")
        print("=" * 80)
        print(f"✨ System designed and implemented by {__author__}")

        return results

    except Exception as e:
        print(f"\n❌ TRAINING FAILED!")
        print("=" * 80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return None


if __name__ == "__main__":
    # Ensure proper multiprocessing for cross-platform compatibility
    import multiprocessing
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Method already set

    # Run the main function
    result = main()

    # Exit with appropriate code
    if result is None:
        sys.exit(1)
    else:
        sys.exit(0)
