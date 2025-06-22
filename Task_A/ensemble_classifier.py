import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, resnet50, EfficientNet_B3_Weights, ResNet50_Weights
import timm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

class EnsembleModel(nn.Module):
    """Ensemble of multiple architectures for robust gender classification"""

    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(EnsembleModel, self).__init__()

        # Model 1: EfficientNet-B3
        self.efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        efficientnet_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Identity()

        # Model 2: ResNet-50
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Model 3: Vision Transformer
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        vit_features = self.vit.num_features

        # Model 4: ConvNeXt
        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        convnext_features = self.convnext.num_features

        # Individual classifiers for each model
        self.efficientnet_classifier = self._create_classifier(efficientnet_features, num_classes, dropout_rate)
        self.resnet_classifier = self._create_classifier(resnet_features, num_classes, dropout_rate)
        self.vit_classifier = self._create_classifier(vit_features, num_classes, dropout_rate)
        self.convnext_classifier = self._create_classifier(convnext_features, num_classes, dropout_rate)

        # Meta classifier for ensemble fusion
        total_features = efficientnet_features + resnet_features + vit_features + convnext_features
        self.meta_classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Attention weights for adaptive ensemble
        self.attention_weights = nn.Parameter(torch.ones(4) / 4)

    def _create_classifier(self, input_features, num_classes, dropout_rate):
        """Create a classifier head"""
        return nn.Sequential(
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, mode='ensemble'):
        """Forward pass with different modes"""

        # Extract features from all models
        efficientnet_feat = self.efficientnet(x)
        resnet_feat = self.resnet(x)
        vit_feat = self.vit(x)
        convnext_feat = self.convnext(x)

        if mode == 'individual':
            # Return individual predictions
            eff_pred = self.efficientnet_classifier(efficientnet_feat)
            res_pred = self.resnet_classifier(resnet_feat)
            vit_pred = self.vit_classifier(vit_feat)
            conv_pred = self.convnext_classifier(convnext_feat)
            return [eff_pred, res_pred, vit_pred, conv_pred]

        elif mode == 'weighted_average':
            # Weighted average ensemble
            eff_pred = self.efficientnet_classifier(efficientnet_feat)
            res_pred = self.resnet_classifier(resnet_feat)
            vit_pred = self.vit_classifier(vit_feat)
            conv_pred = self.convnext_classifier(convnext_feat)

            # Apply softmax to attention weights
            weights = torch.softmax(self.attention_weights, dim=0)

            ensemble_pred = (weights[0] * eff_pred +
                           weights[1] * res_pred +
                           weights[2] * vit_pred +
                           weights[3] * conv_pred)
            return ensemble_pred

        elif mode == 'meta_learning':
            # Meta-learning approach
            combined_features = torch.cat([efficientnet_feat, resnet_feat, vit_feat, convnext_feat], dim=1)
            meta_pred = self.meta_classifier(combined_features)
            return meta_pred

        else:  # default ensemble mode
            # Combine all approaches
            individual_preds = self.forward(x, mode='individual')
            weighted_pred = self.forward(x, mode='weighted_average')
            meta_pred = self.forward(x, mode='meta_learning')

            # Final ensemble
            final_pred = (weighted_pred + meta_pred) / 2
            return final_pred, individual_preds, weighted_pred, meta_pred

class AdvancedEnsembleClassifier:
    """Advanced ensemble classifier with optimization techniques"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.models_dict = {}
        self.best_val_acc = 0.0
        print(f"Using device: {self.device}")

    def create_ensemble_model(self, num_classes=2, dropout_rate=0.3):
        """Create the ensemble model"""
        self.model = EnsembleModel(num_classes=num_classes, dropout_rate=dropout_rate)
        self.model.to(self.device)

        # Print model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Ensemble Model - Total parameters: {total_params:,}")
        print(f"Ensemble Model - Trainable parameters: {trainable_params:,}")

        return self.model

    def train_individual_models(self, train_loader, val_loader, num_epochs=30):
        """Train individual models first for better ensemble initialization"""

        print("Training individual models...")
        individual_models = {
            'efficientnet': self.model.efficientnet,
            'resnet': self.model.resnet,
            'vit': self.model.vit,
            'convnext': self.model.convnext
        }

        individual_classifiers = {
            'efficientnet': self.model.efficientnet_classifier,
            'resnet': self.model.resnet_classifier,
            'vit': self.model.vit_classifier,
            'convnext': self.model.convnext_classifier
        }

        for model_name in individual_models.keys():
            print(f"\nTraining {model_name}...")

            # Create optimizer for individual model
            params = list(individual_models[model_name].parameters()) + \
                    list(individual_classifiers[model_name].parameters())
            optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            best_acc = 0.0

            for epoch in range(num_epochs):
                # Training
                individual_models[model_name].train()
                individual_classifiers[model_name].train()

                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    images, labels = images.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()

                    # Forward pass
                    features = individual_models[model_name](images)
                    outputs = individual_classifiers[model_name](features)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                # Validation
                individual_models[model_name].eval()
                individual_classifiers[model_name].eval()

                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)

                        features = individual_models[model_name](images)
                        outputs = individual_classifiers[model_name](features)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()

                train_acc = 100. * train_correct / train_total
                val_acc = 100. * val_correct / val_total

                if val_acc > best_acc:
                    best_acc = val_acc

                print(f"{model_name} - Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

            print(f"{model_name} best validation accuracy: {best_acc:.2f}%")

    def train_ensemble(self, train_loader, val_loader, num_epochs=50, ensemble_mode='ensemble'):
        """Train the ensemble model"""

        print(f"\nTraining ensemble model in {ensemble_mode} mode...")

        # Setup optimizer with different learning rates
        backbone_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if any(x in name for x in ['classifier', 'meta_classifier', 'attention_weights']):
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},    # Very low LR for pretrained parts
            {'params': classifier_params, 'lr': 1e-4}   # Higher LR for ensemble parts
        ], weight_decay=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in tqdm(train_loader, desc="Training Ensemble"):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                if ensemble_mode == 'ensemble':
                    outputs, _, _, _ = self.model(images, mode='ensemble')
                else:
                    outputs = self.model(images, mode=ensemble_mode)

                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            scheduler.step()

            # Validation
            val_acc = self.validate_ensemble(val_loader, ensemble_mode)
            train_acc = 100. * train_correct / train_total

            print(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'ensemble_mode': ensemble_mode
                }, f'best_ensemble_model_{ensemble_mode}.pth')
                print(f"New best ensemble model saved: {val_acc:.2f}%")

        self.best_val_acc = best_val_acc
        return best_val_acc

    def validate_ensemble(self, val_loader, ensemble_mode='ensemble'):
        """Validate ensemble model"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if ensemble_mode == 'ensemble':
                    outputs, _, _, _ = self.model(images, mode='ensemble')
                else:
                    outputs = self.model(images, mode=ensemble_mode)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100. * correct / total
        return accuracy

    def evaluate_all_ensemble_methods(self, val_loader):
        """Evaluate all ensemble methods and compare performance"""

        print("\n" + "="*80)
        print("COMPREHENSIVE ENSEMBLE EVALUATION")
        print("="*80)

        methods = ['weighted_average', 'meta_learning', 'ensemble']
        results = {}

        self.model.eval()

        for method in methods:
            print(f"\nEvaluating {method}...")

            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Evaluating {method}"):
                    images, labels = images.to(self.device), labels.to(self.device)

                    if method == 'ensemble':
                        outputs, individual_preds, weighted_pred, meta_pred = self.model(images, mode='ensemble')
                    else:
                        outputs = self.model(images, mode=method)

                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probabilities.cpu().numpy())

            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
            auc_roc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])

            results[method] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc_roc': auc_roc
            }

            print(f"{method.replace('_', ' ').title()} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC-ROC: {auc_roc:.4f}")

        # Find best method
        best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\n🏆 Best performing method: {best_method.replace('_', ' ').title()}")
        print(f"Best accuracy: {results[best_method]['accuracy']:.4f}")

        return results, best_method

    def model_compression_and_optimization(self):
        """Apply model compression techniques"""

        print("\nApplying model compression and optimization...")

        # 1. Quantization (Post-training quantization)
        self.model.eval()

        # Prepare model for quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8
        )

        # Calculate model sizes
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024**2)

        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {original_size/quantized_size:.2f}x")

        # Save quantized model
        torch.save(quantized_model.state_dict(), 'quantized_ensemble_model.pth')

        return quantized_model

    def knowledge_distillation(self, train_loader, val_loader, student_model=None, temperature=4.0, alpha=0.7):
        """Apply knowledge distillation to create a smaller, faster model"""

        print("\nApplying knowledge distillation...")

        if student_model is None:
            # Create a smaller student model
            from torchvision.models import resnet34, ResNet34_Weights
            student_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            student_model.fc = nn.Linear(student_model.fc.in_features, 2)

        student_model.to(self.device)

        # Setup training for student
        optimizer = optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        # Teacher model (ensemble) in eval mode
        self.model.eval()

        best_student_acc = 0.0

        for epoch in range(30):  # Fewer epochs for distillation
            print(f"\nDistillation Epoch {epoch+1}/30")

            student_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in tqdm(train_loader, desc="Knowledge Distillation"):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_outputs, _, _, _ = self.model(images, mode='ensemble')
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
                optimizer.step()

                train_loss += total_loss.item()
                _, predicted = student_outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            # Validate student
            student_model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = student_model(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total

            print(f"Student - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

            if val_acc > best_student_acc:
                best_student_acc = val_acc
                torch.save(student_model.state_dict(), 'distilled_student_model.pth')

        print(f"Best student model accuracy: {best_student_acc:.2f}%")

        # Compare model sizes
        teacher_size = sum(p.numel() for p in self.model.parameters())
        student_size = sum(p.numel() for p in student_model.parameters())

        print(f"Teacher model parameters: {teacher_size:,}")
        print(f"Student model parameters: {student_size:,}")
        print(f"Parameter reduction: {teacher_size/student_size:.2f}x")

        return student_model, best_student_acc

    def create_deployment_model(self, model_path='best_ensemble_model_ensemble.pth'):
        """Create optimized model for deployment"""

        print("\nCreating deployment-ready model...")

        # Load best ensemble model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Apply optimizations
        quantized_model = self.model_compression_and_optimization()

        # Create deployment wrapper
        class DeploymentModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            def forward(self, x):
                if not isinstance(x, torch.Tensor):
                    x = self.preprocess(x).unsqueeze(0)

                self.model.eval()
                with torch.no_grad():
                    outputs, _, _, _ = self.model(x, mode='ensemble')
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(outputs, dim=1)
                    confidence = probabilities.max(dim=1)[0]

                return predicted_class, confidence, probabilities

        deployment_model = DeploymentModel(quantized_model)
        torch.save(deployment_model, 'deployment_gender_classifier.pth')

        print("Deployment model saved as 'deployment_gender_classifier.pth'")
        return deployment_model

def main():
    """Main function for ensemble training and evaluation"""

    # This would be called from the main training script
    print("Advanced Ensemble Gender Classifier")
    print("This module provides ensemble methods and optimization techniques")
    print("Import this module and use the AdvancedEnsembleClassifier class")

if __name__ == "__main__":
    main()
