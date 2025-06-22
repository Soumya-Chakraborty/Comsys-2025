import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.cuda.amp import GradScaler, autocast
import timm
import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class GenderDataset(Dataset):
    """Custom dataset for gender classification with advanced augmentation"""

    def __init__(self, root_dir, transform=None, balance_classes=False):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_weights = None

        # Load samples
        for class_name in ['female', 'male']:
            class_dir = os.path.join(root_dir, class_name)
            label = 0 if class_name == 'female' else 1

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, label))

        # Calculate class weights for balancing
        labels = [sample[1] for sample in self.samples]
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        self.class_weights = total_samples / (len(class_counts) * class_counts)

        print(f"Dataset loaded: {len(self.samples)} samples")
        print(f"Class distribution: Female={class_counts[0]}, Male={class_counts[1]}")
        print(f"Class weights: Female={self.class_weights[0]:.4f}, Male={self.class_weights[1]:.4f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Load image with error handling
            image = Image.open(img_path).convert('RGB')

            # Face detection and cropping (optional enhancement)
            image_np = np.array(image)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # If face detected, crop to face region
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Use the first detected face
                # Add some padding around the face
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image_np.shape[1] - x, w + 2*padding)
                h = min(image_np.shape[0] - y, h + 2*padding)
                image = image.crop((x, y, x+w, y+h))

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a dummy image if loading fails
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label

class CNNTransformerHybrid(nn.Module):
    """Hybrid CNN-Transformer architecture for gender classification"""

    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(CNNTransformerHybrid, self).__init__()

        # CNN backbone (EfficientNet)
        self.cnn_backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        cnn_features = self.cnn_backbone.classifier.in_features
        self.cnn_backbone.classifier = nn.Identity()  # Remove original classifier

        # Vision Transformer
        self.vit_backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        vit_features = self.vit_backbone.num_features

        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(cnn_features + vit_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        # Attention mechanism for feature weighting
        self.attention = nn.Sequential(
            nn.Linear(cnn_features + vit_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # CNN features
        cnn_features = self.cnn_backbone(x)

        # ViT features
        vit_features = self.vit_backbone(x)

        # Concatenate features
        combined_features = torch.cat([cnn_features, vit_features], dim=1)

        # Attention weights
        attention_weights = self.attention(combined_features)

        # Apply attention to individual feature sets
        weighted_cnn = cnn_features * attention_weights[:, 0:1]
        weighted_vit = vit_features * attention_weights[:, 1:2]

        # Combine weighted features
        final_features = torch.cat([weighted_cnn, weighted_vit], dim=1)

        # Feature fusion
        fused_features = self.feature_fusion(final_features)

        # Classification
        output = self.classifier(fused_features)

        return output

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class GenderClassifier:
    """Main class for gender classification system with GPU optimizations"""

    def __init__(self, device=None):
        self.device = self._setup_optimal_device(device)
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # GPU optimization features
        self.use_amp = self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        self._setup_gpu_optimizations()

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(self.device)}")
            print(f"Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f} GB")
            print(f"Mixed Precision: {self.use_amp}")

    def _setup_optimal_device(self, device):
        """Setup optimal device configuration"""
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            # Select GPU with most memory
            device_count = torch.cuda.device_count()
            if device_count > 1:
                max_memory = 0
                best_gpu = 0
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    if props.total_memory > max_memory:
                        max_memory = props.total_memory
                        best_gpu = i
                device = torch.device(f'cuda:{best_gpu}')
            else:
                device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        return device

    def _setup_gpu_optimizations(self):
        """Setup GPU-specific optimizations"""
        if self.device.type == 'cuda':
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Clear GPU cache
            torch.cuda.empty_cache()

            # Set memory fraction for low-memory GPUs
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            if total_memory < 8 * 1024**3:  # Less than 8GB
                torch.cuda.set_per_process_memory_fraction(0.8)

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
            'utilization': (allocated / total) * 100
        }

    def clear_gpu_cache(self):
        """Clear GPU cache and run garbage collection"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    def prepare_data(self, train_dir, val_dir, batch_size=32, num_workers=None):
        """Prepare data loaders with advanced augmentation and GPU optimizations"""

        # Auto-determine optimal number of workers
        if num_workers is None:
            num_workers = min(8, os.cpu_count()) if self.device.type == 'cpu' else min(12, os.cpu_count())

        # Advanced data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])

        # Simple preprocessing for validation
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        train_dataset = GenderDataset(train_dir, transform=train_transform)
        val_dataset = GenderDataset(val_dir, transform=val_transform)

        # Create weighted sampler for balanced training
        sample_weights = [train_dataset.class_weights[label] for _, label in train_dataset.samples]
        sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

        # GPU-optimized DataLoader settings
        loader_kwargs = {
            'num_workers': num_workers,
            'pin_memory': self.device.type == 'cuda',
            'persistent_workers': num_workers > 0,
            'prefetch_factor': 2 if num_workers > 0 else 2,
        }

        if self.device.type == 'cuda':
            loader_kwargs.update({
                'pin_memory_device': str(self.device),
            })

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            **loader_kwargs
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs
        )

        print(f"DataLoader config: batch_size={batch_size}, num_workers={num_workers}")
        return train_dataset.class_weights

    def create_model(self, num_classes=2, dropout_rate=0.3):
        """Create the hybrid CNN-Transformer model"""
        self.model = CNNTransformerHybrid(num_classes=num_classes, dropout_rate=dropout_rate)
        self.model.to(self.device)

        # Print model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return self.model

    def setup_training(self, class_weights, learning_rate=1e-4, weight_decay=1e-4):
        """Setup optimizer, loss function, and scheduler"""

        # Use focal loss with class weights
        self.criterion = FocalLoss(alpha=1, gamma=2)

        # Adam optimizer with different learning rates for different parts
        backbone_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'feature_fusion' in name or 'attention' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained parts
            {'params': classifier_params, 'lr': learning_rate}       # Higher LR for new parts
        ], weight_decay=weight_decay)

        # Cosine annealing scheduler with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

    def train_epoch(self):
        """Train for one epoch with GPU optimizations"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Move data to device with non-blocking transfer for GPU
            if self.device.type == 'cuda':
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
            else:
                images, labels = images.to(self.device), labels.to(self.device)

            # Zero gradients (more efficient)
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

            # Clear cache periodically for GPU
            if self.device.type == 'cuda' and batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate the model with GPU optimizations"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                # Move data to device with non-blocking transfer for GPU
                if self.device.type == 'cuda':
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                else:
                    images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass with mixed precision
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

                # Store predictions and probabilities for detailed analysis
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

    def train(self, num_epochs=50, save_best=True, save_path='best_gender_model.pth'):
        """Complete training loop with GPU optimizations"""

        print(f"Starting GPU-optimized training for {num_epochs} epochs...")

        # Print initial GPU memory info
        if self.device.type == 'cuda':
            memory_info = self.get_gpu_memory_info()
            if memory_info:
                print(f"Initial GPU Memory: {memory_info['allocated_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)

            # Training
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validation
            val_loss, val_acc, val_preds, val_labels, val_probs = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.2e}")

            # Print GPU memory usage
            if self.device.type == 'cuda':
                memory_info = self.get_gpu_memory_info()
                if memory_info:
                    print(f"GPU Memory: {memory_info['allocated_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB ({memory_info['utilization']:.1f}%)")

            # Save best model
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accs': self.train_accs,
                    'val_accs': self.val_accs
                }

                if self.use_amp:
                    checkpoint['scaler_state_dict'] = self.scaler.state_dict()

                torch.save(checkpoint, save_path)
                print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

            # Detailed metrics every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.evaluate_detailed_metrics(val_preds, val_labels, val_probs)

            # Periodic GPU cache clearing
            if self.device.type == 'cuda' and (epoch + 1) % 5 == 0:
                self.clear_gpu_cache()

    def evaluate_detailed_metrics(self, preds, labels, probs):
        """Calculate and display detailed evaluation metrics"""

        # Basic metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, preds, average=None
        )

        # AUC-ROC
        probs_array = np.array(probs)
        auc_roc = roc_auc_score(labels, probs_array[:, 1])

        print("\n" + "="*60)
        print("DETAILED EVALUATION METRICS")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

        print("\nPer-Class Metrics:")
        classes = ['Female', 'Male']
        for i, class_name in enumerate(classes):
            print(f"{class_name:8} - Precision: {precision_per_class[i]:.4f}, "
                  f"Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(labels, preds)
        print(f"\nConfusion Matrix:")
        print(f"          Female  Male")
        print(f"Female    {cm[0,0]:6d}  {cm[0,1]:4d}")
        print(f"Male      {cm[1,0]:6d}  {cm[1,1]:4d}")

        # Bias analysis
        self.analyze_bias(labels, preds)

    def analyze_bias(self, labels, preds):
        """Analyze potential bias in predictions"""

        # Calculate demographic parity difference
        female_pred_rate = np.mean([p for l, p in zip(labels, preds) if l == 0])
        male_pred_rate = np.mean([p for l, p in zip(labels, preds) if l == 1])

        # Calculate equalized odds
        female_tpr = np.mean([p for l, p in zip(labels, preds) if l == 0 and p == 0])
        male_tpr = np.mean([p for l, p in zip(labels, preds) if l == 1 and p == 1])

        print("\nBias Analysis:")
        print(f"Female Positive Prediction Rate: {female_pred_rate:.4f}")
        print(f"Male Positive Prediction Rate: {male_pred_rate:.4f}")
        print(f"Demographic Parity Difference: {abs(female_pred_rate - male_pred_rate):.4f}")
        print(f"True Positive Rate Difference: {abs(female_tpr - male_tpr):.4f}")

    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot training and validation curves"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(self.train_accs, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accs, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def load_best_model(self, model_path='best_gender_model.pth'):
        """Load the best saved model with GPU support"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded model with best validation accuracy: {self.best_val_acc:.2f}%")

    def predict_single_image(self, image_path):
        """Predict gender for a single image"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            gender = 'Female' if predicted_class == 0 else 'Male'
            return gender, confidence, probabilities[0].cpu().numpy()

        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return None, 0.0, None

def main():
    """Main function to run the gender classification system"""

    # Initialize classifier
    classifier = GenderClassifier()

    # Prepare data
    train_dir = 'Task_A/train'
    val_dir = 'Task_A/val'

    print("Preparing data...")
    class_weights = classifier.prepare_data(train_dir, val_dir, batch_size=32)

    # Create model
    print("Creating model...")
    classifier.create_model(num_classes=2, dropout_rate=0.3)

    # Setup training
    print("Setting up training...")
    classifier.setup_training(class_weights, learning_rate=1e-4)

    # Train model
    print("Starting training...")
    classifier.train(num_epochs=50, save_best=True, save_path='best_gender_model.pth')

    # Plot training curves
    classifier.plot_training_curves('training_curves.png')

    # Final evaluation on validation set
    print("\nFinal evaluation on validation set:")
    val_loss, val_acc, val_preds, val_labels, val_probs = classifier.validate()
    classifier.evaluate_detailed_metrics(val_preds, val_labels, val_probs)

    print(f"\nTraining completed! Best validation accuracy: {classifier.best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
