import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import timm
import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import math
import random
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArcFaceLoss(nn.Module):
    """ArcFace loss implementation for face recognition"""
    def __init__(self, embedding_dim=512, num_classes=1000, margin=0.5, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Initialize weights
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)

        # Get one-hot labels
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Add margin to target logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)

        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

class FaceViT(nn.Module):
    """Vision Transformer for Face Recognition with ArcFace"""
    def __init__(self, model_name='vit_base_patch16_224', num_classes=1000, embedding_dim=512, pretrained=True):
        super(FaceViT, self).__init__()

        # Load pre-trained ViT
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        backbone_dim = self.backbone.num_features

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # ArcFace loss
        self.arcface = ArcFaceLoss(embedding_dim, num_classes)

    def forward(self, x, labels=None):
        # Extract features
        features = self.backbone(x)
        embeddings = self.embedding(features)

        if self.training and labels is not None:
            # Training mode with ArcFace loss
            logits = self.arcface(embeddings, labels)
            return logits, embeddings
        else:
            # Inference mode
            return embeddings

class RobustAugmentation:
    """Advanced augmentation pipeline for face recognition"""
    def __init__(self, image_size=224, is_training=True):
        if is_training:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            return self.transform(image=image)['image']
        else:
            image_np = np.array(image)
            return self.transform(image=image_np)['image']

class FaceDataset(Dataset):
    """Face Recognition Dataset with distortion handling"""
    def __init__(self, root_dir, transform=None, include_distorted=True, max_samples_per_class=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.include_distorted = include_distorted

        self.samples = []
        self.labels = []
        self.class_names = []

        self._load_dataset(max_samples_per_class)

        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)

        logger.info(f"Dataset loaded: {len(self.samples)} samples, {self.num_classes} classes")

    def _load_dataset(self, max_samples_per_class):
        """Load dataset from directory structure"""
        person_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]

        for person_dir in tqdm(person_dirs, desc="Loading dataset"):
            person_name = person_dir.name
            sample_count = 0

            # Load original images
            for img_file in person_dir.glob("*.jpg"):
                if img_file.name != "Thumbs.db":
                    self.samples.append(str(img_file))
                    self.labels.append(person_name)
                    sample_count += 1

                    if max_samples_per_class and sample_count >= max_samples_per_class:
                        break

            # Load distorted images if enabled
            if self.include_distorted:
                distortion_dir = person_dir / "distortion"
                if distortion_dir.exists():
                    for img_file in distortion_dir.glob("*.jpg"):
                        if max_samples_per_class and sample_count >= max_samples_per_class:
                            break
                        self.samples.append(str(img_file))
                        self.labels.append(person_name)
                        sample_count += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.encoded_labels[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

class FaceRecognitionSystem:
    """Complete Face Recognition System"""
    def __init__(self, model_name='vit_base_patch16_224', embedding_dim=512, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # Initialize components
        self.model = None
        self.train_transform = RobustAugmentation(is_training=True)
        self.val_transform = RobustAugmentation(is_training=False)

        # Training state
        self.best_accuracy = 0.0
        self.training_history = {'loss': [], 'accuracy': []}

        logger.info(f"Initialized Face Recognition System on {self.device}")

    def create_model(self, num_classes):
        """Create the face recognition model"""
        self.model = FaceViT(
            model_name=self.model_name,
            num_classes=num_classes,
            embedding_dim=self.embedding_dim,
            pretrained=True
        ).to(self.device)

        return self.model

    def train_model(self, train_dir, val_split=0.2, batch_size=32, epochs=50, learning_rate=1e-4):
        """Train the face recognition model"""

        # Load dataset
        full_dataset = FaceDataset(train_dir, transform=self.train_transform, include_distorted=True)

        # Create validation split
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        # Update validation dataset transform
        val_dataset.dataset.transform = self.val_transform

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Create model
        self.create_model(full_dataset.num_classes)

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)

            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)

            # Update scheduler
            scheduler.step()

            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model('best_model.pth')

            # Update history
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(val_acc)

            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save label encoder
        self.label_encoder = full_dataset.label_encoder
        self.save_label_encoder('label_encoder.json')

        return self.training_history

    def _train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Training")
        for images, labels, _ in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            logits, embeddings = self.model(images, labels)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

        return total_loss / len(dataloader), correct / total

    def _validate_epoch(self, dataloader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)

                logits, embeddings = self.model(images, labels)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(dataloader), correct / total

    def extract_embeddings(self, image_path):
        """Extract embeddings from a single image"""
        self.model.eval()

        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.val_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embeddings = self.model(image)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def verify_faces(self, image1_path, image2_path, threshold=0.6):
        """Verify if two face images belong to the same person"""
        emb1 = self.extract_embeddings(image1_path)
        emb2 = self.extract_embeddings(image2_path)

        # Compute cosine similarity
        similarity = np.dot(emb1, emb2.T)[0][0]

        return similarity > threshold, similarity

    def identify_face(self, image_path, gallery_dir, top_k=5):
        """Identify a face against a gallery of known faces"""
        query_embedding = self.extract_embeddings(image_path)

        # Build gallery embeddings
        gallery_embeddings = []
        gallery_labels = []

        for person_dir in Path(gallery_dir).iterdir():
            if person_dir.is_dir():
                for img_file in person_dir.glob("*.jpg"):
                    if img_file.name != "Thumbs.db":
                        try:
                            embedding = self.extract_embeddings(str(img_file))
                            gallery_embeddings.append(embedding[0])
                            gallery_labels.append(person_dir.name)
                        except:
                            continue

        if not gallery_embeddings:
            return []

        gallery_embeddings = np.array(gallery_embeddings)

        # Compute similarities
        similarities = np.dot(query_embedding, gallery_embeddings.T)[0]

        # Get top-k matches
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'person': gallery_labels[idx],
                'similarity': similarities[idx],
                'confidence': similarities[idx] * 100
            })

        return results

    def evaluate_on_test_pairs(self, test_pairs_file):
        """Evaluate the model on test pairs for verification"""
        # This would be implemented based on the specific test format
        pass

    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'best_accuracy': self.best_accuracy
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath, num_classes):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model_name = checkpoint.get('model_name', self.model_name)
        self.embedding_dim = checkpoint.get('embedding_dim', self.embedding_dim)

        self.create_model(num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)

        logger.info(f"Model loaded from {filepath}")

    def save_label_encoder(self, filepath):
        """Save label encoder"""
        label_mapping = {
            'classes': self.label_encoder.classes_.tolist(),
            'num_classes': len(self.label_encoder.classes_)
        }
        with open(filepath, 'w') as f:
            json.dump(label_mapping, f)

    def load_label_encoder(self, filepath):
        """Load label encoder"""
        with open(filepath, 'r') as f:
            label_mapping = json.load(f)

        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(label_mapping['classes'])

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.training_history['loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(self.training_history['accuracy'])
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def create_test_pairs(data_dir, num_pairs=1000):
    """Create test pairs for evaluation"""
    data_path = Path(data_dir)
    pairs = []

    # Get all person directories
    person_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    # Create positive pairs (same person)
    for _ in range(num_pairs // 2):
        person_dir = random.choice(person_dirs)
        images = list(person_dir.glob("*.jpg"))

        # Include distorted images
        distortion_dir = person_dir / "distortion"
        if distortion_dir.exists():
            images.extend(list(distortion_dir.glob("*.jpg")))

        if len(images) >= 2:
            img1, img2 = random.sample(images, 2)
            pairs.append((str(img1), str(img2), 1))  # 1 for same person

    # Create negative pairs (different persons)
    for _ in range(num_pairs // 2):
        person1, person2 = random.sample(person_dirs, 2)

        images1 = list(person1.glob("*.jpg"))
        images2 = list(person2.glob("*.jpg"))

        # Include distorted images
        dist1 = person1 / "distortion"
        if dist1.exists():
            images1.extend(list(dist1.glob("*.jpg")))

        dist2 = person2 / "distortion"
        if dist2.exists():
            images2.extend(list(dist2.glob("*.jpg")))

        if images1 and images2:
            img1 = random.choice(images1)
            img2 = random.choice(images2)
            pairs.append((str(img1), str(img2), 0))  # 0 for different persons

    return pairs

def main():
    """Main function to run the face recognition system"""
    # Initialize system
    system = FaceRecognitionSystem(
        model_name='vit_base_patch16_224',
        embedding_dim=512,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Set paths
    train_dir = "train"

    # Train the model
    print("Starting training...")
    history = system.train_model(
        train_dir=train_dir,
        val_split=0.2,
        batch_size=32,
        epochs=50,
        learning_rate=1e-4
    )

    # Plot training history
    system.plot_training_history()

    # Create test pairs for evaluation
    print("Creating test pairs...")
    test_pairs = create_test_pairs(train_dir, num_pairs=1000)

    # Evaluate on test pairs
    print("Evaluating on test pairs...")
    correct = 0
    total = len(test_pairs)

    for img1, img2, label in tqdm(test_pairs, desc="Testing"):
        try:
            is_same, similarity = system.verify_faces(img1, img2, threshold=0.6)
            if (is_same and label == 1) or (not is_same and label == 0):
                correct += 1
        except:
            total -= 1
            continue

    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f}")

    # Example identification
    print("\nExample face identification:")
    if test_pairs:
        example_image = test_pairs[0][0]
        results = system.identify_face(example_image, train_dir, top_k=5)

        for i, result in enumerate(results):
            print(f"Rank {i+1}: {result['person']} (Confidence: {result['confidence']:.2f}%)")

if __name__ == "__main__":
    main()
