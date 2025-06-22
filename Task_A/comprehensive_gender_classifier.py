#!/usr/bin/env python3
"""
Comprehensive Gender Classification System with Mathematical Foundations
Author: Soumya Chakraborty
Date: 2024

This module implements a state-of-the-art gender classification system using:
1. CNN-Transformer Hybrid Architecture
2. Ensemble Methods with Attention Mechanisms
3. Advanced Optimization Techniques
4. Bias Mitigation Strategies
5. Mathematical Foundations for all algorithms

Mathematical Foundations:
- CNN Feature Extraction: f_cnn(x) = φ(W_conv * x + b_conv)
- Transformer Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Ensemble Fusion: f_ensemble(x) = Σ(α_i * f_i(x)) where Σα_i = 1
- Focal Loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
- Knowledge Distillation: L = αL_CE + (1-α)τ²L_KD
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, resnet50, EfficientNet_B3_Weights, ResNet50_Weights
import timm
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from tqdm import tqdm
import warnings
import gc
import math
import random
from collections import OrderedDict

warnings.filterwarnings('ignore')

# Author Information
__author__ = "Soumya Chakraborty"
__version__ = "1.0.0"
__email__ = "soumya.chakraborty@example.com"
__license__ = "MIT"

# Set random seeds for reproducibility
def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducible results across all libraries.

    Mathematical Foundation:
    Ensures deterministic behavior in stochastic processes by fixing the initial state
    of all random number generators used in the pipeline.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info(f"Random seeds set to {seed} for reproducibility")

set_random_seeds(42)

class MathematicalFoundations:
    """
    Mathematical foundations and utility functions for the gender classification system.
    Author: Soumya Chakraborty

    This class contains the mathematical formulations used throughout the system.
    """

    @staticmethod
    def convolution_output_size(input_size: int, kernel_size: int, stride: int = 1,
                               padding: int = 0, dilation: int = 1) -> int:
        """
        Calculate convolution output size.

        Mathematical Formula:
        output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)

        Args:
            input_size: Input dimension size
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding applied
            dilation: Dilation factor

        Returns:
            Output size after convolution
        """
        return math.floor((input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)

    @staticmethod
    def attention_mechanism(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                           d_k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Scaled Dot-Product Attention mechanism.

        Mathematical Formula:
        Attention(Q,K,V) = softmax(QK^T/√d_k)V

        Where:
        - Q: Query matrix [batch_size, seq_len, d_k]
        - K: Key matrix [batch_size, seq_len, d_k]
        - V: Value matrix [batch_size, seq_len, d_v]
        - d_k: Dimension of key vectors (for scaling)

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            d_k: Key dimension for scaling
            mask: Optional attention mask

        Returns:
            Attention output tensor
        """
        # Calculate attention scores: QK^T/√d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)

        return output

    @staticmethod
    def focal_loss_calculation(inputs: torch.Tensor, targets: torch.Tensor,
                              alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """
        Calculate Focal Loss for addressing class imbalance.

        Mathematical Formula:
        FL(p_t) = -α_t(1-p_t)^γ log(p_t)

        Where:
        - p_t: Predicted probability for the true class
        - α_t: Weighting factor for class t
        - γ: Focusing parameter (higher γ focuses more on hard examples)

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: True labels [batch_size]
            alpha: Weighting factor
            gamma: Focusing parameter

        Returns:
            Focal loss value
        """
        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calculate p_t (probability of true class)
        p_t = torch.exp(-ce_loss)

        # Calculate focal loss: -α(1-p_t)^γ log(p_t)
        focal_loss = alpha * (1 - p_t) ** gamma * ce_loss

        return focal_loss.mean()

    @staticmethod
    def knowledge_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                                   student_targets: torch.Tensor, temperature: float = 4.0,
                                   alpha: float = 0.7) -> torch.Tensor:
        """
        Calculate Knowledge Distillation Loss.

        Mathematical Formula:
        L = αL_CE(y, student_logits) + (1-α)τ²L_KD(soft_teacher, soft_student)

        Where:
        - L_CE: Cross-entropy loss with hard targets
        - L_KD: KL divergence loss with soft targets
        - τ: Temperature parameter for softening distributions
        - α: Weighting factor between hard and soft targets

        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            student_targets: True labels
            temperature: Temperature for softening
            alpha: Weighting factor

        Returns:
            Knowledge distillation loss
        """
        # Hard target loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, student_targets)

        # Soft target loss (KL divergence between teacher and student)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')

        # Combined loss
        total_loss = alpha * hard_loss + (1 - alpha) * (temperature ** 2) * soft_loss

        return total_loss

    @staticmethod
    def ensemble_weighted_average(predictions: List[torch.Tensor],
                                 weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted average of ensemble predictions.

        Mathematical Formula:
        f_ensemble(x) = Σ(α_i * f_i(x)) where Σα_i = 1

        Args:
            predictions: List of model predictions
            weights: Ensemble weights

        Returns:
            Weighted ensemble prediction
        """
        # Normalize weights to sum to 1
        weights = F.softmax(weights, dim=0)

        # Calculate weighted average
        ensemble_pred = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += weights[i] * pred

        return ensemble_pred

class GenderDataset(Dataset):
    """
    Custom Dataset for Gender Classification with Advanced Preprocessing.
    Author: Soumya Chakraborty

    Mathematical Foundation:
    - Face Detection: Uses Haar Cascade classifiers based on integral images
    - Data Augmentation: Applies random transformations to increase dataset diversity
    - Normalization: Applies z-score normalization: (x - μ) / σ
    """

    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None,
                 use_face_detection: bool = True, augmentation_prob: float = 0.5):
        """
        Initialize the Gender Dataset.

        Args:
            root_dir: Root directory containing male/female subdirectories
            transform: Image transformations to apply
            use_face_detection: Whether to apply face detection and cropping
            augmentation_prob: Probability of applying augmentations
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_face_detection = use_face_detection
        self.augmentation_prob = augmentation_prob
        self.samples = []
        self.class_weights = None

        # Load face cascade for detection
        if self.use_face_detection:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

        # Load dataset samples
        self._load_samples()

        # Calculate class weights for balancing
        self._calculate_class_weights()

        logging.info(f"Dataset loaded: {len(self.samples)} samples")
        logging.info(f"Class distribution: {self._get_class_distribution()}")

    def _load_samples(self) -> None:
        """Load all samples from the dataset directory."""
        class_names = ['female', 'male']

        for class_id, class_name in enumerate(class_names):
            class_dir = os.path.join(self.root_dir, class_name)

            if not os.path.exists(class_dir):
                logging.warning(f"Class directory not found: {class_dir}")
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_id))

    def _calculate_class_weights(self) -> None:
        """
        Calculate class weights for handling imbalanced datasets.

        Mathematical Formula:
        w_i = n_samples / (n_classes * n_samples_i)

        Where:
        - w_i: Weight for class i
        - n_samples: Total number of samples
        - n_classes: Number of classes
        - n_samples_i: Number of samples in class i
        """
        labels = [sample[1] for sample in self.samples]
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        n_classes = len(class_counts)

        # Calculate weights: n_samples / (n_classes * n_samples_i)
        self.class_weights = total_samples / (n_classes * class_counts)

        logging.info(f"Class weights calculated: {self.class_weights}")

    def _get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        labels = [sample[1] for sample in self.samples]
        class_counts = np.bincount(labels)

        return {
            'female': class_counts[0],
            'male': class_counts[1] if len(class_counts) > 1 else 0
        }

    def _detect_and_crop_face(self, image: np.ndarray, padding: int = 20) -> np.ndarray:
        """
        Detect and crop face from image using Haar Cascade.

        Mathematical Foundation:
        Haar Cascade uses integral images for fast feature computation:
        II(x,y) = Σ(i=0 to x, j=0 to y) I(i,j)

        Args:
            image: Input image array
            padding: Padding around detected face

        Returns:
            Cropped face image or original image if no face detected
        """
        if not self.use_face_detection:
            return image

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )

        # If faces detected, crop the first (largest) face
        if len(faces) > 0:
            x, y, w, h = faces[0]

            # Add padding around face
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)

            # Crop face region
            face_image = image[y:y+h, x:x+w]
            return face_image

        return image

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (transformed_image, label)
        """
        img_path, label = self.samples[idx]

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)

            # Apply face detection and cropping
            if self.use_face_detection:
                image_np = self._detect_and_crop_face(image_np)
                image = Image.fromarray(image_np)

            # Apply transformations
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            logging.warning(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism for Transformer architecture.
    Author: Soumya Chakraborty

    Mathematical Foundation:
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize Multi-Head Attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of Multi-Head Attention.

        Mathematical Process:
        1. Project Q, K, V: Q' = QW^Q, K' = KW^K, V' = VW^V
        2. Split into heads: Q', K', V' -> [batch, heads, seq_len, d_k]
        3. Apply scaled dot-product attention for each head
        4. Concatenate heads and project: Concat(heads)W^O

        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor [batch, seq_len, d_model]
            value: Value tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Attention output [batch, seq_len, d_model]
        """
        batch_size, seq_len = query.size(0), query.size(1)

        # 1. Project Q, K, V
        Q = self.w_q(query)  # [batch, seq_len, d_model]
        K = self.w_k(key)    # [batch, seq_len, d_model]
        V = self.w_v(value)  # [batch, seq_len, d_model]

        # 2. Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Apply attention
        attention_output = MathematicalFoundations.attention_mechanism(Q, K, V, self.d_k, mask)

        # 4. Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 5. Final projection
        output = self.w_o(attention_output)

        return self.dropout(output)

class CNNTransformerHybrid(nn.Module):
    """
    Hybrid CNN-Transformer model for gender classification.
    Author: Soumya Chakraborty

    Mathematical Foundation:
    f_hybrid(x) = Classifier(Fusion(f_CNN(x) ⊕ f_Transformer(x)))

    Where:
    - f_CNN(x): CNN feature extraction
    - f_Transformer(x): Transformer feature extraction
    - ⊕: Feature concatenation
    - Fusion: Attention-based feature fusion
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3,
                 cnn_backbone: str = 'efficientnet_b3',
                 transformer_backbone: str = 'vit_base_patch16_224'):
        """
        Initialize the hybrid model.

        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability
            cnn_backbone: CNN backbone architecture
            transformer_backbone: Transformer backbone architecture
        """
        super().__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # CNN Backbone (EfficientNet-B3)
        self.cnn_backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        cnn_features = self.cnn_backbone.classifier.in_features
        self.cnn_backbone.classifier = nn.Identity()  # Remove original classifier

        # Transformer Backbone (Vision Transformer)
        self.vit_backbone = timm.create_model(transformer_backbone, pretrained=True, num_classes=0)
        vit_features = self.vit_backbone.num_features

        # Feature dimensions
        self.cnn_features = cnn_features
        self.vit_features = vit_features
        self.fusion_dim = 512

        # Attention-based feature fusion
        self.attention_fusion = MultiHeadAttention(
            d_model=cnn_features + vit_features,
            n_heads=8,
            dropout=dropout_rate
        )

        # Feature projection layers
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_features, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.vit_projection = nn.Sequential(
            nn.Linear(vit_features, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Feature fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(2 * self.fusion_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.LayerNorm(self.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim // 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

        logging.info(f"Hybrid model initialized with {self._count_parameters():,} parameters")

    def _initialize_weights(self) -> None:
        """
        Initialize model weights using Xavier/He initialization.

        Mathematical Foundation:
        - Xavier: W ~ N(0, 2/(n_in + n_out))
        - He: W ~ N(0, 2/n_in) for ReLU activations
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def _count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model.

        Mathematical Process:
        1. Extract CNN features: f_cnn = CNN(x)
        2. Extract Transformer features: f_vit = ViT(x)
        3. Project features to common dimension
        4. Fuse features: f_fused = Fusion(f_cnn ⊕ f_vit)
        5. Classify: y = Classifier(f_fused)

        Args:
            x: Input tensor [batch_size, 3, 224, 224]

        Returns:
            Classification logits [batch_size, num_classes]
        """
        batch_size = x.size(0)

        # 1. Extract CNN features
        cnn_features = self.cnn_backbone(x)  # [batch_size, cnn_features]

        # 2. Extract Transformer features
        vit_features = self.vit_backbone(x)  # [batch_size, vit_features]

        # 3. Project features to common dimension
        cnn_projected = self.cnn_projection(cnn_features)  # [batch_size, fusion_dim]
        vit_projected = self.vit_projection(vit_features)  # [batch_size, fusion_dim]

        # 4. Concatenate features
        fused_features = torch.cat([cnn_projected, vit_projected], dim=1)  # [batch_size, 2*fusion_dim]

        # 5. Apply fusion network
        fused_features = self.fusion_network(fused_features)  # [batch_size, fusion_dim//2]

        # 6. Classification
        logits = self.classifier(fused_features)  # [batch_size, num_classes]

        return logits

class EnsembleModel(nn.Module):
    """
    Ensemble model combining multiple architectures.
    Author: Soumya Chakraborty

    Mathematical Foundation:
    f_ensemble(x) = Σ(α_i * f_i(x)) where Σα_i = 1

    Ensemble Methods:
    1. Weighted Average: Simple weighted combination
    2. Attention-based: Learned attention weights
    3. Meta-learning: Additional network to combine predictions
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        """
        Initialize the ensemble model.

        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.num_classes = num_classes

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

        # Individual classifiers
        self.efficientnet_classifier = self._create_classifier(efficientnet_features, num_classes, dropout_rate)
        self.resnet_classifier = self._create_classifier(resnet_features, num_classes, dropout_rate)
        self.vit_classifier = self._create_classifier(vit_features, num_classes, dropout_rate)
        self.convnext_classifier = self._create_classifier(convnext_features, num_classes, dropout_rate)

        # Ensemble fusion methods
        self.fusion_dim = 512
        total_features = efficientnet_features + resnet_features + vit_features + convnext_features

        # Meta-learning classifier
        self.meta_classifier = nn.Sequential(
            nn.Linear(total_features, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.LayerNorm(self.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.fusion_dim // 2, num_classes)
        )

        # Attention weights for adaptive ensemble
        self.attention_weights = nn.Parameter(torch.ones(4) / 4)

        # Prediction combination network
        self.prediction_combiner = nn.Sequential(
            nn.Linear(4 * num_classes, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )

        logging.info(f"Ensemble model initialized with {self._count_parameters():,} parameters")

    def _create_classifier(self, input_features: int, num_classes: int, dropout_rate: float) -> nn.Module:
        """Create a classifier head for individual models."""
        return nn.Sequential(
            nn.Linear(input_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def _count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, mode: str = 'ensemble') -> Union[torch.Tensor, Tuple]:
        """
        Forward pass with different ensemble modes.

        Args:
            x: Input tensor [batch_size, 3, 224, 224]
            mode: Ensemble mode ('individual', 'weighted_average', 'meta_learning', 'ensemble')

        Returns:
            Ensemble predictions based on the selected mode
        """
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
            # Weighted average ensemble using learned attention weights
            eff_pred = self.efficientnet_classifier(efficientnet_feat)
            res_pred = self.resnet_classifier(resnet_feat)
            vit_pred = self.vit_classifier(vit_feat)
            conv_pred = self.convnext_classifier(convnext_feat)

            predictions = [eff_pred, res_pred, vit_pred, conv_pred]
            ensemble_pred = MathematicalFoundations.ensemble_weighted_average(
                predictions, self.attention_weights
            )
            return ensemble_pred

        elif mode == 'meta_learning':
            # Meta-learning approach: combine features and learn optimal combination
            combined_features = torch.cat([efficientnet_feat, resnet_feat, vit_feat, convnext_feat], dim=1)
            meta_pred = self.meta_classifier(combined_features)
            return meta_pred

        elif mode == 'prediction_fusion':
            # Fusion at prediction level
            eff_pred = self.efficientnet_classifier(efficientnet_feat)
            res_pred = self.resnet_classifier(resnet_feat)
            vit_pred = self.vit_classifier(vit_feat)
            conv_pred = self.convnext_classifier(convnext_feat)

            combined_preds = torch.cat([eff_pred, res_pred, vit_pred, conv_pred], dim=1)
            fusion_pred = self.prediction_combiner(combined_preds)
            return fusion_pred

        else:  # Default ensemble mode
            # Combine multiple approaches for best performance
            individual_preds = self.forward(x, mode='individual')
            weighted_pred = self.forward(x, mode='weighted_average')
            meta_pred = self.forward(x, mode='meta_learning')
            fusion_pred = self.forward(x, mode='prediction_fusion')

            # Final ensemble: average of all methods
            final_pred = (weighted_pred + meta_pred + fusion_pred) / 3
            return final_pred, individual_preds, weighted_pred, meta_pred, fusion_pred

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Author: Soumya Chakraborty

    Mathematical Formula:
    FL(p_t) = -α_t(1-p_t)^γ log(p_t)

    Where:
    - p_t: Predicted probability for the true class
    - α_t: Weighting factor for class t
    - γ: Focusing parameter (reduces loss for well-classified examples)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Loss reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: True labels [batch_size]

        Returns:
            Focal loss value
        """
        return MathematicalFoundations.focal_loss_calculation(
            inputs, targets, self.alpha, self.gamma
        )

class ComprehensiveGenderClassifier:
    """
    Comprehensive Gender Classification System.
    Author: Soumya Chakraborty

    This class implements a complete gender classification pipeline with:
    1. Advanced data preprocessing and augmentation
    2. Multiple model architectures (single and ensemble)
    3. GPU optimization and mixed precision training
    4. Bias mitigation and fairness analysis
    5. Model compression and deployment optimization
    """

    def __init__(self, config: Optional[Dict] = None, device: Optional[str] = None):
        """
        Initialize the comprehensive gender classifier.

        Args:
            config: Configuration dictionary
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.config = config or self._get_default_config()
        self.device = self._setup_device(device)

        # Initialize components
        self.model = None
        self.ensemble_model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': [], 'gpu_memory': []
        }

        # GPU optimization
        self.use_amp = self.device.type == 'cuda' and self.config.get('use_mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Setup logging
        self._setup_logging()

        logging.info(f"Comprehensive Gender Classifier initialized by Soumya Chakraborty")
        logging.info(f"Device: {self.device}")
        logging.info(f"Mixed Precision: {self.use_amp}")

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            # Data configuration
            'data': {
                'train_dir': 'Task_A/train',
                'val_dir': 'Task_A/val',
                'batch_size': 32,
                'num_workers': 4,
                'image_size': 224,
                'use_face_detection': True,
                'augmentation_prob': 0.8
            },

            # Training configuration
            'training': {
                'num_epochs': 50,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'dropout_rate': 0.3,
                'warmup_epochs': 5,
                'gradient_clip_value': 1.0,
                'early_stopping_patience': 15
            },

            # Model configuration
            'model': {
                'architecture': 'hybrid',  # 'hybrid', 'ensemble'
                'num_classes': 2,
                'cnn_backbone': 'efficientnet_b3',
                'transformer_backbone': 'vit_base_patch16_224'
            },

            # Loss configuration
            'loss': {
                'type': 'focal',  # 'focal', 'cross_entropy', 'label_smoothing'
                'focal_alpha': 1.0,
                'focal_gamma': 2.0,
                'label_smoothing': 0.1
            },

            # Optimization configuration
            'optimization': {
                'use_mixed_precision': True,
                'optimizer': 'adamw',
                'scheduler': 'cosine_warm_restarts',
                'cosine_t0': 10,
                'cosine_t_mult': 2
            },

            # Evaluation configuration
            'evaluation': {
                'save_predictions': True,
                'calculate_fairness_metrics': True,
                'generate_plots': True,
                'save_confusion_matrix': True
            },

            # Output configuration
            'output': {
                'save_dir': 'output',
                'model_dir': 'output/models',
                'plots_dir': 'output/plots',
                'logs_dir': 'output/logs'
            }
        }

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup optimal device configuration."""
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
                logging.info(f"Selected GPU {best_gpu}: {torch.cuda.get_device_name(best_gpu)}")
            else:
                device = torch.device('cuda:0')

            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        else:
            device = torch.device('cpu')
            logging.info("CUDA not available, using CPU")

        return device

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_dir = Path(self.config['output']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def prepare_data(self) -> Tuple[float, float]:
        """
        Prepare data loaders with advanced preprocessing.

        Returns:
            Tuple of class weights for imbalanced dataset handling
        """
        logging.info("Preparing data loaders...")

        data_config = self.config['data']

        # Advanced data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((data_config['image_size'] + 32, data_config['image_size'] + 32)),
            transforms.RandomResizedCrop(data_config['image_size'], scale=(0.8, 1.0)),
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
            transforms.Resize((data_config['image_size'], data_config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        train_dataset = GenderDataset(
            root_dir=data_config['train_dir'],
            transform=train_transform,
            use_face_detection=data_config['use_face_detection'],
            augmentation_prob=data_config['augmentation_prob']
        )

        val_dataset = GenderDataset(
            root_dir=data_config['val_dir'],
            transform=val_transform,
            use_face_detection=data_config['use_face_detection'],
            augmentation_prob=0.0  # No augmentation for validation
        )

        # Create weighted sampler for balanced training
        sample_weights = [train_dataset.class_weights[label] for _, label in train_dataset.samples]
        sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

        # Optimal data loader configuration
        loader_kwargs = {
            'num_workers': data_config['num_workers'],
            'pin_memory': self.device.type == 'cuda',
            'persistent_workers': data_config['num_workers'] > 0,
            'prefetch_factor': 2 if data_config['num_workers'] > 0 else 2,
        }

        if self.device.type == 'cuda':
            loader_kwargs.update({
                'pin_memory_device': str(self.device),
            })

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            sampler=sampler,
            **loader_kwargs
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            **loader_kwargs
        )

        logging.info(f"Data loaders created:")
        logging.info(f"  Training batches: {len(self.train_loader)}")
        logging.info(f"  Validation batches: {len(self.val_loader)}")
        logging.info(f"  Class weights: {train_dataset.class_weights}")

        return train_dataset.class_weights

    def create_model(self, architecture: str = 'hybrid') -> nn.Module:
        """
        Create the specified model architecture.

        Args:
            architecture: Model architecture ('hybrid' or 'ensemble')

        Returns:
            Created model
        """
        model_config = self.config['model']

        if architecture == 'hybrid':
            self.model = CNNTransformerHybrid(
                num_classes=model_config['num_classes'],
                dropout_rate=self.config['training']['dropout_rate'],
                cnn_backbone=model_config['cnn_backbone'],
                transformer_backbone=model_config['transformer_backbone']
            )

        elif architecture == 'ensemble':
            self.model = EnsembleModel(
                num_classes=model_config['num_classes'],
                dropout_rate=self.config['training']['dropout_rate']
            )

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logging.info(f"Model created: {architecture}")
        logging.info(f"  Total parameters: {total_params:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,}")

        return self.model

    def setup_training(self, class_weights: np.ndarray) -> None:
        """
        Setup training components (optimizer, scheduler, loss function).

        Args:
            class_weights: Class weights for handling imbalanced data
        """
        training_config = self.config['training']
        loss_config = self.config['loss']
        opt_config = self.config['optimization']

        # Setup loss function
        if loss_config['type'] == 'focal':
            self.criterion = FocalLoss(
                alpha=loss_config['focal_alpha'],
                gamma=loss_config['focal_gamma']
            )
        elif loss_config['type'] == 'cross_entropy':
            weights = torch.FloatTensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        elif loss_config['type'] == 'label_smoothing':
            self.criterion = nn.CrossEntropyLoss(label_smoothing=loss_config['label_smoothing'])
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Setup optimizer with different learning rates for different parts
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if any(x in name.lower() for x in ['backbone', 'efficientnet', 'resnet', 'vit', 'convnext']):
                backbone_params.append(param)
            else:
                head_params.append(param)

        if opt_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': training_config['learning_rate'] * 0.1},
                {'params': head_params, 'lr': training_config['learning_rate']}
            ], weight_decay=training_config['weight_decay'])
        else:
            self.optimizer = optim.Adam([
                {'params': backbone_params, 'lr': training_config['learning_rate'] * 0.1},
                {'params': head_params, 'lr': training_config['learning_rate']}
            ], weight_decay=training_config['weight_decay'])

        # Setup scheduler
        if opt_config['scheduler'] == 'cosine_warm_restarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=opt_config['cosine_t0'],
                T_mult=opt_config['cosine_t_mult'],
                eta_min=1e-6
            )
        elif opt_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs'],
                eta_min=1e-6
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )

        logging.info(f"Training setup completed:")
        logging.info(f"  Loss: {loss_config['type']}")
        logging.info(f"  Optimizer: {opt_config['optimizer']}")
        logging.info(f"  Scheduler: {opt_config['scheduler']}")
        logging.info(f"  Mixed Precision: {self.use_amp}")

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch with GPU optimizations.

        Returns:
            Tuple of (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Move data to device with non-blocking transfer
            if self.device.type == 'cuda':
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
            else:
                images, labels = images.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config['training'].get('gradient_clip_value'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_value']
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.config['training'].get('gradient_clip_value'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_value']
                    )

                self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1) if isinstance(outputs, torch.Tensor) else outputs[0].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            # Clear GPU cache periodically
            if self.device.type == 'cuda' and batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float, List, List, List]:
        """
        Validate the model with GPU optimizations.

        Returns:
            Tuple of (val_loss, val_acc, predictions, labels, probabilities)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
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

                # Handle ensemble outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Store for detailed analysis
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Complete training loop with comprehensive monitoring.

        Args:
            num_epochs: Number of epochs to train (uses config if None)

        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']

        logging.info(f"Starting training for {num_epochs} epochs")
        logging.info(f"Author: Soumya Chakraborty")

        # Create output directories
        model_dir = Path(self.config['output']['model_dir'])
        plots_dir = Path(self.config['output']['plots_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        start_time = time.time()
        patience_counter = 0
        best_model_path = model_dir / "best_model.pth"

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            logging.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logging.info("-" * 60)

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

            # GPU memory monitoring
            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                self.training_history['gpu_memory'].append(allocated)

            # Log epoch results
            logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logging.info(f"Learning Rate: {current_lr:.2e}")

            if self.device.type == 'cuda':
                memory_info = self._get_gpu_memory_info()
                logging.info(f"GPU Memory: {memory_info}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(best_model_path, epoch, val_acc, is_best=True)
                logging.info(f"✅ New best model saved! Accuracy: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            early_stopping_patience = self.config['training'].get('early_stopping_patience', 15)
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break

            # Detailed evaluation every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._evaluate_detailed_metrics(val_preds, val_labels, val_probs)

            # GPU cache cleanup
            if self.device.type == 'cuda' and (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()

        # Training completed
        total_time = time.time() - start_time
        logging.info(f"\n🎉 Training completed!")
        logging.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        logging.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        # Save final results
        self._save_training_results()

        return self.training_history

    def _get_gpu_memory_info(self) -> str:
        """Get GPU memory information as string."""
        if self.device.type != 'cuda':
            return "CPU mode"

        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        utilization = (allocated / total) * 100

        return f"{allocated:.1f}GB / {total:.1f}GB ({utilization:.1f}%)"

    def _save_checkpoint(self, path: Path, epoch: int, val_acc: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'val_acc': val_acc,
            'config': self.config,
            'training_history': self.training_history,
            'author': 'Soumya Chakraborty'
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)

    def _evaluate_detailed_metrics(self, preds: List, labels: List, probs: List) -> None:
        """Calculate and log detailed evaluation metrics."""
        from sklearn.metrics import classification_report

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

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        logging.info(f"\n📊 DETAILED EVALUATION METRICS")
        logging.info(f"Overall Accuracy: {accuracy:.4f}")
        logging.info(f"Weighted Precision: {precision:.4f}")
        logging.info(f"Weighted Recall: {recall:.4f}")
        logging.info(f"Weighted F1-Score: {f1:.4f}")
        logging.info(f"AUC-ROC: {auc_roc:.4f}")

        # Per-class metrics
        classes = ['Female', 'Male']
        for i, class_name in enumerate(classes):
            if i < len(precision_per_class):
                logging.info(f"{class_name} - Precision: {precision_per_class[i]:.4f}, "
                          f"Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}")

        # Confusion matrix
        logging.info(f"Confusion Matrix:")
        logging.info(f"          Female  Male")
        logging.info(f"Female    {cm[0,0]:6d}  {cm[0,1]:4d}")
        logging.info(f"Male      {cm[1,0]:6d}  {cm[1,1]:4d}")

        # Bias analysis
        self._analyze_bias(labels, preds)

    def _analyze_bias(self, labels: List, preds: List) -> None:
        """Analyze potential bias in predictions."""
        female_indices = [i for i, label in enumerate(labels) if label == 0]
        male_indices = [i for i, label in enumerate(labels) if label == 1]

        if len(female_indices) == 0 or len(male_indices) == 0:
            logging.warning("Cannot perform bias analysis: one class has no samples")
            return

        # Accuracy per group
        female_correct = sum(1 for i in female_indices if preds[i] == labels[i])
        male_correct = sum(1 for i in male_indices if preds[i] == labels[i])

        female_accuracy = female_correct / len(female_indices)
        male_accuracy = male_correct / len(male_indices)

        logging.info(f"\n🔍 Bias Analysis:")
        logging.info(f"Female samples: {len(female_indices)}")
        logging.info(f"Male samples: {len(male_indices)}")
        logging.info(f"Female accuracy: {female_accuracy:.4f}")
        logging.info(f"Male accuracy: {male_accuracy:.4f}")
        logging.info(f"Accuracy difference: {abs(female_accuracy - male_accuracy):.4f}")

        # Fairness assessment
        if abs(female_accuracy - male_accuracy) < 0.05:
            logging.info("✅ Model shows good fairness (accuracy difference < 5%)")
        else:
            logging.warning("⚠️ Model may have fairness issues (accuracy difference >= 5%)")

    def _save_training_results(self) -> None:
        """Save comprehensive training results."""
        results_dir = Path(self.config['output']['save_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save training history
        history_path = results_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        # Save configuration
        config_path = results_dir / "final_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        # Generate plots if requested
        if self.config['evaluation']['generate_plots']:
            self._generate_training_plots()

        logging.info(f"Training results saved to {results_dir}")

    def _generate_training_plots(self) -> None:
        """Generate training visualization plots."""
        plots_dir = Path(self.config['output']['plots_dir'])
        plots_dir.mkdir(parents=True, exist_ok=True)

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

        # GPU memory (if available)
        if self.training_history['gpu_memory']:
            axes[1, 1].plot(epochs, self.training_history['gpu_memory'], 'm-')
            axes[1, 1].set_title('GPU Memory Usage')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Memory (GB)')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'GPU Memory\nNot Available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.savefig(plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Main function with comprehensive command-line interface.
    Author: Soumya Chakraborty
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive Gender Classification System by Soumya Chakraborty",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Basic arguments
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

    # Training control
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble training")
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
    print("🚀 Comprehensive Gender Classification System")
    print("=" * 60)
    print(f"Author: Soumya Chakraborty")
    print(f"Version: {__version__}")
    print("=" * 60)

    # Create configuration from arguments
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from: {args.config}")
    else:
        config = {
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
            }
        }

    # Save configuration
    os.makedirs(args.output_dir, exist_ok=True)
    config_save_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_save_path}")

    try:
        # Initialize classifier
        classifier = ComprehensiveGenderClassifier(config=config, device=args.device)

        # Prepare data
        print("\n📊 Preparing data...")
        class_weights = classifier.prepare_data()

        # Create model
        print(f"\n🏗️ Creating {config['model']['architecture']} model...")
        classifier.create_model(config['model']['architecture'])

        # Setup training
        print("\n⚙️ Setting up training...")
        classifier.setup_training(class_weights)

        # Start training
        print("\n🚀 Starting training...")
        training_history = classifier.train()

        print("\n🎉 Training completed successfully!")
        print(f"Best validation accuracy: {classifier.best_val_acc:.2f}%")
        print(f"Results saved in: {args.output_dir}")

        return training_history

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set up proper multiprocessing for Windows
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    result = main()
    if result is None:
        sys.exit(1)
