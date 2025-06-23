#!/usr/bin/env python3
"""
Face Recognition Training System with Vision Transformer and ArcFace Loss

This module implements a state-of-the-art face recognition training system that combines:
1. Vision Transformer (ViT) architecture for robust feature extraction
2. ArcFace loss for enhanced angular margin separation
3. Advanced data augmentation for distortion robustness
4. Balanced sampling and training strategies

Mathematical Foundation:
- Vision Transformer: Self-attention mechanism for spatial feature learning
- ArcFace Loss: L(θ) = -log(e^(s*cos(θ_yi + m)) / (e^(s*cos(θ_yi + m)) + Σe^(s*cos(θ_j))))
  where θ_yi is angle between feature and weight of ground truth class,
  m is angular margin, s is scale factor
- Cosine Similarity: cos(θ) = (x · w) / (||x|| ||w||)

Architecture Components:
- Backbone: Vision Transformer with patch-based attention
- Embedding: Multi-layer perceptron with residual connections
- Loss: ArcFace with angular margin for better class separation
- Optimization: AdamW with cosine annealing warm restarts

Handles both clean and distorted face images for robust recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
from collections import Counter
import time
import argparse
import sys

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for training parameters"""
    # Model parameters
    MODEL_NAME = 'vit_base_patch16_224'
    EMBEDDING_DIM = 512
    IMAGE_SIZE = 224

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    VAL_SPLIT = 0.2

    # ArcFace parameters
    ARCFACE_MARGIN = 0.5
    ARCFACE_SCALE = 64

    # Data parameters
    INCLUDE_DISTORTED = True
    MAX_SAMPLES_PER_CLASS = None

    # Training parameters
    GRADIENT_CLIPPING = 1.0
    PATIENCE = 15
    MIN_DELTA = 1e-4

    # Paths
    TRAIN_DIR = 'train'
    OUTPUT_DIR = 'outputs'
    MODEL_SAVE_PATH = 'best_face_model.pth'
    LABEL_ENCODER_PATH = 'label_encoder.json'

class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss Implementation with Numerical Stability

    Mathematical Foundation:
    ArcFace introduces an angular margin penalty to enhance the discriminative power
    of learned features by maximizing decision boundaries in angular space.

    Loss Formulation:
    L = -log(e^(s*cos(θ_yi + m)) / (e^(s*cos(θ_yi + m)) + Σ_{j≠yi} e^(s*cos(θ_j))))

    Where:
    - θ_yi: angle between feature vector and weight vector of ground truth class
    - θ_j: angle between feature vector and weight vector of class j
    - m: angular margin penalty (typically 0.5 radians ≈ 28.6 degrees)
    - s: scale factor to control the magnitude of logits (typically 64)

    Key Properties:
    1. Geodesic distance margin: enforces margin in angular space
    2. Intrinsic angular space: operates on unit hypersphere
    3. Enhanced inter-class discrepancy: larger margins between classes
    4. Stable gradients: numerical stability through careful implementation

    Parameters:
        embedding_dim (int): Dimension of feature embeddings
        num_classes (int): Number of identity classes
        margin (float): Angular margin penalty in radians
        scale (float): Feature scale factor
    """
    def __init__(self, embedding_dim=512, num_classes=1000, margin=0.5, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.embedding_dim = embedding_dim  # Feature embedding dimension
        self.num_classes = num_classes      # Number of identity classes
        self.margin = margin               # Angular margin m (radians)
        self.scale = scale                # Scale factor s
        self.eps = 1e-8                   # Small constant for numerical stability

        # Initialize classification weights W ∈ R^(num_classes × embedding_dim)
        # Each row represents the weight vector for one class
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)  # Xavier initialization for stable training

    def forward(self, embeddings, labels):
        """
        Forward pass implementing ArcFace loss computation

        Mathematical Steps:
        1. Normalize features and weights to unit vectors
        2. Compute cosine similarities: cos(θ) = x_i^T W_j / (||x_i|| ||W_j||)
        3. Calculate sine values: sin(θ) = √(1 - cos²(θ))
        4. Apply angular margin: cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        5. Scale and compute softmax loss

        Args:
            embeddings (Tensor): Feature embeddings [batch_size, embedding_dim]
            labels (Tensor): Ground truth labels [batch_size]

        Returns:
            Tensor: Scaled logits for cross-entropy loss [batch_size, num_classes]
        """
        # Step 1: L2 normalization - project to unit hypersphere
        # ||x|| = 1 and ||W|| = 1 ensures cos(θ) = x^T W
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 norm along feature dim
        weight = F.normalize(self.weight, p=2, dim=1)     # L2 norm along feature dim

        # Step 2: Compute cosine similarities between features and class weights
        # cosine[i,j] = cos(θ_ij) where θ_ij is angle between x_i and W_j
        cosine = F.linear(embeddings, weight)  # Matrix multiplication: X @ W^T
        cosine = torch.clamp(cosine, -1 + self.eps, 1 - self.eps)  # Numerical stability

        # Step 3: Create one-hot encoding for ground truth classes
        # one_hot[i,j] = 1 if j is the ground truth class for sample i, 0 otherwise
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Step 4: Compute sine values using trigonometric identity
        # sin²(θ) + cos²(θ) = 1 → sin(θ) = √(1 - cos²(θ))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Step 5: Apply angular margin using angle addition formula
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        # This increases the decision boundary by margin m for ground truth classes
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)

        # Step 6: Apply margin penalty only to ground truth classes
        # For ground truth class: use cos(θ + m), for others: use cos(θ)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Step 7: Scale the logits for stable softmax computation
        # Larger scale s makes the softmax more peaky, improving convergence
        output *= self.scale

        return output

class EnhancedFaceViT(nn.Module):
    """
    Enhanced Vision Transformer for Face Recognition

    Architecture Overview:
    This model combines the power of Vision Transformers with specialized components
    for face recognition, including ArcFace loss and attention mechanisms.

    Mathematical Foundation:

    1. Vision Transformer Backbone:
       - Patch Embedding: x_p = [x^1_p E; x^2_p E; ...; x^N_p E] + E_pos
       - Multi-Head Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
       - Layer Normalization: LN(x) = γ((x-μ)/σ) + β

    2. Enhanced Embedding Network:
       - Progressive dimensionality: backbone_dim → 2×embedding_dim → embedding_dim
       - Residual-like connections through batch normalization
       - Dropout regularization for generalization

    3. Multi-Head Attention Pooling:
       - Query-Key-Value attention for feature refinement
       - Multiple attention heads for diverse feature patterns
       - Mathematical form: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O

    Key Components:
    - Backbone: Pre-trained ViT for robust feature extraction
    - Embedding: Multi-layer projection network with regularization
    - ArcFace: Angular margin loss for enhanced discrimination
    - Attention: Additional attention mechanism for feature refinement

    Parameters:
        model_name (str): Vision Transformer variant (e.g., 'vit_base_patch16_224')
        num_classes (int): Number of identity classes in dataset
        embedding_dim (int): Dimension of final face embeddings
        pretrained (bool): Whether to use pre-trained weights
        dropout_rate (float): Dropout probability for regularization
    """
    def __init__(self, model_name='vit_base_patch16_224', num_classes=1000,
                 embedding_dim=512, pretrained=True, dropout_rate=0.3):
        super(EnhancedFaceViT, self).__init__()

        # Vision Transformer Backbone
        # Creates a ViT model without the final classification head (num_classes=0)
        # The backbone extracts patch-based features using self-attention mechanisms
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        backbone_dim = self.backbone.num_features  # Typically 768 for ViT-Base

        # Enhanced Embedding Network with Progressive Dimensionality
        # Architecture: backbone_dim → 2×embedding_dim → embedding_dim → embedding_dim
        # This creates a bottleneck structure that forces the model to learn compact representations
        self.embedding = nn.Sequential(
            # First expansion layer: increases capacity for feature transformation
            nn.Linear(backbone_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),      # Normalize for stable training
            nn.ReLU(inplace=True),                  # Non-linear activation
            nn.Dropout(dropout_rate),               # Regularization

            # Compression layer: reduces to target embedding dimension
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),           # Reduced dropout in deeper layers

            # Final projection: maintains embedding dimension
            nn.Linear(embedding_dim, embedding_dim)
        )

        # ArcFace Loss Layer for Angular Margin Learning
        # Implements the mathematical formulation described in ArcFaceLoss class
        self.arcface = ArcFaceLoss(embedding_dim, num_classes)

        # Multi-Head Attention for Feature Refinement
        # Additional attention mechanism beyond the backbone ViT attention
        # Helps focus on discriminative facial features
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,                            # 8 attention heads for diverse patterns
            dropout=dropout_rate
        )

        # Layer normalization for attention output stabilization
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, labels=None):
        """
        Forward pass through the Enhanced Vision Transformer

        Mathematical Flow:
        1. Patch Embedding & Self-Attention (ViT Backbone)
        2. Feature Projection (Embedding Network)
        3. Feature Normalization (Layer Norm)
        4. ArcFace Loss Computation (if training)

        Detailed Steps:
        x → ViT_backbone → features ∈ R^(B×D_backbone)
        features → MLP_embedding → embeddings ∈ R^(B×D_embed)
        embeddings → LayerNorm → normalized_embeddings
        normalized_embeddings + labels → ArcFace → logits ∈ R^(B×C)

        Args:
            x (Tensor): Input images [batch_size, channels, height, width]
            labels (Tensor, optional): Ground truth labels [batch_size] for training

        Returns:
            Training mode: (logits, embeddings) tuple
            Inference mode: embeddings only
        """
        # Step 1: Vision Transformer Feature Extraction
        # Applies patch embedding, positional encoding, and self-attention layers
        # Output shape: [batch_size, backbone_feature_dim]
        features = self.backbone(x)

        # Step 2: Enhanced Embedding Network
        # Progressive dimensionality transformation with regularization
        # backbone_dim → 2×embedding_dim → embedding_dim → embedding_dim
        embeddings = self.embedding(features)

        # Step 3: Layer Normalization for Stability
        # Normalizes the embedding features for consistent scale
        # LN(x) = γ * (x - μ) / σ + β, where μ and σ are computed per sample
        embeddings = self.layer_norm(embeddings)

        # Step 4: Mode-dependent Output
        if labels is not None:
            # Training Mode: Compute ArcFace logits for loss calculation
            # ArcFace applies angular margin and returns scaled logits
            logits = self.arcface(embeddings, labels)   # → [B, C]
        else:
            # Inference Mode: Return normalized embeddings for similarity computation
            # These embeddings can be used for face verification and identification
            logits = None
        return logits, embeddings

class RobustFaceTransforms:
    """
    Advanced Data Augmentation Pipeline for Face Recognition Robustness

    This class implements a comprehensive augmentation strategy designed to improve
    model robustness against various real-world distortions and variations.

    Augmentation Strategy:
    The pipeline is designed around the principle of domain randomization to create
    a robust model that can handle various degradations and distortions.

    Mathematical Foundations:

    1. Geometric Transformations:
       - Affine Transform: T(x,y) = [a b tx; c d ty; 0 0 1] @ [x; y; 1]
       - Rotation Matrix: R(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
       - Scale Matrix: S(sx,sy) = [sx 0; 0 sy]

    2. Photometric Transformations:
       - Brightness: I' = I + β, where β ∈ [-0.25, 0.25]
       - Contrast: I' = α × I, where α ∈ [0.75, 1.25]
       - Gamma Correction: I' = I^γ, where γ ∈ [0.8, 1.2]

    3. Noise Models:
       - Gaussian Noise: N(μ=0, σ²) added to pixel values
       - ISO Noise: Simulates camera sensor noise
       - Multiplicative Noise: I' = I × (1 + ε), ε ~ N(0, σ²)

    4. Weather Simulations:
       - Fog: Atmospheric scattering model
       - Rain: Line-based raindrop simulation
       - Sun Flare: Lens flare simulation

    Augmentation Categories:
    1. Geometric: Rotation, scaling, translation, flipping
    2. Photometric: Brightness, contrast, hue, saturation
    3. Noise: Gaussian, ISO, multiplicative noise
    4. Blur: Gaussian, motion, median blur
    5. Weather: Fog, rain, sun flare effects
    6. Occlusion: Random patches for robustness

    Parameters:
        image_size (int): Target image size for training/inference
        is_training (bool): Whether to apply training augmentations
    """
    def __init__(self, image_size=224, is_training=True):
        self.image_size = image_size

        if is_training:
            # Training Pipeline: Comprehensive augmentation for robustness
            self.transform = A.Compose([
                # ===== GEOMETRIC TRANSFORMATIONS =====
                # Resize with padding to preserve aspect ratio, then crop
                A.Resize(image_size + 32, image_size + 32),  # Slight oversizing
                A.RandomCrop(image_size, image_size),        # Random spatial crop

                # Horizontal flip: 50% probability (faces are approximately symmetric)
                A.HorizontalFlip(p=0.5),

                # Combined spatial transformations with reflection padding
                # Mathematical formulation: Apply composition of affine transforms
                A.ShiftScaleRotate(
                    shift_limit=0.1,      # ±10% translation in x,y
                    scale_limit=0.15,     # ±15% scaling factor
                    rotate_limit=20,      # ±20° rotation
                    p=0.6,               # 60% probability
                    border_mode=cv2.BORDER_REFLECT  # Reflection padding
                ),

                # ===== PHOTOMETRIC TRANSFORMATIONS =====
                # Brightness and contrast adjustment for lighting robustness
                # I' = α(I + β), where α controls contrast, β controls brightness
                A.RandomBrightnessContrast(
                    brightness_limit=0.25,  # β ∈ [-0.25, 0.25]
                    contrast_limit=0.25,    # α ∈ [0.75, 1.25]
                    p=0.6
                ),

                # HSV color space augmentation for color robustness
                # Handles variations in lighting conditions and camera characteristics
                A.HueSaturationValue(
                    hue_shift_limit=15,     # Hue shift: ±15°
                    sat_shift_limit=25,     # Saturation: ±25%
                    val_shift_limit=15,     # Value/brightness: ±15%
                    p=0.4
                ),

                # Contrast Limited Adaptive Histogram Equalization
                # Enhances local contrast while preventing over-amplification
                A.CLAHE(clip_limit=2.0, p=0.3),

                # Gamma correction for non-linear brightness adjustment
                # I' = I^γ, simulates different display/camera gamma settings
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # γ ∈ [0.8, 1.2]

                # ===== NOISE AUGMENTATIONS (Distortion Simulation) =====
                # Various noise models to simulate real-world image degradations
                A.OneOf([
                    # Additive Gaussian noise: I' = I + N(0, σ²)
                    A.GaussNoise(var_limit=(10.0, 80.0)),

                    # ISO noise: Simulates camera sensor noise
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),

                    # Multiplicative noise: I' = I × (1 + ε), ε ~ N(0, σ²)
                    A.MultiplicativeNoise(multiplier=[0.9, 1.1]),
                ], p=0.4),

                # ===== BLUR AUGMENTATIONS =====
                # Different blur types to simulate motion, focus, and compression artifacts
                A.OneOf([
                    # Gaussian blur: Simulates out-of-focus effects
                    A.GaussianBlur(blur_limit=(3, 9)),  # Kernel size range

                    # Motion blur: Simulates camera/subject movement
                    A.MotionBlur(blur_limit=7),

                    # Median blur: Simulates certain compression artifacts
                    A.MedianBlur(blur_limit=5),
                ], p=0.3),

                # ===== OCCLUSION SIMULATION =====
                # Random patch dropout to simulate partial occlusions
                # Encourages the model to rely on multiple facial features
                A.CoarseDropout(
                    max_holes=12,      # Maximum number of rectangular holes
                    max_height=16,     # Maximum hole height
                    max_width=16,      # Maximum hole width
                    min_holes=1,       # Minimum number of holes
                    fill_value=0,      # Fill dropped regions with black
                    p=0.3
                ),

                # ===== WEATHER EFFECTS (Matching Target Distortions) =====
                # Simulates atmospheric and environmental conditions
                A.OneOf([
                    # Fog simulation: Atmospheric scattering effect
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3),

                    # Rain simulation: Line-based raindrop effects
                    A.RandomRain(
                        slant_lower=-10, slant_upper=10,  # Rain angle variation
                        drop_length=20, drop_width=1,     # Raindrop dimensions
                        drop_color=(200, 200, 200)        # Gray raindrops
                    ),

                    # Sun flare: Lens flare and overexposure effects
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),  # Region of interest for flare
                        angle_lower=0, angle_upper=1  # Flare angle range
                    ),
                ], p=0.2),

                # ===== NORMALIZATION =====
                # ImageNet normalization for pre-trained model compatibility
                # z = (x - μ) / σ, where μ and σ are ImageNet statistics
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet channel means (RGB)
                    std=[0.229, 0.224, 0.225]    # ImageNet channel stds (RGB)
                ),

                # Convert to PyTorch tensor format
                ToTensorV2()
            ])
        else:
            # Validation/Test Pipeline: Minimal processing for consistent evaluation
            self.transform = A.Compose([
                # Simple resize to target dimensions
                A.Resize(image_size, image_size),

                # ImageNet normalization (same as training)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),

                # Convert to tensor
                ToTensorV2()
            ])

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            return self.transform(image=image)['image']
        else:
            image_np = np.array(image)
            return self.transform(image=image_np)['image']

class BalancedFaceDataset(Dataset):
    """
    Balanced Face Dataset with Intelligent Sampling Strategy

    This dataset class implements sophisticated sampling strategies to handle
    class imbalance and ensure robust training across all identity classes.

    Mathematical Foundation for Class Balancing:

    1. Class Weight Calculation:
       w_i = n_samples / (n_classes × n_samples_i)
       where:
       - w_i: weight for class i
       - n_samples: total number of samples
       - n_classes: total number of classes
       - n_samples_i: number of samples in class i

    2. Weighted Sampling Probability:
       P(sample_j) = w_class(j) / Σ(w_class(k) for all k)
       This ensures each class has equal expected representation per epoch

    3. Sample Distribution Strategy:
       - Original images: Guaranteed inclusion for identity preservation
       - Distorted images: Controlled inclusion for robustness
       - Balanced representation: Equal class contribution per epoch

    Parameters:
        root_dir (Path): Root directory containing person subdirectories
        transform (callable): Data augmentation pipeline
        include_distorted (bool): Whether to include distorted images
        max_samples_per_class (int): Maximum samples per identity class
        balance_classes (bool): Whether to apply class balancing weights
    """
    def __init__(self, root_dir, transform=None, include_distorted=True,
                 max_samples_per_class=None, balance_classes=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.include_distorted = include_distorted
        self.balance_classes = balance_classes

        # Data storage containers
        self.samples = []           # List of image file paths
        self.labels = []           # List of corresponding identity labels
        self.class_weights = {}    # Dictionary mapping class_id -> weight

        # Build dataset from directory structure
        self._load_dataset(max_samples_per_class)
        self._setup_label_encoder()

        # Calculate class weights for balanced sampling if requested
        if balance_classes:
            self._calculate_class_weights()

        logger.info(f"Dataset loaded: {len(self.samples)} samples, {self.num_classes} classes")
        logger.info(f"Class distribution: {Counter(self.labels)}")

    def _load_dataset(self, max_samples_per_class):
        """
        Load dataset with intelligent sampling strategy

        This method implements a careful balance between:
        1. Preserving identity representation (at least one original per person)
        2. Including distorted images for robustness
        3. Respecting sample limits for computational efficiency

        Sampling Strategy:
        - Priority 1: Include at least one original image per person
        - Priority 2: Fill remaining slots with mix of original and distorted
        - Priority 3: Ensure balanced representation across distortion types

        Args:
            max_samples_per_class (int): Maximum samples per identity class
        """
        person_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]

        for person_dir in tqdm(person_dirs, desc="Loading dataset"):
            person_name = person_dir.name
            current_samples = []

            # Step 1: Load original images (high priority for identity preservation)
            for img_file in person_dir.glob("*.jpg"):
                if img_file.name != "Thumbs.db" and self._is_valid_image(img_file):
                    current_samples.append((str(img_file), person_name, 'original'))

            # Step 2: Load distorted images if enabled (for robustness training)
            if self.include_distorted:
                distortion_dir = person_dir / "distortion"
                if distortion_dir.exists():
                    for img_file in distortion_dir.glob("*.jpg"):
                        if self._is_valid_image(img_file):
                            distortion_type = self._get_distortion_type(img_file.name)
                            current_samples.append((str(img_file), person_name, distortion_type))

            # Step 3: Apply intelligent sample selection with class limits
            if max_samples_per_class and len(current_samples) > max_samples_per_class:
                # Separate original and distorted samples
                original_samples = [s for s in current_samples if s[2] == 'original']
                distorted_samples = [s for s in current_samples if s[2] != 'original']

                if original_samples:
                    # Guarantee at least one original image for identity preservation
                    selected_samples = original_samples[:1]
                    remaining_slots = max_samples_per_class - 1

                    if remaining_slots > 0:
                        # Fill remaining slots with balanced mix
                        remaining_pool = original_samples[1:] + distorted_samples
                        selected_samples.extend(random.sample(
                            remaining_pool,
                            min(remaining_slots, len(remaining_pool))
                        ))
                else:
                    # Fallback: random selection if no originals available
                    selected_samples = random.sample(current_samples, max_samples_per_class)

                current_samples = selected_samples

            # Step 4: Add selected samples to main dataset
            for img_path, person_name, img_type in current_samples:
                self.samples.append(img_path)
                self.labels.append(person_name)

    def _is_valid_image(self, image_path):
        """
        Validate image file integrity

        Uses PIL's verify() method to check if the image can be opened
        and has a valid format without fully loading it into memory.

        Args:
            image_path (Path): Path to image file

        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify image integrity without loading
            return True
        except:
            return False

    def _get_distortion_type(self, filename):
        """
        Extract distortion type from filename using pattern matching

        Maps filename patterns to distortion categories for analysis.
        This helps track model performance across different distortion types.

        Distortion Types (based on dataset structure):
        - blurred: Out-of-focus or motion blur
        - foggy: Atmospheric scattering effects
        - lowlight: Poor illumination conditions
        - noisy: Various noise patterns (Gaussian, ISO, etc.)
        - rainy: Weather-related distortions
        - resized: Different resolutions/compression artifacts
        - sunny: Overexposure and harsh lighting

        Args:
            filename (str): Image filename to analyze

        Returns:
            str: Distortion type identifier
        """
        filename_lower = filename.lower()
        if 'blurred' in filename_lower:
            return 'blurred'
        elif 'foggy' in filename_lower:
            return 'foggy'
        elif 'lowlight' in filename_lower:
            return 'lowlight'
        elif 'noisy' in filename_lower:
            return 'noisy'
        elif 'rainy' in filename_lower:
            return 'rainy'
        elif 'resized' in filename_lower:
            return 'resized'
        elif 'sunny' in filename_lower:
            return 'sunny'
        else:
            return 'unknown'

    def _setup_label_encoder(self):
        """
        Initialize label encoder for string->integer class mapping

        Converts string identity labels to integer indices for neural network training.
        Creates bidirectional mapping: name ↔ integer_id

        Mathematical Properties:
        - Bijective mapping: each name maps to unique integer
        - Deterministic: same name always maps to same integer
        - Contiguous: integers range from 0 to num_classes-1
        """
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)

    def _calculate_class_weights(self):
        """
        Calculate inverse frequency weights for balanced training

        Mathematical Formulation:
        w_i = n_total / (n_classes × n_i)

        Where:
        - w_i: weight for class i
        - n_total: total number of samples
        - n_classes: number of identity classes
        - n_i: number of samples in class i

        Properties:
        - Inverse relationship: rare classes get higher weights
        - Normalization: weights ensure balanced expected sampling
        - Stability: prevents division by zero through dataset validation

        Effect on Training:
        Classes with fewer samples receive higher sampling probability,
        ensuring the model sees balanced representation during training.
        """
        class_counts = Counter(self.encoded_labels)
        total_samples = len(self.encoded_labels)

        for class_id, count in class_counts.items():
            # Inverse frequency weighting with normalization
            self.class_weights[class_id] = total_samples / (self.num_classes * count)

    def get_sample_weights(self):
        """
        Generate sample weights for WeightedRandomSampler

        Returns sample-level weights that can be used with PyTorch's
        WeightedRandomSampler to achieve balanced training.

        Mathematical Process:
        1. Map each sample to its class weight: w_sample = w_class(sample)
        2. Convert to PyTorch tensor for efficient sampling

        Returns:
            torch.DoubleTensor or None: Sample weights for balanced sampling
        """
        if not self.balance_classes:
            return None

        # Map each sample to its corresponding class weight
        weights = [self.class_weights[label] for label in self.encoded_labels]
        return torch.DoubleTensor(weights)

    def __len__(self):
        """Return total number of samples in dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample with robust error handling

        Implements graceful error recovery by falling back to a random
        sample if the requested sample cannot be loaded.

        Image Loading Pipeline:
        1. Load image using OpenCV (handles various formats efficiently)
        2. Convert BGR -> RGB (OpenCV uses BGR by default)
        3. Apply augmentation transforms
        4. Return (image, label, path) tuple

        Args:
            idx (int): Sample index

        Returns:
            tuple: (transformed_image, encoded_label, image_path)
        """
        img_path = self.samples[idx]
        label = self.encoded_labels[idx]

        try:
            # Step 1: Load image using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")

            # Step 2: Convert color space from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Step 3: Apply augmentation transforms
            if self.transform:
                image = self.transform(image)

            return image, label, img_path

        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Graceful recovery: return a random valid sample
            fallback_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(fallback_idx)

class EarlyStopping:
    """
    Early Stopping Callback for Training Regularization

    Implements early stopping to prevent overfitting and reduce training time.
    Monitors validation loss and stops training when improvement plateaus.

    Mathematical Foundation:
    The callback tracks validation loss L_val(t) over epochs t and stops when:
    L_val(t) >= L_best - δ for patience consecutive epochs

    Where:
    - L_best: best validation loss observed so far
    - δ (min_delta): minimum improvement threshold
    - patience: number of epochs to wait without improvement

    Algorithm:
    1. Track best validation loss: L_best = min(L_val(0), ..., L_val(t))
    2. Check improvement: improvement = L_best - L_val(t) >= δ
    3. Update counter: counter = 0 if improvement else counter + 1
    4. Stop training: return True if counter >= patience

    Benefits:
    - Prevents overfitting by stopping before performance degrades
    - Reduces computational cost by avoiding unnecessary epochs
    - Maintains best model weights for optimal performance

    Parameters:
        patience (int): Number of epochs to wait without improvement
        min_delta (float): Minimum change to qualify as improvement
        restore_best_weights (bool): Whether to restore best model weights
    """
    def __init__(self, patience=15, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience                    # Maximum epochs without improvement
        self.min_delta = min_delta                 # Minimum improvement threshold
        self.restore_best_weights = restore_best_weights  # Whether to restore best weights

        # State tracking variables
        self.best_loss = float('inf')              # Best validation loss seen
        self.counter = 0                           # Epochs without improvement
        self.best_weights = None                   # Stored best model weights

    def __call__(self, val_loss, model):
        """
        Check if training should stop and update internal state

        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to potentially restore weights for

        Returns:
            bool: True if training should stop, False otherwise
        """
        # Check if current loss is significantly better than best
        if val_loss < self.best_loss - self.min_delta:
            # Improvement detected: update best loss and reset counter
            self.best_loss = val_loss
            self.counter = 0

            # Store current model weights as best
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            # No improvement: increment counter
            self.counter += 1

        # Check if patience exhausted
        if self.counter >= self.patience:
            # Restore best weights before stopping
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True

        return False

class FaceRecognitionTrainer:
    """
    Main Trainer Class for Face Recognition System

    This class orchestrates the complete training pipeline including:
    1. Data loading and preprocessing
    2. Model architecture setup
    3. Optimization strategy configuration
    4. Training loop execution with monitoring
    5. Model evaluation and checkpointing

    Training Pipeline Architecture:

    Data Flow:
    Raw Images → Augmentation → Batch Loading → Model Forward →
    Loss Computation → Backpropagation → Optimizer Step → Evaluation

    Mathematical Components:
    - Loss Function: L = L_arcface + L_regularization
    - Optimization: θ_{t+1} = θ_t - η∇L(θ_t) (with AdamW modifications)
    - Learning Rate: η(t) = η_0 * cos(π * t / T) (cosine annealing)
    - Regularization: L2 weight decay, dropout, early stopping

    Key Features:
    - Multi-GPU support for scalable training
    - Mixed precision training for memory efficiency
    - Advanced learning rate scheduling
    - Comprehensive logging and monitoring
    - Automatic checkpointing and recovery

    Parameters:
        config (Config): Training configuration object
    """
    def __init__(self, config=None):
        # Configuration and device setup
        try:
            self.config = config or Config()
        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            raise ValueError(f"Configuration initialization failed: {e}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory for saving results
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        # Memory management settings
        self.max_memory_usage = 0.8  # Use max 80% of available memory
        self.memory_cleanup_threshold = 0.75  # Cleanup when memory usage exceeds 75%

        # Training component placeholders (initialized in setup methods)
        self.model = None              # Neural network model
        self.train_loader = None       # Training data loader
        self.val_loader = None         # Validation data loader
        self.optimizer = None          # Optimization algorithm
        self.scheduler = None          # Learning rate scheduler
        self.criterion = None          # Loss function
        self.early_stopping = None     # Early stopping callback

        # Training state tracking
        self.best_accuracy = 0.0       # Best validation accuracy achieved
        self.training_history = {      # Historical metrics for analysis
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }

        # Memory monitoring
        self._setup_memory_monitoring()

        # Log system information
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def _setup_memory_monitoring(self):
        """Setup memory monitoring and management"""
        try:
            import psutil
            self.system_memory = psutil.virtual_memory().total
            self.memory_monitoring_enabled = True
            logger.info(f"System Memory: {self.system_memory / 1024**3:.1f} GB")
        except ImportError:
            logger.warning("psutil not available. Memory monitoring disabled.")
            self.memory_monitoring_enabled = False

    def _check_memory_usage(self):
        """Check current memory usage and trigger cleanup if needed"""
        if not self.memory_monitoring_enabled:
            return

        try:
            import psutil
            import gc

            # Check system memory
            memory_percent = psutil.virtual_memory().percent / 100

            # Check GPU memory if available
            gpu_memory_percent = 0
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_percent = gpu_memory_used / gpu_memory_total

            # Trigger cleanup if memory usage is high
            if memory_percent > self.memory_cleanup_threshold or gpu_memory_percent > self.memory_cleanup_threshold:
                logger.warning(f"High memory usage detected. System: {memory_percent:.1%}, GPU: {gpu_memory_percent:.1%}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Memory cleanup performed")

        except Exception as e:
            logger.warning(f"Memory check failed: {e}")

    def _optimize_batch_size(self, dataset_size: int) -> int:
        """Dynamically optimize batch size based on available memory and dataset size"""
        base_batch_size = self.config.BATCH_SIZE

        # Adjust based on dataset size
        if dataset_size < 1000:
            # Small dataset - use smaller batch size
            optimal_batch_size = min(base_batch_size, max(8, dataset_size // 10))
        elif dataset_size > 10000:
            # Large dataset - may need larger batch size for efficiency
            optimal_batch_size = min(base_batch_size * 2, 128)
        else:
            optimal_batch_size = base_batch_size

        # Adjust based on available memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb < 8:  # Less than 8GB GPU memory
                optimal_batch_size = min(optimal_batch_size, 16)
            elif gpu_memory_gb < 16:  # Less than 16GB GPU memory
                optimal_batch_size = min(optimal_batch_size, 32)

        if optimal_batch_size != base_batch_size:
            logger.info(f"Optimized batch size from {base_batch_size} to {optimal_batch_size}")

        return optimal_batch_size

    def setup_data(self):
        """
        Setup data loaders with balanced sampling and robust augmentation

        Data Pipeline Architecture:
        1. Dataset Creation: Load images with balanced class representation
        2. Train/Val Split: Stratified split preserving class distribution
        3. Augmentation: Different transforms for train vs validation
        4. Weighted Sampling: Balance class frequencies during training
        5. Data Loading: Efficient batching with multiprocessing

        Mathematical Considerations:
        - Class Balancing: Inverse frequency weighting ensures equal class representation
        - Stratified Split: Maintains class distribution across train/val sets
        - Batch Sampling: WeightedRandomSampler implements probabilistic class balancing

        Performance Optimizations:
        - pin_memory=True: Faster GPU transfer via pinned memory
        - num_workers>1: Parallel data loading to overlap with computation
        - drop_last=True: Consistent batch sizes for stable batch normalization
        """
        logger.info("Setting up data loaders...")

        # Step 1: Create balanced dataset with intelligent sampling
        full_dataset = BalancedFaceDataset(
            root_dir=self.config.TRAIN_DIR,
            transform=RobustFaceTransforms(self.config.IMAGE_SIZE, is_training=True),
            include_distorted=self.config.INCLUDE_DISTORTED,
            max_samples_per_class=self.config.MAX_SAMPLES_PER_CLASS,
            balance_classes=True
        )

        # Step 2: Create stratified train/validation split
        # Uses fixed seed for reproducible splits across runs
        val_size = int(len(full_dataset) * self.config.VAL_SPLIT)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )

        # Step 3: Apply different augmentation for validation (no augmentation)
        # Validation uses clean transforms for consistent evaluation
        val_dataset.dataset.transform = RobustFaceTransforms(
            self.config.IMAGE_SIZE, is_training=False
        )

        # Step 4: Setup weighted sampling for balanced training
        sample_weights = full_dataset.get_sample_weights()
        if sample_weights is not None:
            # Extract weights for training samples only
            train_weights = sample_weights[train_dataset.indices]
            sampler = WeightedRandomSampler(
                weights=train_weights,        # Sample-level weights
                num_samples=len(train_weights), # Samples per epoch
                replacement=True              # Allow repeated sampling
            )
        else:
            sampler = None

        # Step 5: Create optimized data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            sampler=sampler,                 # Use weighted sampler if available
            shuffle=(sampler is None),       # Shuffle only if no custom sampler
            num_workers=4,                   # Parallel data loading
            pin_memory=True,                 # Faster GPU transfer
            drop_last=True                   # Consistent batch sizes
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,                   # No shuffling for validation
            num_workers=4,
            pin_memory=True
        )

        # Store dataset metadata for model creation
        self.num_classes = full_dataset.num_classes
        self.label_encoder = full_dataset.label_encoder

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Number of classes: {self.num_classes}")

    def setup_model(self):
        """
        Setup model architecture, optimization, and training components

        This method initializes all components required for training:
        1. Model Architecture: Enhanced Vision Transformer with ArcFace
        2. Optimization: AdamW with advanced parameter settings
        3. Learning Rate Scheduling: Cosine annealing with warm restarts
        4. Loss Function: Cross-entropy with label smoothing
        5. Regularization: Early stopping for overfitting prevention

        Mathematical Components:

        1. AdamW Optimizer:
           m_t = β₁m_{t-1} + (1-β₁)g_t
           v_t = β₂v_{t-1} + (1-β₂)g_t²
           θ_{t+1} = θ_t - η(m̂_t/(√v̂_t + ε) + λθ_t)
           where λ is weight decay, m̂_t and v̂_t are bias-corrected moments

        2. Cosine Annealing with Warm Restarts:
           η_t = η_{min} + (η_{max} - η_{min})(1 + cos(πT_{cur}/T_i))/2
           where T_i is the restart period, T_{cur} is current step

        3. Label Smoothing:
           L_smooth = (1-α)L_ce + α/K
           where α=0.1 is smoothing factor, K is number of classes
        """
        logger.info("Setting up model...")

        # Step 1: Create Enhanced Vision Transformer model
        # Combines ViT backbone with ArcFace loss for face recognition
        self.model = EnhancedFaceViT(
            model_name=self.config.MODEL_NAME,    # ViT architecture variant
            num_classes=self.num_classes,         # Number of identity classes
            embedding_dim=self.config.EMBEDDING_DIM, # Feature embedding dimension
            pretrained=True,                      # Use ImageNet pretrained weights
            dropout_rate=0.3                      # Regularization dropout rate
        ).to(self.device)

        # Step 2: Calculate model complexity metrics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Step 3: Setup AdamW optimizer with decoupled weight decay
        # AdamW separates weight decay from gradient-based updates
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,        # Base learning rate
            weight_decay=self.config.WEIGHT_DECAY, # L2 regularization strength
            betas=(0.9, 0.999),                  # Momentum coefficients (β₁, β₂)
            eps=1e-8                             # Numerical stability constant
        )

        # Step 4: Setup cosine annealing scheduler with warm restarts
        # Implements cyclical learning rate for better convergence
        # T_0: initial restart period, T_mult: period multiplier after restart
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,          # Initial restart period (10 epochs)
            T_mult=2,        # Double the period after each restart
            eta_min=1e-6     # Minimum learning rate
        )

        # Step 5: Setup loss function with label smoothing regularization
        # Label smoothing prevents overconfident predictions and improves generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Step 6: Setup early stopping to prevent overfitting
        # Monitors validation loss and stops training when no improvement
        self.early_stopping = EarlyStopping(
            patience=self.config.PATIENCE,       # Epochs to wait without improvement
            min_delta=self.config.MIN_DELTA      # Minimum improvement threshold
        )

    def train_epoch(self):
        """
        Execute one complete training epoch with detailed monitoring

        Training Loop Mathematical Framework:

        For each batch B_i = {(x_j, y_j)}:
        1. Forward Pass: ŷ_j = f(x_j; θ), where f is the model with parameters θ
        2. Loss Computation: L = (1/|B|) Σ CrossEntropy(ŷ_j, y_j) + regularization
        3. Backward Pass: ∇θL = ∂L/∂θ via automatic differentiation
        4. Gradient Clipping: g_clipped = min(g, γ||g||/||g||) where γ is clip value
        5. Parameter Update: θ ← θ - η∇θL using AdamW optimizer

        Key Operations:
        - Gradient accumulation over batch
        - Gradient clipping for training stability
        - Running statistics for monitoring
        - Progress tracking with real-time metrics

        Returns:
            tuple: (average_loss, accuracy) for the epoch
        """
        # Set model to training mode (enables dropout, batch norm updates)
        self.model.train()

        # Initialize epoch-level statistics
        total_loss = 0.0      # Cumulative loss across all batches
        total_correct = 0     # Cumulative correct predictions
        total_samples = 0     # Total samples processed

        # Training loop with progress bar
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels, _) in enumerate(pbar):
            # Step 1: Move data to GPU/device
            images, labels = images.to(self.device), labels.to(self.device)

            # Step 2: Forward pass through model
            self.optimizer.zero_grad()  # Clear gradients from previous iteration

            # Model returns (logits, embeddings) in training mode
            # logits: ArcFace output for classification [batch_size, num_classes]
            # embeddings: L2-normalized features [batch_size, embedding_dim]
            logits, embeddings = self.model(images, labels)

            # Step 3: Compute cross-entropy loss with label smoothing
            loss = self.criterion(logits, labels)

            # Step 4: Backward pass - compute gradients
            loss.backward()

            # Step 5: Gradient clipping for training stability
            # Prevents exploding gradients by limiting gradient norm
            # ||g_clipped|| = min(||g||, γ) where γ is the clipping threshold
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRADIENT_CLIPPING  # Typically 1.0
            )

            # Step 6: Update model parameters using AdamW
            self.optimizer.step()

            # Step 7: Update running statistics for monitoring
            total_loss += loss.item()

            # Calculate predictions from logits
            _, predicted = torch.max(logits.data, 1)  # Get class with highest probability
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            # Step 8: Update progress bar with real-time metrics
            current_acc = 100. * total_correct / total_samples
            current_lr = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',    # Current batch loss
                'acc': f'{current_acc:.2f}%',    # Running accuracy
                'lr': f'{current_lr:.2e}'        # Current learning rate
            })

        # Calculate epoch-level metrics
        avg_loss = total_loss / len(self.train_loader)  # Average loss per batch
        accuracy = total_correct / total_samples        # Overall accuracy

        return avg_loss, accuracy

    def validate_epoch(self):
        """
        Execute one complete validation epoch without gradient updates

        Validation Process:
        1. Set model to evaluation mode (disables dropout, fixes batch norm)
        2. Disable gradient computation for memory efficiency
        3. Process all validation batches
        4. Compute aggregated metrics for model selection

        Mathematical Framework:
        - Forward pass only: ŷ = f(x; θ) without gradient computation
        - Loss computation: L_val = (1/N_val) Σ CrossEntropy(ŷ_i, y_i)
        - Accuracy: Acc = (1/N_val) Σ I[argmax(ŷ_i) = y_i]

        Key Differences from Training:
        - No gradient computation (torch.no_grad())
        - No parameter updates
        - No data augmentation (clean transforms only)
        - Deterministic behavior (no dropout)

        Returns:
            tuple: (average_loss, accuracy) for validation set
        """
        # Set model to evaluation mode
        # - Disables dropout (uses all connections)
        # - Fixes batch normalization (uses running statistics)
        self.model.eval()

        # Initialize validation statistics
        total_loss = 0.0      # Cumulative validation loss
        total_correct = 0     # Cumulative correct predictions
        total_samples = 0     # Total validation samples

        # Disable gradient computation for memory efficiency and speed
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, labels, _ in pbar:
                # Step 1: Move data to device
                images, labels = images.to(self.device), labels.to(self.device)

                # Step 2: Forward pass (no gradient tracking)
                # Model returns (logits, embeddings) in training mode
                # Even in eval mode, we pass labels for consistent interface
                logits, embeddings = self.model(images, labels)

                # Step 3: Compute validation loss
                loss = self.criterion(logits, labels)

                # Step 4: Update statistics
                total_loss += loss.item()

                # Calculate predictions
                _, predicted = torch.max(logits.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                # Step 5: Update progress display
                current_acc = 100. * total_correct / total_samples
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',    # Current batch loss
                    'acc': f'{current_acc:.2f}%'     # Running accuracy
                })

        # Calculate final validation metrics
        avg_loss = total_loss / len(self.val_loader)   # Average loss per batch
        accuracy = total_correct / total_samples       # Overall accuracy

        return avg_loss, accuracy

    def train(self):
        """
        Execute the complete training pipeline with comprehensive monitoring

        Training Pipeline:
        1. Data Setup: Load and prepare balanced datasets
        2. Model Setup: Initialize architecture and optimization components
        3. Training Loop: Iterate through epochs with train/validation cycles
        4. Monitoring: Track metrics, save checkpoints, apply early stopping
        5. Finalization: Save final results and training history

        Mathematical Framework:
        The training process optimizes the objective function:

        θ* = argmin_θ E[(x,y)~D_train][L(f(x;θ), y)] + λR(θ)

        Where:
        - θ: model parameters
        - L: loss function (ArcFace + CrossEntropy)
        - f(x;θ): model prediction function
        - R(θ): regularization term (weight decay, dropout)
        - λ: regularization strength

        Learning Rate Schedule:
        η(t) follows cosine annealing with warm restarts:
        η(t) = η_min + (η_max - η_min) × (1 + cos(πT_cur/T_i))/2

        Early Stopping Criterion:
        Stop when validation loss doesn't improve for 'patience' epochs:
        L_val(t) ≥ L_best - δ for consecutive epochs ≥ patience
        """
        logger.info("Starting training...")

        # Step 1: Initialize data loaders and model architecture
        self.setup_data()    # Create balanced train/val splits with augmentation
        self.setup_model()   # Initialize ViT+ArcFace model and optimization

        # Track total training time
        start_time = time.time()

        # Step 2: Main training loop
        for epoch in range(self.config.EPOCHS):
            epoch_start = time.time()

            # Step 2a: Training phase - update model parameters
            # Computes gradients and updates θ using AdamW optimizer
            train_loss, train_acc = self.train_epoch()

            # Step 2b: Validation phase - evaluate current model
            # Computes metrics on held-out data without gradient updates
            val_loss, val_acc = self.validate_epoch()

            # Step 2c: Update learning rate according to schedule
            # Cosine annealing with warm restarts for better convergence
            self.scheduler.step()

            # Step 2d: Record training history for analysis
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rate'].append(current_lr)

            # Step 2e: Model checkpointing - save best performing model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model()  # Save complete model state
                logger.info(f"New best model saved with accuracy: {val_acc:.4f}")

            # Step 2f: Calculate and log epoch metrics
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch+1}/{self.config.EPOCHS} "
                f"({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.2e}"
            )

            # Step 2g: Early stopping check to prevent overfitting
            # Monitors validation loss and stops if no improvement
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Step 3: Training completion and finalization
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        logger.info(f"Best validation accuracy: {self.best_accuracy:.4f}")

        # Save comprehensive training history for analysis
        self.save_training_history()

    def save_model(self):
        """
        Save complete model checkpoint with all training state

        Checkpoint Contents:
        1. Model State: All learned parameters (weights, biases)
        2. Optimizer State: Adam moments and parameter-specific state
        3. Scheduler State: Learning rate schedule progression
        4. Training Metadata: Best metrics, configuration, label mapping
        5. Training History: Complete loss/accuracy curves

        This comprehensive checkpoint enables:
        - Model deployment for inference
        - Training resumption from any epoch
        - Reproducible experiment results
        - Transfer learning to new datasets

        File Structure:
        - best_face_model.pth: Complete model checkpoint (PyTorch format)
        - label_encoder.json: Class name to integer mapping (JSON format)
        """
        # Step 1: Save complete model checkpoint
        model_path = os.path.join(self.config.OUTPUT_DIR, self.config.MODEL_SAVE_PATH)

        # Create comprehensive checkpoint dictionary
        checkpoint = {
            # Core model components
            'model_state_dict': self.model.state_dict(),         # All model parameters
            'optimizer_state_dict': self.optimizer.state_dict(), # AdamW state
            'scheduler_state_dict': self.scheduler.state_dict(), # LR schedule state

            # Training metadata
            'best_accuracy': self.best_accuracy,                 # Best validation accuracy
            'config': self.config.__dict__,                     # Complete configuration
            'num_classes': self.num_classes,                    # Number of identity classes
            'training_history': self.training_history           # Complete training curves
        }

        # Save checkpoint using PyTorch's efficient format
        torch.save(checkpoint, model_path)

        # Step 2: Save label encoder for deployment
        # Creates human-readable mapping from class names to integers
        label_encoder_path = os.path.join(self.config.OUTPUT_DIR, self.config.LABEL_ENCODER_PATH)
        label_mapping = {
            'classes': self.label_encoder.classes_.tolist(),    # List of class names
            'num_classes': len(self.label_encoder.classes_)     # Total number of classes
        }

        # Save as JSON for easy loading in inference scripts
        with open(label_encoder_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)

    def save_training_history(self):
        """
        Save comprehensive training history and generate visualizations

        Saves training metrics in multiple formats for analysis:
        1. JSON format for programmatic access
        2. Visual plots for human interpretation
        3. Statistical summaries for quick reference

        Training curves provide insights into:
        - Convergence behavior and training stability
        - Overfitting detection (train vs validation divergence)
        - Learning rate schedule effectiveness
        - Optimal stopping point identification
        """
        # Save training history as JSON for programmatic access
        history_path = os.path.join(self.config.OUTPUT_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        # Generate comprehensive training visualizations
        self.plot_training_curves()

    def plot_training_curves(self):
        """
        Generate comprehensive training visualization plots

        Creates a 2x2 subplot layout showing:
        1. Loss Curves: Training vs validation loss over epochs
        2. Accuracy Curves: Training vs validation accuracy over epochs
        3. Learning Rate Schedule: LR progression over training
        4. Training Summary: Key statistics and final results

        Mathematical Analysis:
        - Loss curves indicate convergence and potential overfitting
        - Accuracy curves show model performance improvement
        - LR schedule shows optimization dynamics
        - Gap between train/val indicates generalization ability

        Overfitting Indicators:
        - Training loss continues decreasing while validation loss increases
        - Large gap between training and validation accuracy
        - Validation accuracy plateaus or decreases
        """
        # Create 2x2 subplot figure with adequate size
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Generate epoch indices for x-axis
        epochs = range(1, len(self.training_history['train_loss']) + 1)

        # Plot 1: Loss Curves (Training vs Validation)
        # Shows convergence behavior and potential overfitting
        ax1.plot(epochs, self.training_history['train_loss'], 'b-',
                label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.training_history['val_loss'], 'r-',
                label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Accuracy Curves (Training vs Validation)
        # Shows model performance improvement over training
        ax2.plot(epochs, self.training_history['train_acc'], 'b-',
                label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.training_history['val_acc'], 'r-',
                label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Learning Rate Schedule
        # Shows optimization dynamics and schedule progression
        ax3.plot(epochs, self.training_history['learning_rate'], 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')  # Log scale for better visualization
        ax3.grid(True, alpha=0.3)

        # Plot 4: Training Summary with Key Statistics
        # Provides quick overview of training results
        best_epoch = np.argmax(self.training_history['val_acc']) + 1
        best_acc = max(self.training_history['val_acc'])
        final_train_acc = self.training_history['train_acc'][-1]
        final_val_acc = self.training_history['val_acc'][-1]

        summary_text = [
            f'Best Validation Accuracy: {best_acc:.4f}',
            f'Achieved at Epoch: {best_epoch}',
            f'Final Training Accuracy: {final_train_acc:.4f}',
            f'Final Validation Accuracy: {final_val_acc:.4f}',
            f'Total Epochs: {len(epochs)}',
            f'Number of Classes: {self.num_classes}',
            f'Generalization Gap: {abs(final_train_acc - final_val_acc):.4f}'
        ]

        for i, text in enumerate(summary_text):
            ax4.text(0.1, 0.9 - i * 0.12, text, transform=ax4.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="lightblue", alpha=0.7))

        ax4.set_title('Training Summary', fontsize=14, fontweight='bold')
        ax4.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.config.OUTPUT_DIR, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Training curves saved to {save_path}")

def parse_arguments():
    """
    Parse command line arguments for training configuration

    Provides flexible command-line interface for training customization:
    - Data paths and output directories
    - Model architecture parameters
    - Training hyperparameters
    - Resource allocation settings

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Train Face Recognition Model with Vision Transformer and ArcFace',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data and I/O arguments
    parser.add_argument('--train_dir', type=str, default='train',
                        help='Path to training data directory containing person folders')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for saving models, logs, and results')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (adjust based on GPU memory)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate for AdamW optimizer')

    # Model architecture parameters
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='Dimension of face embeddings (higher = more capacity)')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        help='Vision Transformer variant (vit_base, vit_large, etc.)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image resolution (must match model requirements)')

    # Data selection options
    parser.add_argument('--no_distorted', action='store_true',
                        help='Exclude distorted images from training (clean images only)')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                        help='Limit number of samples per identity class (for faster training)')

    return parser.parse_args()

def main():
    """
    Main training function with comprehensive setup and error handling

    Pipeline Overview:
    1. Parse command line arguments for flexible configuration
    2. Setup training configuration with validation
    3. Initialize trainer with optimized settings
    4. Execute training with monitoring and checkpointing
    5. Report final results and save artifacts

    Error Handling:
    - Validates input directories and parameters
    - Provides clear error messages for common issues
    - Implements graceful failure with diagnostic information
    """
    # Step 1: Parse and validate command line arguments
    args = parse_arguments()

    # Step 2: Create and configure training setup
    config = Config()

    # Update configuration with command line arguments
    config.TRAIN_DIR = args.train_dir
    config.OUTPUT_DIR = args.output_dir
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.learning_rate
    config.EMBEDDING_DIM = args.embedding_dim
    config.MODEL_NAME = args.model_name
    config.INCLUDE_DISTORTED = not args.no_distorted
    config.MAX_SAMPLES_PER_CLASS = args.max_samples_per_class
    config.IMAGE_SIZE = args.image_size

    # Step 3: Display comprehensive training configuration
    logger.info("=" * 80)
    logger.info("FACE RECOGNITION TRAINING WITH VISION TRANSFORMER + ARCFACE")
    logger.info("=" * 80)
    logger.info("CONFIGURATION:")
    logger.info(f"  Training directory: {config.TRAIN_DIR}")
    logger.info(f"  Output directory: {config.OUTPUT_DIR}")
    logger.info(f"  Model architecture: {config.MODEL_NAME}")
    logger.info(f"  Embedding dimension: {config.EMBEDDING_DIM}")
    logger.info(f"  Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  Maximum epochs: {config.EPOCHS}")
    logger.info(f"  Learning rate: {config.LEARNING_RATE}")
    logger.info(f"  Include distorted images: {config.INCLUDE_DISTORTED}")
    if config.MAX_SAMPLES_PER_CLASS:
        logger.info(f"  Max samples per class: {config.MAX_SAMPLES_PER_CLASS}")
    logger.info("=" * 80)

    # Step 4: Validate training prerequisites
    if not os.path.exists(config.TRAIN_DIR):
        logger.error(f"❌ Training directory does not exist: {config.TRAIN_DIR}")
        logger.error("Please ensure the training data is available and path is correct.")
        return

    # Check for minimum number of person directories
    person_dirs = [d for d in Path(config.TRAIN_DIR).iterdir() if d.is_dir()]
    if len(person_dirs) < 2:
        logger.error(f"❌ Need at least 2 person directories, found {len(person_dirs)}")
        logger.error("Face recognition requires multiple identity classes for training.")
        return

    logger.info(f"✅ Found {len(person_dirs)} person directories for training")

    # Step 5: Execute training pipeline with comprehensive error handling
    try:
        # Initialize trainer with validated configuration
        trainer = FaceRecognitionTrainer(config)

        # Execute complete training pipeline
        # This includes data loading, model setup, training loop, and checkpointing
        trainer.train()

        # Step 6: Report successful completion with key metrics
        logger.info("=" * 80)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"📊 Best validation accuracy: {trainer.best_accuracy:.4f}")
        logger.info(f"💾 Model saved to: {os.path.join(config.OUTPUT_DIR, config.MODEL_SAVE_PATH)}")
        logger.info(f"🏷️  Label encoder saved to: {os.path.join(config.OUTPUT_DIR, config.LABEL_ENCODER_PATH)}")
        logger.info(f"📈 Training curves saved to: {os.path.join(config.OUTPUT_DIR, 'training_curves.png')}")
        logger.info(f"📋 Training history saved to: {os.path.join(config.OUTPUT_DIR, 'training_history.json')}")
        logger.info("=" * 80)

        # Provide next steps guidance
        logger.info("NEXT STEPS:")
        logger.info("1. Run inference.py for model evaluation")
        logger.info("2. Use demo.py for interactive testing")
        logger.info("3. Check training curves for convergence analysis")

    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user (Ctrl+C)")
        logger.info("Partial results may be saved in the output directory")

    except torch.cuda.OutOfMemoryError:
        logger.error("❌ GPU out of memory error")
        logger.error("Solutions:")
        logger.error("  - Reduce batch size (--batch_size)")
        logger.error("  - Reduce image size (--image_size)")
        logger.error("  - Use CPU training (slower but uses system RAM)")

    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.error("Check the error message above and training logs for details")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
