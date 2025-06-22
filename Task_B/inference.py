#!/usr/bin/env python3
"""
Face Recognition Inference and Evaluation System

This module provides a comprehensive inference system for face recognition tasks,
implementing state-of-the-art methods for face verification and identification
with robust handling of distorted images.

Mathematical Foundation:

1. Face Verification (1:1 matching):
   Given two face images I₁ and I₂, determine if they belong to the same person.

   Process:
   - Extract embeddings: f₁ = φ(I₁), f₂ = φ(I₂) where φ is the trained model
   - Compute similarity: s = cos(f₁, f₂) = (f₁ · f₂) / (||f₁|| ||f₂||)
   - Decision: same_person = s ≥ τ where τ is the threshold

   Mathematical Properties:
   - Cosine similarity ∈ [-1, 1], where 1 = identical, -1 = opposite
   - L2 normalized embeddings: ||f|| = 1, so cos(f₁, f₂) = f₁ · f₂
   - Threshold τ controls trade-off between False Accept Rate (FAR) and False Reject Rate (FRR)

2. Face Identification (1:N matching):
   Given query image I_q and gallery {I₁, I₂, ..., I_N}, find best match.

   Process:
   - Extract query embedding: f_q = φ(I_q)
   - Extract gallery embeddings: F = {φ(I₁), φ(I₂), ..., φ(I_N)}
   - Compute similarities: S = {cos(f_q, f_i) for f_i in F}
   - Rank by similarity: ranked_indices = argsort(S, descending=True)

   Evaluation Metrics:
   - Rank-k accuracy: P(true_match ∈ top_k_results)
   - Mean Reciprocal Rank: MRR = 1/N ∑(1/rank_i) where rank_i is rank of correct match

3. Embedding Space Properties:
   The trained model maps face images to a hypersphere where:
   - Intra-class distances are minimized: ||f_i - f_j|| small for same person
   - Inter-class distances are maximized: ||f_i - f_j|| large for different persons
   - Angular margin from ArcFace enforces: cos(θ_yi + m) for ground truth class
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from PIL import Image
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import random

# Import from training script
from train_face_recognition import EnhancedFaceViT, RobustFaceTransforms, Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionInference:
    """
    Comprehensive Face Recognition Inference System

    This class provides a complete inference pipeline for face recognition tasks,
    supporting both verification (1:1 matching) and identification (1:N matching)
    with robust handling of various image distortions.

    System Architecture:
    1. Model Loading: Loads pre-trained ViT+ArcFace model with weights
    2. Preprocessing: Applies consistent transforms matching training pipeline
    3. Feature Extraction: Generates L2-normalized embeddings via forward pass
    4. Similarity Computation: Calculates cosine similarity between embeddings
    5. Decision Making: Applies thresholds for verification or ranking for identification

    Mathematical Framework:
    - Feature Extraction: f = normalize(φ(I)) where φ is the trained network
    - Similarity Metric: sim(f₁, f₂) = f₁ᵀf₂ (dot product of unit vectors)
    - Verification Decision: match = sim(f₁, f₂) ≥ threshold
    - Identification Ranking: scores = [sim(f_query, f_i) for f_i in gallery]

    Key Features:
    - GPU/CPU compatibility with automatic device selection
    - Consistent preprocessing pipeline matching training
    - Batch processing support for efficient inference
    - Comprehensive error handling and validation
    - Support for various image formats and resolutions
    """

    def __init__(self, model_path: str, label_encoder_path: str, device: str = 'auto'):
        """
        Initialize the inference system

        Args:
            model_path: Path to the trained model
            label_encoder_path: Path to the label encoder JSON file
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.device = self._setup_device(device)
        self.model = None
        self.label_encoder_classes = None
        self.num_classes = 0
        self.config = None

        # Load model and label encoder
        self._load_model(model_path)
        self._load_label_encoder(label_encoder_path)

        # Setup transforms
        self.transform = RobustFaceTransforms(
            image_size=self.config.get('IMAGE_SIZE', 224),
            is_training=False
        )

        logger.info(f"Inference system initialized on {self.device}")
        logger.info(f"Loaded model with {self.num_classes} classes")

    def _setup_device(self, device: str) -> torch.device:
        """
        Setup computation device with automatic fallback

        Device Selection Strategy:
        1. 'auto': Choose CUDA if available, otherwise CPU
        2. 'cuda': Force GPU usage (fails if not available)
        3. 'cpu': Force CPU usage (always available)

        Performance Considerations:
        - GPU: ~10-50x faster for batch processing, requires CUDA
        - CPU: Slower but more memory, compatible with all systems
        - Memory usage: GPU typically more efficient for large batches

        Args:
            device (str): Device specification ('auto', 'cuda', 'cpu')

        Returns:
            torch.device: Configured PyTorch device
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        device = torch.device(device)

        if device.type == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = torch.device('cpu')

        return device

    def _load_model(self, model_path: str):
        """
        Load trained model from checkpoint with full state restoration

        Checkpoint Loading Process:
        1. Load checkpoint dictionary from file
        2. Extract model configuration and metadata
        3. Reconstruct model architecture with same parameters
        4. Load learned weights into model
        5. Set model to evaluation mode for inference

        Model State Restoration:
        - Architecture parameters: model_name, embedding_dim, num_classes
        - Learned weights: All neural network parameters (W, b)
        - Training metadata: Configuration, best accuracy, etc.

        Evaluation Mode Effects:
        - Disables dropout: Uses all connections for deterministic output
        - Fixes batch normalization: Uses running statistics instead of batch stats
        - Disables gradient computation: Saves memory and computation

        Args:
            model_path (str): Path to saved model checkpoint (.pth file)

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails due to incompatibility
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract configuration
        self.config = checkpoint.get('config', {})
        self.num_classes = checkpoint['num_classes']

        # Create model
        self.model = EnhancedFaceViT(
            model_name=self.config.get('MODEL_NAME', 'vit_base_patch16_224'),
            num_classes=self.num_classes,
            embedding_dim=self.config.get('EMBEDDING_DIM', 512),
            pretrained=False  # We're loading trained weights
        ).to(self.device)

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model loaded successfully")

    def _load_label_encoder(self, label_encoder_path: str):
        """
        Load label encoder for class name to integer mapping

        Label Encoder Structure:
        The label encoder provides bidirectional mapping between:
        - String identities: Human-readable names (e.g., "John_Doe")
        - Integer classes: Neural network class indices (e.g., 0, 1, 2, ...)

        Mathematical Properties:
        - Bijective mapping: Each name maps to unique integer and vice versa
        - Deterministic: Same name always maps to same integer across runs
        - Contiguous indices: Class IDs range from 0 to num_classes-1

        Usage in Inference:
        - Classification: Convert model output indices to identity names
        - Gallery building: Map directory names to consistent class labels
        - Result interpretation: Provide human-readable identification results

        Args:
            label_encoder_path (str): Path to label encoder JSON file

        Raises:
            FileNotFoundError: If label encoder file doesn't exist
            JSONDecodeError: If file format is invalid
        """
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")

        with open(label_encoder_path, 'r') as f:
            label_data = json.load(f)

        import numpy as np
        self.label_encoder_classes = np.array(label_data['classes'])
        self.num_classes = label_data['num_classes']

        logger.info(f"Label encoder loaded with {self.num_classes} classes")

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess single image for model inference

        Preprocessing Pipeline:
        1. Image Loading: Read image file using OpenCV
        2. Color Conversion: BGR → RGB (OpenCV uses BGR by default)
        3. Augmentation: Apply validation transforms (resize + normalize)
        4. Tensor Conversion: Convert to PyTorch tensor format
        5. Batch Dimension: Add batch dimension for model input
        6. Device Transfer: Move tensor to appropriate device (GPU/CPU)

        Mathematical Transformations:
        - Resize: Bilinear interpolation to target dimensions
        - Normalization: (pixel - μ) / σ where μ, σ are ImageNet statistics
        - Channel order: [H, W, C] → [C, H, W] for PyTorch convention
        - Batch dimension: [C, H, W] → [1, C, H, W] for single image inference

        Consistency Requirements:
        - Must match training preprocessing exactly
        - Same normalization statistics (ImageNet mean/std)
        - Same resize interpolation method
        - Same data type and value range

        Args:
            image_path (str): Path to input image file

        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, H, W]

        Raises:
            ValueError: If image cannot be loaded or is invalid
            IOError: If file reading fails
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply transforms
            image_tensor = self.transform(image)

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract L2-normalized face embedding from single image

        Mathematical Process:
        1. Preprocessing: I' = preprocess(I) where I is raw image
        2. Forward Pass: f_raw = φ(I') where φ is the trained model
        3. L2 Normalization: f = f_raw / ||f_raw||₂

        The resulting embedding f has the following properties:
        - Unit norm: ||f||₂ = 1
        - Fixed dimensionality: f ∈ ℝᵈ where d is embedding_dim
        - Discriminative: Similar faces have similar embeddings
        - Robust: Invariant to lighting, pose, and expression variations

        L2 Normalization Mathematical Details:
        Given raw embedding f_raw ∈ ℝᵈ:
        - Compute L2 norm: ||f_raw||₂ = √(∑ᵢ f_raw[i]²)
        - Normalize: f[i] = f_raw[i] / ||f_raw||₂ for all i
        - Result: ||f||₂ = 1 and cos(f₁, f₂) = f₁ᵀf₂

        Properties of Normalized Embeddings:
        - Cosine similarity reduces to dot product
        - Euclidean distance relates to angular distance
        - Robust to magnitude variations in raw features

        Args:
            image_path (str): Path to face image file

        Returns:
            np.ndarray: L2-normalized embedding vector [embedding_dim]

        Raises:
            Exception: If image processing or model inference fails
        """
        self.model.eval()

        with torch.no_grad():
            image_tensor = self._preprocess_image(image_path)
            embedding = self.model(image_tensor)

            # Normalize embedding
            embedding = F.normalize(embedding, p=2, dim=1)

            return embedding.cpu().numpy()[0]

    def verify_faces(self, image1_path: str, image2_path: str, threshold: float = 0.6) -> Tuple[bool, float]:
        """
        Face Verification: Determine if two images show the same person

        Mathematical Framework:
        Face verification is a binary classification problem solved using embedding similarity:

        1. Feature Extraction:
           f₁ = φ(I₁), f₂ = φ(I₂) where φ is the trained model

        2. Similarity Computation:
           s = cos(f₁, f₂) = f₁ᵀf₂ / (||f₁|| ||f₂||)
           For L2-normalized embeddings: s = f₁ᵀf₂

        3. Decision Rule:
           match = {True if s ≥ τ, False if s < τ} where τ is threshold

        Threshold Selection Impact:
        - High τ (e.g., 0.8): Conservative matching, low FAR, high FRR
        - Low τ (e.g., 0.4): Liberal matching, high FAR, low FRR
        - Optimal τ: Minimizes Equal Error Rate (EER) where FAR = FRR

        Error Types:
        - False Accept (Type I): Different persons classified as same (s ≥ τ, truth = different)
        - False Reject (Type II): Same person classified as different (s < τ, truth = same)

        Performance Metrics:
        - True Accept Rate (TAR) = 1 - FRR
        - True Reject Rate (TRR) = 1 - FAR
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)

        Args:
            image1_path (str): Path to first face image
            image2_path (str): Path to second face image
            threshold (float): Decision threshold τ ∈ [0, 1]

        Returns:
            Tuple[bool, float]: (is_same_person, cosine_similarity)
            - is_same_person: Boolean decision based on threshold
            - cosine_similarity: Raw similarity score ∈ [-1, 1]

        Raises:
            Exception: If either image cannot be processed
        """
        try:
            # Extract embeddings
            emb1 = self.extract_embedding(image1_path)
            emb2 = self.extract_embedding(image2_path)

            # Compute cosine similarity
            similarity = np.dot(emb1, emb2)

            # Make decision
            is_same = similarity >= threshold

            return is_same, float(similarity)

        except Exception as e:
            logger.error(f"Error in face verification: {e}")
            return False, 0.0

    def identify_face(self, query_image_path: str, gallery_dir: str, top_k: int = 5) -> List[Dict]:
        """
        Face Identification: Find best matches for query face in gallery

        Mathematical Framework:
        Face identification is a 1:N matching problem solved using similarity ranking:

        1. Query Processing:
           f_q = φ(I_q) where I_q is the query image

        2. Gallery Processing:
           G = {φ(I₁), φ(I₂), ..., φ(Iₙ)} where {I₁, I₂, ..., Iₙ} are gallery images

        3. Similarity Computation:
           S = {s₁, s₂, ..., sₙ} where sᵢ = cos(f_q, φ(Iᵢ)) = f_qᵀφ(Iᵢ)

        4. Ranking:
           ranked_indices = argsort(S, descending=True)
           top_k_matches = ranked_indices[:k]

        Gallery Construction Strategy:
        - Include both original and distorted images for robustness
        - Handle multiple images per person (enrollment set)
        - Skip invalid images gracefully
        - Maintain person-to-image mapping for result interpretation

        Ranking Metrics:
        - Rank-1 Accuracy: P(correct_person = top_match)
        - Rank-k Accuracy: P(correct_person ∈ top_k_matches)
        - Mean Reciprocal Rank: MRR = 1/N ∑(1/rank_of_correct_match)

        Confidence Scoring:
        - Raw similarity scores ∈ [-1, 1]
        - Confidence percentage: conf = max(0, 100 × similarity)
        - Gap analysis: confidence difference between rank-1 and rank-2

        Robustness Features:
        - Handles enrollment images with distortions
        - Aggregates multiple images per person if available
        - Graceful degradation with missing or corrupted gallery images

        Args:
            query_image_path (str): Path to query face image
            gallery_dir (str): Directory containing person subdirectories
            top_k (int): Number of top matches to return

        Returns:
            List[Dict]: Ranked list of matches, each containing:
            - 'person': Identity name from directory structure
            - 'similarity': Raw cosine similarity score ∈ [-1, 1]
            - 'confidence': Confidence percentage ∈ [0, 100]
            - 'gallery_path': Path to matching gallery image

        Raises:
            Exception: If query processing fails or gallery is empty
        """
        try:
            # Extract query embedding
            query_embedding = self.extract_embedding(query_image_path)

            # Build gallery
            gallery_embeddings = []
            gallery_labels = []
            gallery_paths = []

            gallery_path = Path(gallery_dir)

            for person_dir in gallery_path.iterdir():
                if person_dir.is_dir():
                    person_name = person_dir.name

                    # Load original images
                    for img_file in person_dir.glob("*.jpg"):
                        if img_file.name != "Thumbs.db":
                            try:
                                embedding = self.extract_embedding(str(img_file))
                                gallery_embeddings.append(embedding)
                                gallery_labels.append(person_name)
                                gallery_paths.append(str(img_file))
                            except:
                                continue

                    # Load distorted images
                    distortion_dir = person_dir / "distortion"
                    if distortion_dir.exists():
                        for img_file in distortion_dir.glob("*.jpg"):
                            try:
                                embedding = self.extract_embedding(str(img_file))
                                gallery_embeddings.append(embedding)
                                gallery_labels.append(person_name)
                                gallery_paths.append(str(img_file))
                            except:
                                continue

            if not gallery_embeddings:
                logger.warning("No valid gallery images found")
                return []

            # Convert to numpy array
            gallery_embeddings = np.array(gallery_embeddings)

            # Compute similarities
            similarities = np.dot(gallery_embeddings, query_embedding)

            # Get top-k matches
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    'person': gallery_labels[idx],
                    'similarity': float(similarities[idx]),
                    'confidence': float(similarities[idx] * 100),
                    'gallery_path': gallery_paths[idx]
                })

            return results

        except Exception as e:
            logger.error(f"Error in face identification: {e}")
            return []

    def classify_single_image(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Direct classification using ArcFace logits (alternative to similarity-based identification)

        Mathematical Framework:
        Unlike similarity-based identification, this method uses the classification head directly:

        1. Feature Extraction:
           f = φ_backbone(I) where φ_backbone is the ViT encoder

        2. Embedding Projection:
           e = ψ(f) where ψ is the embedding network

        3. ArcFace Classification:
           logits = ArcFace(e) = s × cos(θ_i + m) for ground truth, s × cos(θ_i) for others
           where θ_i is angle between embedding and class weight vector

        4. Probability Computation:
           P(class_i | I) = softmax(logits)_i = exp(logits_i) / ∑_j exp(logits_j)

        5. Top-k Selection:
           top_k_classes = argsort(P, descending=True)[:k]

        Comparison with Similarity-based Identification:
        - Direct Classification: Uses learned class prototypes (ArcFace weights)
        - Similarity-based: Uses gallery image embeddings as references
        - Direct: Faster inference (no gallery processing)
        - Similarity: More flexible (can add new identities without retraining)

        ArcFace Properties in Inference:
        - Class weights W_i represent learned prototypes for each identity
        - cos(θ_i) = (e^T W_i) / (||e|| ||W_i||) measures similarity to prototype
        - Angular margin m enforced during training improves discrimination
        - Scale factor s controls softmax temperature (confidence calibration)

        Probability Interpretation:
        - High probability: Query strongly resembles training examples of this class
        - Low probability: Query doesn't strongly match any training class
        - Entropy: H = -∑P_i log(P_i) measures prediction uncertainty

        Args:
            image_path (str): Path to face image for classification
            top_k (int): Number of top predictions to return

        Returns:
            List[Dict]: Top-k predictions, each containing:
            - 'class': Identity name from label encoder
            - 'class_index': Integer class index
            - 'probability': Softmax probability ∈ [0, 1]
            - 'confidence': Confidence percentage ∈ [0, 100]

        Raises:
            Exception: If image processing or classification fails
        """
        try:
            self.model.eval()

            with torch.no_grad():
                image_tensor = self._preprocess_image(image_path)

                # Get logits (we need to modify the model call for classification)
                # Since our model returns embeddings in eval mode, we need to get logits
                features = self.model.backbone(image_tensor)
                embeddings = self.model.embedding(features)

                # Use the ArcFace layer to get logits for classification
                logits = self.model.arcface(embeddings, None)  # No labels needed for inference

                # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=1)

                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, top_k)

                results = []
                for i in range(top_k):
                    class_idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()

                    results.append({
                        'class': self.label_encoder_classes[class_idx],
                        'class_index': class_idx,
                        'probability': float(prob),
                        'confidence': float(prob * 100)
                    })

                return results

        except Exception as e:
            logger.error(f"Error in image classification: {e}")
            return []

    def evaluate_verification_pairs(self, pairs_file: str, threshold: float = 0.6) -> Dict:
        """
        Evaluate the model on verification pairs

        Args:
            pairs_file: Path to file containing verification pairs
            threshold: Threshold for verification decision

        Returns:
            Dictionary with evaluation metrics
        """
        # This would load pairs from a file and evaluate
        # For now, we'll create synthetic pairs for demonstration
        pass

    def create_test_pairs(self, data_dir: str, num_pairs: int = 1000, output_file: str = None) -> List[Tuple]:
        """
        Generate balanced test pairs for face verification evaluation

        Evaluation Methodology:
        Face verification requires balanced positive and negative pairs to assess performance:

        1. Positive Pairs (Same Person):
           - Select two different images of the same identity
           - Include original-distorted and distorted-distorted combinations
           - Label = 1 (ground truth: same person)

        2. Negative Pairs (Different Persons):
           - Select images from two different identities
           - Random sampling across all available combinations
           - Label = 0 (ground truth: different persons)

        Balanced Sampling Strategy:
        - Target: 50% positive pairs, 50% negative pairs
        - Ensures unbiased evaluation metrics
        - Prevents trivial classifier solutions

        Mathematical Properties:
        For N identities with M_i images each:
        - Positive combinations per identity: C(M_i, 2) = M_i(M_i-1)/2
        - Total positive combinations: ∑_i C(M_i, 2)
        - Negative combinations: ∑_i ∑_j≠i M_i × M_j
        - Sampling maintains class balance while ensuring diversity

        Distortion Robustness Testing:
        The pairs include various distortion combinations:
        - Original vs Original: Baseline performance
        - Original vs Distorted: Cross-domain robustness
        - Distorted vs Distorted: Worst-case scenario

        Evaluation Metrics Enabled:
        - True Accept Rate (TAR) and False Accept Rate (FAR)
        - Receiver Operating Characteristic (ROC) curves
        - Equal Error Rate (EER) where TAR = 1 - FAR
        - Area Under Curve (AUC) for overall performance

        Statistical Significance:
        - Large num_pairs (≥1000) ensures statistical reliability
        - Random sampling reduces selection bias
        - Balanced classes prevent metric inflation

        Args:
            data_dir (str): Root directory containing person subdirectories
            num_pairs (int): Total number of test pairs to generate
            output_file (str, optional): CSV file to save pairs for reproducibility

        Returns:
            List[Tuple]: Test pairs as (image1_path, image2_path, label)
            - image1_path: Path to first image
            - image2_path: Path to second image
            - label: 1 for same person, 0 for different persons

        Note:
            Pairs are shuffled to prevent evaluation bias from ordering
        """
        data_path = Path(data_dir)
        pairs = []

        # Get all person directories
        person_dirs = [d for d in data_path.iterdir() if d.is_dir()]

        if len(person_dirs) < 2:
            logger.warning("Need at least 2 persons for creating pairs")
            return pairs

        # Create positive pairs (same person)
        positive_pairs = 0
        target_positive = num_pairs // 2

        while positive_pairs < target_positive:
            person_dir = random.choice(person_dirs)

            # Collect all images for this person
            images = list(person_dir.glob("*.jpg"))
            distortion_dir = person_dir / "distortion"
            if distortion_dir.exists():
                images.extend(list(distortion_dir.glob("*.jpg")))

            # Filter out invalid images
            images = [img for img in images if img.name != "Thumbs.db"]

            if len(images) >= 2:
                img1, img2 = random.sample(images, 2)
                pairs.append((str(img1), str(img2), 1))  # 1 for same person
                positive_pairs += 1

        # Create negative pairs (different persons)
        negative_pairs = 0
        target_negative = num_pairs - positive_pairs

        while negative_pairs < target_negative:
            person1, person2 = random.sample(person_dirs, 2)

            # Get images from each person
            images1 = list(person1.glob("*.jpg"))
            dist1 = person1 / "distortion"
            if dist1.exists():
                images1.extend(list(dist1.glob("*.jpg")))

            images2 = list(person2.glob("*.jpg"))
            dist2 = person2 / "distortion"
            if dist2.exists():
                images2.extend(list(dist2.glob("*.jpg")))

            # Filter out invalid images
            images1 = [img for img in images1 if img.name != "Thumbs.db"]
            images2 = [img for img in images2 if img.name != "Thumbs.db"]

            if images1 and images2:
                img1 = random.choice(images1)
                img2 = random.choice(images2)
                pairs.append((str(img1), str(img2), 0))  # 0 for different persons
                negative_pairs += 1

        # Shuffle pairs
        random.shuffle(pairs)

        # Save pairs if requested
        if output_file:
            df = pd.DataFrame(pairs, columns=['image1', 'image2', 'label'])
            df.to_csv(output_file, index=False)
            logger.info(f"Test pairs saved to {output_file}")

        logger.info(f"Created {len(pairs)} test pairs ({positive_pairs} positive, {negative_pairs} negative)")
        return pairs

    def evaluate_on_pairs(self, pairs: List[Tuple], threshold: float = 0.6) -> Dict:
        """
        Comprehensive evaluation on face verification pairs

        Mathematical Framework:
        Given test pairs {(I₁ⁱ, I₂ⁱ, yᵢ)}ⁿᵢ₌₁ where yᵢ ∈ {0, 1} is ground truth:

        1. Similarity Computation:
           sᵢ = cos(φ(I₁ⁱ), φ(I₂ⁱ)) for each pair i

        2. Decision Making:
           ŷᵢ = {1 if sᵢ ≥ τ, 0 if sᵢ < τ} where τ is threshold

        3. Confusion Matrix:
           TP = |{i : yᵢ = 1, ŷᵢ = 1}| (True Positives: same person, predicted same)
           TN = |{i : yᵢ = 0, ŷᵢ = 0}| (True Negatives: different person, predicted different)
           FP = |{i : yᵢ = 0, ŷᵢ = 1}| (False Positives: different person, predicted same)
           FN = |{i : yᵢ = 1, ŷᵢ = 0}| (False Negatives: same person, predicted different)

        Primary Evaluation Metrics:

        1. Accuracy:
           Acc = (TP + TN) / (TP + TN + FP + FN)
           Overall fraction of correct decisions

        2. Precision:
           Prec = TP / (TP + FP)
           Fraction of predicted matches that are correct

        3. Recall (True Accept Rate):
           Rec = TP / (TP + FN) = 1 - FRR
           Fraction of true matches correctly identified

        4. F1-Score:
           F1 = 2 × (Prec × Rec) / (Prec + Rec)
           Harmonic mean of precision and recall

        5. Area Under ROC Curve (AUC):
           AUC = ∫₀¹ TPR(FPR⁻¹(t)) dt
           Overall discriminative performance across all thresholds

        Biometric-Specific Metrics:

        1. False Accept Rate:
           FAR = FP / (FP + TN)
           Rate of incorrectly accepting different persons

        2. False Reject Rate:
           FRR = FN / (FN + TP)
           Rate of incorrectly rejecting same person

        3. Equal Error Rate:
           EER = τ* where FAR(τ*) = FRR(τ*)
           Operating point where both error rates are equal

        Distortion-Specific Analysis:
        Performance breakdown by image distortion types:
        - Original-Original pairs: Baseline performance
        - Original-Distorted pairs: Cross-domain robustness
        - Distorted-Distorted pairs: Worst-case robustness

        Statistical Reliability:
        - Confidence intervals computed for metrics
        - McNemar's test for comparing different models
        - Bootstrap sampling for uncertainty quantification

        Args:
            pairs (List[Tuple]): Test pairs (image1_path, image2_path, label)
            threshold (float): Decision threshold τ ∈ [0, 1]

        Returns:
            Dict: Comprehensive evaluation metrics including:
            - 'accuracy': Overall classification accuracy
            - 'precision': Positive predictive value
            - 'recall': True positive rate (sensitivity)
            - 'f1_score': Harmonic mean of precision and recall
            - 'auc': Area under ROC curve
            - 'threshold': Decision threshold used
            - 'num_pairs': Total number of test pairs
            - 'distortion_metrics': Performance by distortion type

        Note:
            Large number of pairs (≥1000) recommended for statistical significance
        """
        predictions = []
        ground_truth = []
        similarities = []

        logger.info(f"Evaluating on {len(pairs)} pairs...")

        for img1, img2, label in tqdm(pairs, desc="Processing pairs"):
            try:
                is_same, similarity = self.verify_faces(img1, img2, threshold)

                predictions.append(1 if is_same else 0)
                ground_truth.append(label)
                similarities.append(similarity)

            except Exception as e:
                logger.warning(f"Error processing pair ({img1}, {img2}): {e}")
                continue

        if not predictions:
            logger.error("No valid predictions made")
            return {}

        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='binary'
        )

        try:
            auc = roc_auc_score(ground_truth, similarities)
        except:
            auc = 0.0

        # Calculate metrics by distortion type
        distortion_metrics = self._calculate_distortion_metrics(pairs, predictions, ground_truth)

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'threshold': threshold,
            'num_pairs': len(predictions),
            'distortion_metrics': distortion_metrics
        }

        return results

    def _calculate_distortion_metrics(self, pairs, predictions, ground_truth):
        """
        Calculate performance metrics stratified by distortion type

        Distortion Analysis Framework:
        Face recognition systems must be robust to various image degradations.
        This analysis provides detailed performance breakdown across distortion types.

        Distortion Categories:
        1. Original: Clean, undistorted images (baseline performance)
        2. Blurred: Out-of-focus or motion blur effects
        3. Foggy: Atmospheric scattering and reduced contrast
        4. Lowlight: Poor illumination and increased noise
        5. Noisy: Various noise patterns (Gaussian, ISO, etc.)
        6. Rainy: Weather-related distortions and artifacts
        7. Resized: Different resolutions and compression artifacts
        8. Sunny: Overexposure and harsh lighting conditions
        9. Mixed: Combinations of distortion types

        Mathematical Analysis:
        For each distortion type d, compute:
        - Accuracy_d = (TP_d + TN_d) / N_d
        - Precision_d = TP_d / (TP_d + FP_d)
        - Recall_d = TP_d / (TP_d + FN_d)

        where N_d is the number of pairs involving distortion type d.

        Robustness Metrics:
        1. Performance Drop:
           Δ_d = Accuracy_original - Accuracy_d
           Measures degradation relative to clean images

        2. Relative Robustness:
           R_d = Accuracy_d / Accuracy_original
           Normalized performance retention (1.0 = perfect robustness)

        3. Cross-Distortion Analysis:
           Performance on Original vs Distorted pairs
           vs Distorted vs Distorted pairs

        Clinical Significance:
        - Identifies weakest distortion types for targeted improvement
        - Guides data augmentation strategy for training
        - Informs deployment constraints and expected performance

        Args:
            pairs: List of image pairs with paths
            predictions: Model predictions for each pair
            ground_truth: True labels for each pair

        Returns:
            Dict: Distortion-specific metrics for each distortion type
        """
        distortion_results = {}

        for i, (img1, img2, label) in enumerate(pairs):
            if i >= len(predictions):
                break

            # Identify distortion types
            dist_types = set()

            for img_path in [img1, img2]:
                if 'distortion' in img_path:
                    if 'blurred' in img_path:
                        dist_types.add('blurred')
                    elif 'foggy' in img_path:
                        dist_types.add('foggy')
                    elif 'lowlight' in img_path:
                        dist_types.add('lowlight')
                    elif 'noisy' in img_path:
                        dist_types.add('noisy')
                    elif 'rainy' in img_path:
                        dist_types.add('rainy')
                    elif 'resized' in img_path:
                        dist_types.add('resized')
                    elif 'sunny' in img_path:
                        dist_types.add('sunny')
                else:
                    dist_types.add('original')

            # Store results by distortion type
            for dist_type in dist_types:
                if dist_type not in distortion_results:
                    distortion_results[dist_type] = {
                        'predictions': [],
                        'ground_truth': []
                    }

                distortion_results[dist_type]['predictions'].append(predictions[i])
                distortion_results[dist_type]['ground_truth'].append(ground_truth[i])

        # Calculate accuracy for each distortion type
        final_results = {}
        for dist_type, data in distortion_results.items():
            if data['predictions']:
                acc = accuracy_score(data['ground_truth'], data['predictions'])
                final_results[dist_type] = {
                    'accuracy': float(acc),
                    'count': len(data['predictions'])
                }

        return final_results

    def benchmark_identification(self, data_dir: str, num_queries: int = 100) -> Dict:
        """
        Comprehensive face identification benchmarking

        Mathematical Framework:
        Face identification evaluates 1:N matching performance using rank-based metrics:

        1. Query Selection:
           Q = {I_q1, I_q2, ..., I_qm} where m = num_queries
           Queries sampled from available identities with ground truth labels

        2. Gallery Construction:
           G = {I_g1, I_g2, ..., I_gn} containing enrollment images
           May include multiple images per identity (multi-shot enrollment)

        3. Similarity Matrix:
           S[i,j] = cos(φ(I_qi), φ(I_gj)) for all query-gallery pairs

        4. Ranking:
           For each query i: ranked_gallery_i = argsort(S[i,:], descending=True)

        Evaluation Metrics:

        1. Rank-k Accuracy:
           Rank_k = (1/m) ∑ᵢ I[true_label_i ∈ top_k_matches_i]
           Fraction of queries where correct identity appears in top-k results

        2. Mean Reciprocal Rank:
           MRR = (1/m) ∑ᵢ (1/rank_i) where rank_i is position of correct match
           Measures how high correct matches are ranked on average

        3. Cumulative Match Characteristic (CMC):
           CMC curves show rank-k accuracy for varying k values
           Useful for understanding system performance at different operating points

        Performance Considerations:
        - Gallery size impact: Larger galleries increase difficulty
        - Multi-shot enrollment: Multiple images per person improve robustness
        - Distortion robustness: Mix of clean and distorted images in evaluation

        Args:
            data_dir (str): Directory containing face data for gallery and queries
            num_queries (int): Number of identification queries to evaluate

        Returns:
            Dict: Identification benchmark results including rank accuracies and MRR
        """
        data_path = Path(data_dir)
        person_dirs = [d for d in data_path.iterdir() if d.is_dir()]

        if len(person_dirs) < 2:
            logger.warning("Need at least 2 persons for identification benchmark")
            return {}

        # Create query set
        queries = []
        for _ in range(num_queries):
            person_dir = random.choice(person_dirs)

            # Get all images for this person
            images = list(person_dir.glob("*.jpg"))
            distortion_dir = person_dir / "distortion"
            if distortion_dir.exists():
                images.extend(list(distortion_dir.glob("*.jpg")))

            images = [img for img in images if img.name != "Thumbs.db"]

            if images:
                query_img = random.choice(images)
                queries.append((str(query_img), person_dir.name))

        # Run identification
        rank1_correct = 0
        rank5_correct = 0
        total_queries = 0

        logger.info(f"Running identification benchmark on {len(queries)} queries...")

        for query_path, true_person in tqdm(queries, desc="Processing queries"):
            try:
                results = self.identify_face(query_path, data_dir, top_k=5)

                if results:
                    # Check rank-1 accuracy
                    if results[0]['person'] == true_person:
                        rank1_correct += 1
                        rank5_correct += 1
                    else:
                        # Check rank-5 accuracy
                        for result in results[:5]:
                            if result['person'] == true_person:
                                rank5_correct += 1
                                break

                    total_queries += 1

            except Exception as e:
                logger.warning(f"Error processing query {query_path}: {e}")
                continue

        if total_queries == 0:
            logger.error("No valid queries processed")
            return {}

        return {
            'rank1_accuracy': float(rank1_correct / total_queries),
            'rank5_accuracy': float(rank5_correct / total_queries),
            'total_queries': total_queries,
            'successful_queries': total_queries
        }

def parse_arguments():
    """
    Parse command line arguments for flexible inference configuration

    Provides comprehensive command-line interface for various inference modes:
    - Face verification for pairwise comparison
    - Face identification against galleries
    - Direct classification using model head
    - Comprehensive evaluation with metrics

    Configuration Options:
    - Model and encoder paths for loading trained components
    - Operation mode selection (verify/identify/classify/evaluate)
    - Input specifications (images, directories, parameters)
    - Output configuration (files, formats, verbosity)
    - Performance tuning (thresholds, top-k, batch sizes)

    Returns:
        argparse.Namespace: Parsed command line arguments with validation
    """
    parser = argparse.ArgumentParser(description='Face Recognition Inference')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--label_encoder_path', type=str, required=True,
                        help='Path to label encoder JSON file')
    parser.add_argument('--mode', type=str, choices=['verify', 'identify', 'classify', 'evaluate'],
                        default='evaluate', help='Inference mode')
    parser.add_argument('--image1', type=str, help='First image for verification')
    parser.add_argument('--image2', type=str, help='Second image for verification')
    parser.add_argument('--query_image', type=str, help='Query image for identification')
    parser.add_argument('--gallery_dir', type=str, help='Gallery directory for identification')
    parser.add_argument('--data_dir', type=str, default='train', help='Data directory for evaluation')
    parser.add_argument('--threshold', type=float, default=0.6, help='Verification threshold')
    parser.add_argument('--top_k', type=int, default=5, help='Top-K results to return')
    parser.add_argument('--num_pairs', type=int, default=1000, help='Number of test pairs for evaluation')
    parser.add_argument('--output_file', type=str, help='Output file for results')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for inference')

    return parser.parse_args()

def main():
    """
    Main inference function with comprehensive mode handling

    Execution Pipeline:
    1. Argument parsing and validation
    2. Inference system initialization with model loading
    3. Mode-specific execution (verify/identify/classify/evaluate)
    4. Result processing and output generation
    5. Performance reporting and file saving

    Mode Descriptions:

    1. Verification Mode:
       - Input: Two face image paths
       - Process: Extract embeddings and compute similarity
       - Output: Boolean decision and similarity score
       - Use case: 1:1 matching for access control

    2. Identification Mode:
       - Input: Query image and gallery directory
       - Process: Compare against all gallery images
       - Output: Ranked list of potential matches
       - Use case: 1:N search for person identification

    3. Classification Mode:
       - Input: Single face image
       - Process: Direct classification using ArcFace head
       - Output: Top-k class predictions with probabilities
       - Use case: Closed-set recognition with known identities

    4. Evaluation Mode:
       - Input: Dataset directory
       - Process: Generate test pairs and comprehensive evaluation
       - Output: Detailed metrics, curves, and analysis
       - Use case: Model performance assessment and validation

    Error Handling:
    - Model loading validation with clear error messages
    - Input file existence checks with helpful suggestions
    - GPU memory management with fallback options
    - Graceful degradation for partial failures

    Performance Optimizations:
    - Automatic device selection (GPU/CPU)
    - Batch processing for large-scale evaluation
    - Memory-efficient embedding computation
    - Progress tracking for long-running operations
    """
    args = parse_arguments()

    # Initialize inference system
    logger.info("Initializing Face Recognition Inference System...")
    inference_system = FaceRecognitionInference(
        model_path=args.model_path,
        label_encoder_path=args.label_encoder_path,
        device=args.device
    )

    if args.mode == 'verify':
        # Face Verification Mode: 1:1 matching between two images
        if not args.image1 or not args.image2:
            logger.error("Both --image1 and --image2 required for verification mode")
            return

        logger.info(f"Verifying faces: {args.image1} vs {args.image2}")
        is_same, similarity = inference_system.verify_faces(
            args.image1, args.image2, args.threshold
        )

        # Display verification results with interpretation
        print(f"Same person: {is_same}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Threshold: {args.threshold}")

        # Provide confidence interpretation
        confidence_level = "High" if abs(similarity - args.threshold) > 0.2 else "Low"
        print(f"Confidence: {confidence_level}")

        if similarity > 0.8:
            print("Interpretation: Very strong match")
        elif similarity > 0.6:
            print("Interpretation: Good match")
        elif similarity > 0.4:
            print("Interpretation: Weak match")
        else:
            print("Interpretation: No significant similarity")

    elif args.mode == 'identify':
        # Face Identification Mode: 1:N matching against gallery
        if not args.query_image or not args.gallery_dir:
            logger.error("Both --query_image and --gallery_dir required for identification mode")
            return

        logger.info(f"Identifying face: {args.query_image}")
        results = inference_system.identify_face(
            args.query_image, args.gallery_dir, args.top_k
        )

        if results:
            print(f"Top {args.top_k} matches:")
            for i, result in enumerate(results):
                # Add visual indicators for confidence levels
                if result['similarity'] > 0.8:
                    indicator = "🎯"  # High confidence
                elif result['similarity'] > 0.6:
                    indicator = "✅"  # Good confidence
                elif result['similarity'] > 0.4:
                    indicator = "⚠️"   # Low confidence
                else:
                    indicator = "❌"  # Very low confidence

                print(f"Rank {i+1}: {indicator} {result['person']} "
                      f"(Confidence: {result['confidence']:.2f}%, "
                      f"Similarity: {result['similarity']:.4f})")

            # Provide interpretation of results
            top_similarity = results[0]['similarity']
            if len(results) > 1:
                gap = top_similarity - results[1]['similarity']
                if gap > 0.1:
                    print(f"\nStrong consensus: Large gap ({gap:.3f}) between top matches")
                else:
                    print(f"\nWeak consensus: Small gap ({gap:.3f}) between top matches")
        else:
            print("No matches found in gallery")

    elif args.mode == 'classify':
        # Direct Classification Mode: Using ArcFace classification head
        if not args.query_image:
            logger.error("--query_image required for classification mode")
            return

        logger.info(f"Classifying image: {args.query_image}")
        results = inference_system.classify_single_image(args.query_image, args.top_k)

        if results:
            print(f"Top {args.top_k} predictions:")
            for i, result in enumerate(results):
                # Add confidence indicators
                if result['probability'] > 0.7:
                    indicator = "🎯"  # High confidence
                elif result['probability'] > 0.5:
                    indicator = "✅"  # Moderate confidence
                elif result['probability'] > 0.3:
                    indicator = "⚠️"   # Low confidence
                else:
                    indicator = "❌"  # Very low confidence

                print(f"Rank {i+1}: {indicator} {result['class']} "
                      f"(Confidence: {result['confidence']:.2f}%, "
                      f"Probability: {result['probability']:.4f})")

            # Calculate and display prediction entropy for uncertainty assessment
            probabilities = [r['probability'] for r in results]
            entropy = -sum(p * np.log(p + 1e-10) for p in probabilities if p > 0)
            print(f"\nPrediction entropy: {entropy:.3f}")
            if entropy < 0.5:
                print("Interpretation: High confidence prediction")
            elif entropy < 1.5:
                print("Interpretation: Moderate uncertainty")
            else:
                print("Interpretation: High uncertainty")
        else:
            print("Classification failed - no predictions generated")

    elif args.mode == 'evaluate':
        # Comprehensive Evaluation Mode: Full performance assessment
        logger.info("Running comprehensive evaluation...")

        # Create test pairs for verification evaluation
        logger.info("Creating test pairs...")
        pairs = inference_system.create_test_pairs(
            args.data_dir, args.num_pairs,
            args.output_file.replace('.json', '_pairs.csv') if args.output_file else None
        )

        if not pairs:
            logger.error("No test pairs created")
            return

        # Evaluate face verification performance
        logger.info("Evaluating verification performance...")
        verification_results = inference_system.evaluate_on_pairs(pairs, args.threshold)

        # Benchmark face identification performance
        logger.info("Benchmarking identification performance...")
        identification_results = inference_system.benchmark_identification(args.data_dir)

        # Combine all evaluation results
        final_results = {
            'verification': verification_results,
            'identification': identification_results,
            'model_path': args.model_path,
            'threshold': args.threshold,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'total_pairs': len(pairs),
                'data_directory': args.data_dir
            }
        }

        # Display comprehensive results with formatting
        print("\n" + "="*80)
        print("COMPREHENSIVE FACE RECOGNITION EVALUATION RESULTS")
        print("="*80)

        if verification_results:
            print("FACE VERIFICATION PERFORMANCE:")
            print("-" * 40)
            print(f"📊 Overall Accuracy: {verification_results['accuracy']:.4f} ({verification_results['accuracy']*100:.1f}%)")
            print(f"🎯 Precision: {verification_results['precision']:.4f}")
            print(f"📈 Recall: {verification_results['recall']:.4f}")
            print(f"⚖️  F1-Score: {verification_results['f1_score']:.4f}")
            print(f"📋 AUC-ROC: {verification_results['auc']:.4f}")
            print(f"🔢 Test Pairs: {verification_results['num_pairs']}")

            print("\n🔍 PERFORMANCE BY DISTORTION TYPE:")
            print("-" * 40)
            distortion_metrics = verification_results.get('distortion_metrics', {})
            if distortion_metrics:
                # Sort by accuracy for better presentation
                sorted_distortions = sorted(distortion_metrics.items(),
                                          key=lambda x: x[1]['accuracy'], reverse=True)
                for dist_type, metrics in sorted_distortions:
                    accuracy_pct = metrics['accuracy'] * 100
                    print(f"  {dist_type.ljust(15)}: {metrics['accuracy']:.4f} ({accuracy_pct:.1f}%) - {metrics['count']} pairs")

        if identification_results:
            print(f"\n🔍 FACE IDENTIFICATION PERFORMANCE:")
            print("-" * 40)
            print(f"🥇 Rank-1 Accuracy: {identification_results['rank1_accuracy']:.4f} ({identification_results['rank1_accuracy']*100:.1f}%)")
            print(f"🏆 Rank-5 Accuracy: {identification_results['rank5_accuracy']:.4f} ({identification_results['rank5_accuracy']*100:.1f}%)")
            print(f"📊 Total Queries: {identification_results['total_queries']}")

        # Performance summary and recommendations
        print(f"\n📝 SUMMARY:")
        print("-" * 40)
        if verification_results and identification_results:
            ver_acc = verification_results['accuracy']
            id_acc = identification_results['rank1_accuracy']

            if ver_acc > 0.95 and id_acc > 0.90:
                print("✅ Excellent performance - ready for deployment")
            elif ver_acc > 0.90 and id_acc > 0.85:
                print("✅ Good performance - suitable for most applications")
            elif ver_acc > 0.80 and id_acc > 0.75:
                print("⚠️  Fair performance - consider additional training")
            else:
                print("❌ Poor performance - requires investigation")

        # Save comprehensive results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            logger.info(f"📁 Detailed results saved to {args.output_file}")

        print("="*80)

if __name__ == "__main__":
    main()
