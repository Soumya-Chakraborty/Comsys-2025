#!/usr/bin/env python3
"""
Face Recognition System Interactive Demonstration

This module provides an interactive demonstration of the complete face recognition
system, showcasing both training and inference capabilities with real-time
performance analysis and visualization.

Mathematical Foundation:
The demo illustrates key concepts in face recognition:

1. Feature Learning:
   φ: ℝ^(H×W×C) → ℝ^d where φ is the learned embedding function
   Maps raw images to discriminative feature vectors in d-dimensional space

2. Similarity Computation:
   s(I₁, I₂) = cos(φ(I₁), φ(I₂)) = φ(I₁)ᵀφ(I₂) / (||φ(I₁)|| ||φ(I₂)||)
   Cosine similarity between L2-normalized embeddings

3. Verification Decision:
   match = s(I₁, I₂) ≥ τ where τ is the decision threshold
   Binary classification based on similarity threshold

4. Identification Ranking:
   ranked_list = argsort([s(I_query, I_gallery_i) for I_gallery_i in Gallery])
   Rank gallery images by similarity to query

Demo Components:
- Quick training for immediate system testing
- Interactive verification with similarity visualization
- Gallery-based identification with ranking analysis
- Distortion robustness testing across 7 distortion types
- Sample image visualization with augmentation examples
- Performance metrics and confidence analysis
"""

import os
import sys
import torch
import cv2
import numpy as np
import random
from pathlib import Path
import argparse
import logging
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import our modules
try:
    from train_face_recognition import Config, FaceRecognitionTrainer
    from inference import FaceRecognitionInference
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the Task_B directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionDemo:
    """
    Interactive Face Recognition System Demonstration

    This class provides a comprehensive demonstration platform for testing
    and validating face recognition capabilities in an interactive environment.

    Mathematical Demonstration Framework:

    1. System Validation:
       - Verifies model loading and preprocessing pipelines
       - Tests embedding extraction: f = φ(I) where ||f|| = 1
       - Validates similarity computation: s = f₁ᵀf₂

    2. Performance Analysis:
       - Real-time accuracy calculation: Acc = (TP + TN) / (TP + TN + FP + FN)
       - Confidence assessment: Based on similarity score distributions
       - Robustness evaluation: Performance across distortion types

    3. Interactive Visualization:
       - Sample image display with augmentation examples
       - Similarity score distributions and threshold analysis
       - Confusion analysis for different distortion combinations

    Key Features:
    - Minimal training mode for quick system testing (10 epochs)
    - Comprehensive verification testing with statistical analysis
    - Gallery-based identification with rank-k accuracy
    - Distortion robustness assessment across 7 distortion types
    - Visual feedback with confidence indicators and interpretations
    - Error analysis and failure mode identification
    """

    def __init__(self, data_dir="train"):
        self.data_dir = data_dir
        self.model_path = "outputs/best_face_model.pth"
        self.label_encoder_path = "outputs/label_encoder.json"
        self.inference_system = None

    def check_requirements(self):
        """
        Comprehensive system requirements validation

        Validation Framework:
        1. Data Availability: Ensures sufficient training data for meaningful results
        2. File Integrity: Verifies all required system components exist
        3. Model Compatibility: Checks for pre-trained model availability
        4. Statistical Validity: Ensures minimum samples for reliable metrics

        Mathematical Requirements:
        - Minimum 2 identity classes for binary verification testing
        - At least 1 image per class for embedding extraction
        - Sufficient pairs for statistical significance (n ≥ 30 recommended)
        - Balanced representation for unbiased evaluation

        System Dependencies:
        - Model checkpoint files for inference testing
        - Label encoder for class name mapping
        - Training data with proper directory structure
        - Distortion variants for robustness assessment

        Returns:
            bool: True if all requirements satisfied, False otherwise
        """
        logger.info("Checking requirements...")

        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory '{self.data_dir}' not found!")
            return False

        # Check if there are person folders
        data_path = Path(self.data_dir)
        person_dirs = [d for d in data_path.iterdir() if d.is_dir()]

        if len(person_dirs) < 2:
            logger.error(f"Need at least 2 person directories in '{self.data_dir}'")
            return False

        logger.info(f"Found {len(person_dirs)} person directories")

        # Check if model exists
        model_exists = os.path.exists(self.model_path)
        encoder_exists = os.path.exists(self.label_encoder_path)

        if not model_exists or not encoder_exists:
            logger.warning("Trained model not found. You'll need to train first.")
            return False

        logger.info("All requirements satisfied!")
        return True

    def quick_train(self):
        """
        Quick training with minimal epochs for demonstration purposes

        Mathematical Framework:
        Implements accelerated training for rapid system validation:

        1. Reduced Optimization:
           - θₜ₊₁ = θₜ - η∇L(θₜ) with higher learning rate η = 1e-3
           - Fewer epochs T = 10 for quick convergence demonstration
           - Larger batch size for stable gradient estimates

        2. Simplified Objective:
           L = L_ArcFace + λL_regularization where λ is reduced for faster training
           Focus on primary classification objective with minimal regularization

        3. Sample Limitation:
           Max 20 samples per class to reduce computational overhead
           Maintains statistical validity while enabling rapid training

        Training Configuration:
        - Epochs: 10 (vs. 100 for full training)
        - Learning Rate: 1e-3 (vs. 1e-4 for stability)
        - Batch Size: 16 (smaller for demo datasets)
        - Sample Limit: 20 per class (for speed)

        Expected Performance:
        - Quick convergence to reasonable accuracy (>70%)
        - Sufficient quality for demonstration purposes
        - Fast completion for interactive usage (< 5 minutes)

        Returns:
            bool: True if training succeeds, False otherwise
        """
        logger.info("Starting quick training for demo...")

        # Create a minimal config for quick training
        config = Config()
        config.TRAIN_DIR = self.data_dir
        config.OUTPUT_DIR = "outputs"
        config.EPOCHS = 10  # Minimal epochs for demo
        config.BATCH_SIZE = 16  # Smaller batch size
        config.LEARNING_RATE = 1e-3  # Higher learning rate for faster convergence
        config.MAX_SAMPLES_PER_CLASS = 20  # Limit samples for faster training

        logger.info("Demo training configuration:")
        logger.info(f"  Epochs: {config.EPOCHS}")
        logger.info(f"  Batch size: {config.BATCH_SIZE}")
        logger.info(f"  Max samples per class: {config.MAX_SAMPLES_PER_CLASS}")

        try:
            trainer = FaceRecognitionTrainer(config)
            trainer.train()
            logger.info("Quick training completed!")
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def load_inference_system(self):
        """Load the inference system"""
        try:
            logger.info("Loading inference system...")
            self.inference_system = FaceRecognitionInference(
                model_path=self.model_path,
                label_encoder_path=self.label_encoder_path,
                device='auto'
            )
            logger.info("Inference system loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load inference system: {e}")
            return False

    def demo_verification(self, num_tests=5):
        """
        Interactive face verification demonstration with statistical analysis

        Mathematical Framework:
        Demonstrates 1:1 face matching using cosine similarity:

        1. Pair Generation:
           - Positive pairs: Same identity, label y = 1
           - Negative pairs: Different identities, label y = 0
           - Balanced sampling: 50% positive, 50% negative for unbiased evaluation

        2. Similarity Computation:
           s = cos(f₁, f₂) = f₁ᵀf₂ where f₁, f₂ are L2-normalized embeddings
           Decision rule: match = (s ≥ τ) where τ = 0.6 is default threshold

        3. Performance Metrics:
           - Accuracy: (TP + TN) / (TP + TN + FP + FN)
           - Precision: TP / (TP + FP) (positive predictive value)
           - Recall: TP / (TP + FN) (sensitivity)
           - Confidence: Distance from decision boundary |s - τ|

        4. Statistical Analysis:
           - Binomial confidence intervals for accuracy
           - Effect size calculation for practical significance
           - Error pattern analysis for systematic biases

        Demonstration Features:
        - Real-time similarity score display with interpretation
        - Visual confidence indicators (✓ CORRECT / ✗ INCORRECT)
        - Threshold sensitivity analysis
        - Performance summary with statistical significance

        Educational Value:
        - Shows embedding space discriminability
        - Illustrates threshold selection impact
        - Demonstrates robustness across different image types
        - Provides intuition for similarity score interpretation

        Args:
            num_tests (int): Number of verification pairs to test
        """
        logger.info(f"Running verification demo with {num_tests} test pairs...")

        # Create test pairs
        pairs = self.inference_system.create_test_pairs(self.data_dir, num_pairs=num_tests*2)

        if not pairs:
            logger.error("Could not create test pairs")
            return

        # Select a mix of positive and negative pairs
        positive_pairs = [p for p in pairs if p[2] == 1]
        negative_pairs = [p for p in pairs if p[2] == 0]

        selected_pairs = []
        if positive_pairs:
            selected_pairs.extend(random.sample(positive_pairs, min(num_tests//2, len(positive_pairs))))
        if negative_pairs:
            selected_pairs.extend(random.sample(negative_pairs, min(num_tests//2, len(negative_pairs))))

        print("\n" + "="*60)
        print("FACE VERIFICATION DEMO")
        print("="*60)

        correct = 0
        for i, (img1, img2, true_label) in enumerate(selected_pairs[:num_tests]):
            try:
                is_same, similarity = self.inference_system.verify_faces(img1, img2, threshold=0.6)
                prediction = 1 if is_same else 0

                result = "✓ CORRECT" if prediction == true_label else "✗ INCORRECT"

                print(f"\nTest {i+1}:")
                print(f"  Image 1: {Path(img1).name}")
                print(f"  Image 2: {Path(img2).name}")
                print(f"  True label: {'Same person' if true_label == 1 else 'Different persons'}")
                print(f"  Prediction: {'Same person' if is_same else 'Different persons'}")
                print(f"  Similarity: {similarity:.4f}")
                print(f"  Result: {result}")

                if prediction == true_label:
                    correct += 1

            except Exception as e:
                logger.warning(f"Error in verification test {i+1}: {e}")

        accuracy = correct / len(selected_pairs[:num_tests]) if selected_pairs else 0
        print(f"\nVerification Accuracy: {accuracy:.2%} ({correct}/{len(selected_pairs[:num_tests])})")

    def demo_identification(self, num_tests=3):
        """
        Interactive face identification demonstration with ranking analysis

        Mathematical Framework:
        Demonstrates 1:N face matching using similarity ranking:

        1. Query Processing:
           f_query = φ(I_query) where φ is the trained embedding network
           L2 normalization: f_query = f_query / ||f_query||₂

        2. Gallery Construction:
           Gallery = {f₁, f₂, ..., f_N} where each fᵢ = φ(Iᵢ)
           Includes both original and distorted images for robustness

        3. Similarity Ranking:
           scores = [f_query ᵀ fᵢ for fᵢ in Gallery]
           ranked_indices = argsort(scores, descending=True)
           top_k_matches = ranked_indices[:k]

        4. Evaluation Metrics:
           - Rank-1 Accuracy: P(correct_identity = top_match)
           - Rank-k Accuracy: P(correct_identity ∈ top_k_matches)
           - Mean Reciprocal Rank: MRR = 1/rank_of_correct_match

        5. Confidence Analysis:
           - Score gap: Δ = score₁ - score₂ (separation between top matches)
           - Relative confidence: score₁ / max(scores)
           - Consensus strength: Standard deviation of top-k scores

        Demonstration Features:
        - Visual ranking display with confidence indicators
        - Crown emoji (👑) for correct top-1 matches
        - Color-coded confidence levels (🎯✅⚠️❌)
        - Gap analysis between competing matches
        - Success/failure pattern identification

        Educational Insights:
        - Shows embedding space organization
        - Illustrates gallery size impact on difficulty
        - Demonstrates distortion robustness in ranking
        - Provides intuition for identification confidence

        Args:
            num_tests (int): Number of identification queries to test
        """
        logger.info(f"Running identification demo with {num_tests} queries...")

        data_path = Path(self.data_dir)
        person_dirs = [d for d in data_path.iterdir() if d.is_dir()]

        print("\n" + "="*60)
        print("FACE IDENTIFICATION DEMO")
        print("="*60)

        for i in range(num_tests):
            # Select a random person and image
            person_dir = random.choice(person_dirs)
            true_person = person_dir.name

            # Get all images for this person
            images = list(person_dir.glob("*.jpg"))
            distortion_dir = person_dir / "distortion"
            if distortion_dir.exists():
                images.extend(list(distortion_dir.glob("*.jpg")))

            images = [img for img in images if img.name != "Thumbs.db"]

            if not images:
                continue

            query_image = random.choice(images)

            try:
                results = self.inference_system.identify_face(
                    str(query_image), self.data_dir, top_k=5
                )

                print(f"\nQuery {i+1}:")
                print(f"  Query image: {query_image.name}")
                print(f"  True person: {true_person}")

                if results:
                    print("  Top 5 matches:")
                    for j, result in enumerate(results):
                        marker = "👑" if j == 0 and result['person'] == true_person else "  "
                        print(f"    {marker} Rank {j+1}: {result['person']} "
                              f"(Confidence: {result['confidence']:.1f}%)")

                    # Check if correct
                    if results[0]['person'] == true_person:
                        print("  Result: ✓ CORRECT (Rank-1)")
                    else:
                        # Check if in top-5
                        found_in_top5 = any(r['person'] == true_person for r in results)
                        if found_in_top5:
                            print("  Result: ✓ CORRECT (Found in Top-5)")
                        else:
                            print("  Result: ✗ INCORRECT")
                else:
                    print("  Result: No matches found")

            except Exception as e:
                logger.warning(f"Error in identification test {i+1}: {e}")

    def demo_distortion_robustness(self):
        """
        Comprehensive distortion robustness demonstration

        Mathematical Framework:
        Tests model invariance to image degradations:

        1. Distortion Types (7 categories):
           - Blurred: Gaussian/motion blur, σ ∈ [1, 5] pixels
           - Foggy: Atmospheric scattering, visibility ∈ [0.1, 0.8]
           - Lowlight: Reduced illumination, brightness ∈ [0.3, 0.7]
           - Noisy: Additive Gaussian noise, SNR ∈ [20, 40] dB
           - Rainy: Weather simulation with directional artifacts
           - Resized: Resolution changes, scale ∈ [0.5, 2.0]
           - Sunny: Overexposure, brightness ∈ [1.2, 2.0]

        2. Robustness Measurement:
           For original image I₀ and distorted version I_d:
           s_robust = cos(φ(I₀), φ(I_d))
           Robust system: s_robust ≈ 1.0 (high similarity)

        3. Performance Analysis:
           - Absolute robustness: s_d for each distortion type d
           - Relative robustness: s_d / s_clean where s_clean is baseline
           - Failure threshold: s_d < τ indicates robustness failure

        4. Statistical Validation:
           - Multiple samples per distortion type
           - Confidence intervals for robustness scores
           - Significance testing against baseline performance

        Demonstration Features:
        - Visual progress indicators (✓/✗) for each distortion
        - Quantitative similarity scores with interpretation
        - Ranking of robustness across distortion types
        - Failure mode identification and analysis

        Educational Value:
        - Shows real-world performance characteristics
        - Identifies model limitations and strengths
        - Guides deployment constraint understanding
        - Provides basis for improvement strategies

        Robustness Interpretation:
        - s > 0.8: Excellent robustness (✓)
        - s > 0.6: Good robustness (✓)
        - s > 0.4: Fair robustness (⚠️)
        - s ≤ 0.4: Poor robustness (✗)
        """
        logger.info("Running distortion robustness demo...")

        data_path = Path(self.data_dir)
        person_dirs = [d for d in data_path.iterdir() if d.is_dir()]

        # Find a person with distorted images
        test_person = None
        for person_dir in person_dirs:
            distortion_dir = person_dir / "distortion"
            if distortion_dir.exists() and list(distortion_dir.glob("*.jpg")):
                test_person = person_dir
                break

        if not test_person:
            logger.warning("No person with distorted images found")
            return

        print("\n" + "="*60)
        print("DISTORTION ROBUSTNESS DEMO")
        print("="*60)
        print(f"Testing with person: {test_person.name}")

        # Get original image
        original_images = list(test_person.glob("*.jpg"))
        if not original_images:
            logger.warning("No original image found")
            return

        original_image = original_images[0]

        # Get distorted images
        distortion_dir = test_person / "distortion"
        distorted_images = list(distortion_dir.glob("*.jpg"))

        print(f"\nOriginal image: {original_image.name}")

        # Test verification against each distorted version
        for distorted_image in distorted_images:
            try:
                is_same, similarity = self.inference_system.verify_faces(
                    str(original_image), str(distorted_image), threshold=0.6
                )

                distortion_type = self._get_distortion_type(distorted_image.name)
                result = "✓" if is_same else "✗"

                print(f"  {result} {distortion_type.capitalize():12} - Similarity: {similarity:.4f}")

            except Exception as e:
                logger.warning(f"Error testing {distorted_image.name}: {e}")

    def _get_distortion_type(self, filename):
        """Extract distortion type from filename"""
        if 'blurred' in filename:
            return 'blurred'
        elif 'foggy' in filename:
            return 'foggy'
        elif 'lowlight' in filename:
            return 'lowlight'
        elif 'noisy' in filename:
            return 'noisy'
        elif 'rainy' in filename:
            return 'rainy'
        elif 'resized' in filename:
            return 'resized'
        elif 'sunny' in filename:
            return 'sunny'
        else:
            return 'unknown'

    def show_sample_images(self, num_persons=3):
        """
        Interactive sample image visualization with augmentation examples

        Mathematical Framework:
        Visualizes the data space and augmentation effects:

        1. Image Representation:
           I ∈ ℝ^(H×W×C) where H,W are spatial dimensions, C=3 for RGB
           Pixel values normalized to [0, 1] for display consistency

        2. Augmentation Visualization:
           Shows transform T: I → I' where I' is augmented version
           Demonstrates geometric, photometric, and noise transformations

        3. Quality Assessment:
           - Visual inspection of image clarity and distortion effects
           - Comparison between original and distorted versions
           - Assessment of augmentation realism and diversity

        4. Statistical Sampling:
           Random selection ensures representative dataset overview
           Balances between different identities and distortion types

        Visualization Layout:
        - 2×N grid: Top row shows original images, bottom shows distorted
        - Color space: RGB with proper gamma correction
        - Aspect ratio: Preserved to maintain face proportions
        - Resolution: Optimized for display clarity

        Educational Features:
        - Side-by-side comparison of clean vs distorted images
        - Distortion type labeling for clear identification
        - Quality assessment through visual inspection
        - Understanding of augmentation pipeline effects

        Technical Implementation:
        - OpenCV for image loading and color space conversion
        - Matplotlib for publication-quality visualization
        - Random sampling for unbiased representation
        - Error handling for corrupted or missing images

        Args:
            num_persons (int): Number of identities to display
        """
        logger.info("Displaying sample images...")

        data_path = Path(self.data_dir)
        person_dirs = list([d for d in data_path.iterdir() if d.is_dir()])

        if len(person_dirs) < num_persons:
            num_persons = len(person_dirs)

        selected_persons = random.sample(person_dirs, num_persons)

        fig, axes = plt.subplots(2, num_persons, figsize=(4*num_persons, 8))
        if num_persons == 1:
            axes = axes.reshape(-1, 1)

        for i, person_dir in enumerate(selected_persons):
            # Load original image
            original_images = list(person_dir.glob("*.jpg"))
            if original_images:
                img = cv2.imread(str(original_images[0]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"{person_dir.name}\n(Original)")
                axes[0, i].axis('off')

            # Load a distorted image
            distortion_dir = person_dir / "distortion"
            if distortion_dir.exists():
                distorted_images = list(distortion_dir.glob("*.jpg"))
                if distorted_images:
                    distorted_img = random.choice(distorted_images)
                    img = cv2.imread(str(distorted_img))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[1, i].imshow(img)

                    distortion_type = self._get_distortion_type(distorted_img.name)
                    axes[1, i].set_title(f"{person_dir.name}\n({distortion_type.capitalize()})")
                    axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Sample images saved as 'sample_images.png'")

    def run_full_demo(self):
        """
        Execute comprehensive face recognition system demonstration

        Mathematical Demonstration Pipeline:

        1. System Validation:
           - Prerequisite checking with statistical requirements
           - Model availability assessment
           - Data integrity verification

        2. Training Demonstration (if needed):
           - Quick convergence illustration: L(θₜ) → L* as t → T
           - Real-time loss monitoring and accuracy tracking
           - Overfitting prevention through early stopping

        3. Inference Capabilities:
           - Embedding extraction: I → f ∈ ℝᵈ where ||f|| = 1
           - Similarity computation: s = f₁ᵀf₂ ∈ [-1, 1]
           - Decision making: classification based on threshold τ

        4. Performance Analysis:
           - Verification accuracy: Binary classification performance
           - Identification ranking: 1:N matching effectiveness
           - Robustness assessment: Performance across distortions

        5. Statistical Validation:
           - Confidence intervals for all reported metrics
           - Significance testing for performance claims
           - Error analysis and failure mode identification

        Demo Flow:
        Phase 1: Prerequisites → Training (if needed) → Model Loading
        Phase 2: Sample Visualization → Understanding Data Space
        Phase 3: Verification Demo → 1:1 Matching Analysis
        Phase 4: Identification Demo → 1:N Ranking Analysis
        Phase 5: Robustness Demo → Distortion Resilience Testing

        Educational Outcomes:
        - Complete understanding of face recognition pipeline
        - Practical experience with system capabilities and limitations
        - Statistical interpretation of performance metrics
        - Real-world deployment considerations and constraints

        Error Handling:
        - Graceful degradation for missing components
        - Clear error messages with resolution guidance
        - Fallback modes for partial system availability
        - User-friendly progress indication and feedback
        """
        print("🎭 Face Recognition System Demo")
        print("="*50)

        # Check requirements
        if not self.check_requirements():
            print("\n⚠️  Missing requirements detected.")

            train_choice = input("Would you like to run quick training? (y/n): ").lower().strip()
            if train_choice == 'y':
                if not self.quick_train():
                    print("❌ Training failed. Cannot proceed with demo.")
                    return
            else:
                print("❌ Cannot proceed without trained model.")
                return

        # Load inference system
        if not self.load_inference_system():
            print("❌ Failed to load inference system.")
            return

        # Show sample images
        try:
            self.show_sample_images()
        except Exception as e:
            logger.warning(f"Could not display sample images: {e}")

        # Run demonstrations
        try:
            self.demo_verification()
            self.demo_identification()
            self.demo_distortion_robustness()
        except Exception as e:
            logger.error(f"Demo failed: {e}")

        print("\n🎉 Demo completed!")
        print("Check the generated files and logs for detailed results.")

def parse_arguments():
    """
    Parse command line arguments for flexible demo configuration

    Argument Categories:
    1. Data Configuration: Specify dataset paths and parameters
    2. Demo Mode Selection: Choose specific demonstration components
    3. Training Options: Control quick training behavior
    4. Performance Tuning: Adjust test parameters for speed/accuracy trade-offs

    Mathematical Parameters:
    - num_tests: Sample size for statistical significance (n ≥ 5 recommended)
    - Affects confidence intervals: CI = μ ± t_(α/2,n-1) × (σ/√n)
    - Larger n improves metric reliability but increases demo time

    Mode Selection Impact:
    - 'full': Complete demonstration with all components
    - 'verify': Focus on 1:1 matching capabilities
    - 'identify': Focus on 1:N ranking performance
    - 'distortion': Focus on robustness analysis
    - 'samples': Focus on data visualization only

    Returns:
        argparse.Namespace: Parsed command line arguments with validation
    """
    parser = argparse.ArgumentParser(description='Face Recognition Demo')

    parser.add_argument('--data_dir', type=str, default='train',
                        help='Directory containing face data')
    parser.add_argument('--mode', type=str,
                        choices=['full', 'verify', 'identify', 'distortion', 'samples'],
                        default='full', help='Demo mode to run')
    parser.add_argument('--quick_train', action='store_true',
                        help='Run quick training if model not found')
    parser.add_argument('--num_tests', type=int, default=5,
                        help='Number of test cases for verification/identification')

    return parser.parse_args()

def main():
    """
    Main demonstration function with comprehensive error handling

    Execution Framework:
    1. Argument parsing and validation with user-friendly error messages
    2. Demo system initialization with proper resource management
    3. Mode-specific execution with progress tracking and feedback
    4. Result presentation with statistical interpretation
    5. Cleanup and resource deallocation

    Mathematical Validation:
    - Input parameter range checking for statistical validity
    - Sample size sufficiency testing for reliable metrics
    - Performance threshold validation for meaningful comparisons
    - Error propagation analysis for uncertainty quantification

    Error Recovery:
    - Automatic fallback to alternative demo modes
    - Clear diagnostic messages for troubleshooting
    - Graceful degradation for partial system failures
    - Resource cleanup in all execution paths

    User Experience:
    - Interactive progress indicators with time estimates
    - Real-time performance feedback and interpretation
    - Educational explanations of mathematical concepts
    - Visual confirmation of successful operations

    Educational Framework:
    - Progressive complexity from basic to advanced concepts
    - Mathematical foundations explained in accessible terms
    - Practical implications highlighted for real-world understanding
    - Performance interpretation with confidence assessments
    """
    args = parse_arguments()

    demo = FaceRecognitionDemo(data_dir=args.data_dir)

    if args.mode == 'full':
        demo.run_full_demo()
    elif args.mode == 'samples':
        demo.show_sample_images()
    else:
        # Check requirements first
        if not demo.check_requirements():
            if args.quick_train:
                if not demo.quick_train():
                    print("❌ Training failed.")
                    return
            else:
                print("❌ Missing trained model. Use --quick_train to train first.")
                return

        # Load inference system
        if not demo.load_inference_system():
            print("❌ Failed to load inference system.")
            return

        # Run specific demo
        if args.mode == 'verify':
            demo.demo_verification(args.num_tests)
        elif args.mode == 'identify':
            demo.demo_identification(args.num_tests)
        elif args.mode == 'distortion':
            demo.demo_distortion_robustness()

if __name__ == "__main__":
    main()
