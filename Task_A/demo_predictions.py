#!/usr/bin/env python3
"""
Demo script for testing gender classification predictions on individual images
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Import our custom modules
from gender_classifier import GenderClassifier, CNNTransformerHybrid
from ensemble_classifier import EnsembleModel

class GenderPredictionDemo:
    """Demo class for gender prediction on individual images"""

    def __init__(self, model_path=None, model_type='single', device=None):
        """Initialize the prediction demo

        Args:
            model_path: Path to the trained model
            model_type: Type of model ('single', 'ensemble', 'deployment')
            device: Device to run inference on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model = None
        self.transform = self.create_transform()

        print(f"Initializing Gender Prediction Demo")
        print(f"Device: {self.device}")
        print(f"Model type: {model_type}")

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("⚠️  No model path provided or model not found")
            print("Please train a model first or provide a valid model path")

    def create_transform(self):
        """Create image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """Load the trained model"""
        try:
            print(f"Loading model from: {model_path}")

            if self.model_type == 'single':
                # Load single CNN-Transformer model
                self.model = CNNTransformerHybrid(num_classes=2, dropout_rate=0.3)
                checkpoint = torch.load(model_path, map_location=self.device)

                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)

            elif self.model_type == 'ensemble':
                # Load ensemble model
                self.model = EnsembleModel(num_classes=2, dropout_rate=0.3)
                checkpoint = torch.load(model_path, map_location=self.device)

                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)

            elif self.model_type == 'deployment':
                # Load deployment model (already wrapped)
                self.model = torch.load(model_path, map_location=self.device)

            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model = None

    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Optional: Face detection and cropping
            image_np = np.array(image)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # If face detected, crop to face region with padding
            if len(faces) > 0:
                x, y, w, h = faces[0]
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image_np.shape[1] - x, w + 2*padding)
                h = min(image_np.shape[0] - y, h + 2*padding)
                image = image.crop((x, y, x+w, y+h))
                print(f"✅ Face detected and cropped")
            else:
                print("ℹ️  No face detected, using full image")

            # Apply transform
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            return image_tensor, image

        except Exception as e:
            print(f"❌ Error preprocessing image: {str(e)}")
            return None, None

    def predict_single_image(self, image_path, show_image=True, save_result=False):
        """Predict gender for a single image"""

        if self.model is None:
            print("❌ No model loaded")
            return None

        print(f"\n🔍 Predicting gender for: {image_path}")

        # Preprocess image
        image_tensor, processed_image = self.preprocess_image(image_path)
        if image_tensor is None:
            return None

        # Make prediction
        try:
            with torch.no_grad():
                if self.model_type == 'ensemble':
                    # Ensemble model with different modes
                    outputs, individual_preds, weighted_pred, meta_pred = self.model(image_tensor, mode='ensemble')
                elif self.model_type == 'deployment':
                    # Deployment model (returns processed results)
                    predicted_class, confidence, probabilities = self.model(image_tensor)
                    outputs = None  # Not needed for deployment model
                else:
                    # Single model
                    outputs = self.model(image_tensor)

                if self.model_type != 'deployment':
                    # Calculate probabilities and predictions
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                    prob_female = probabilities[0][0].item()
                    prob_male = probabilities[0][1].item()
                else:
                    # Deployment model already returns processed results
                    predicted_class = predicted_class.item() if torch.is_tensor(predicted_class) else predicted_class
                    confidence = confidence.item() if torch.is_tensor(confidence) else confidence
                    prob_female = probabilities[0].item() if torch.is_tensor(probabilities[0]) else probabilities[0]
                    prob_male = probabilities[1].item() if torch.is_tensor(probabilities[1]) else probabilities[1]

            # Format results
            gender = 'Female' if predicted_class == 0 else 'Male'
            result = {
                'image_path': image_path,
                'predicted_gender': gender,
                'confidence': confidence,
                'probabilities': {
                    'female': prob_female,
                    'male': prob_male
                }
            }

            # Print results
            print(f"📊 Prediction Results:")
            print(f"   Gender: {gender}")
            print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"   Probabilities:")
            print(f"     Female: {prob_female:.4f} ({prob_female*100:.2f}%)")
            print(f"     Male: {prob_male:.4f} ({prob_male*100:.2f}%)")

            # Show image with prediction
            if show_image:
                self.display_prediction(processed_image, result)

            # Save results
            if save_result:
                self.save_prediction_result(result, processed_image)

            return result

        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            return None

    def predict_batch(self, image_folder, output_file=None):
        """Predict gender for all images in a folder"""

        if self.model is None:
            print("❌ No model loaded")
            return None

        print(f"\n📁 Processing images in folder: {image_folder}")

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []

        for file_path in Path(image_folder).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))

        if not image_files:
            print("❌ No image files found in the specified folder")
            return None

        print(f"Found {len(image_files)} images")

        # Process each image
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            result = self.predict_single_image(image_path, show_image=False, save_result=False)
            if result:
                results.append(result)

        # Save batch results
        if output_file:
            self.save_batch_results(results, output_file)

        # Print summary
        self.print_batch_summary(results)

        return results

    def display_prediction(self, image, result):
        """Display image with prediction results"""
        try:
            plt.figure(figsize=(10, 8))

            # Display image
            plt.subplot(2, 1, 1)
            plt.imshow(image)
            plt.title(f"Predicted: {result['predicted_gender']} (Confidence: {result['confidence']:.2%})")
            plt.axis('off')

            # Display probability bar chart
            plt.subplot(2, 1, 2)
            genders = ['Female', 'Male']
            probabilities = [result['probabilities']['female'], result['probabilities']['male']]
            colors = ['pink', 'lightblue']

            bars = plt.bar(genders, probabilities, color=colors, alpha=0.7)
            plt.ylim(0, 1)
            plt.ylabel('Probability')
            plt.title('Gender Classification Probabilities')

            # Add value labels on bars
            for bar, prob in zip(bars, probabilities):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{prob:.2%}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️  Could not display image: {str(e)}")

    def save_prediction_result(self, result, image):
        """Save prediction result to file"""
        try:
            # Create output directory
            output_dir = "prediction_results"
            os.makedirs(output_dir, exist_ok=True)

            # Save image with annotation
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Original image
            ax1.imshow(image)
            ax1.set_title(f"Input Image")
            ax1.axis('off')

            # Probability chart
            genders = ['Female', 'Male']
            probabilities = [result['probabilities']['female'], result['probabilities']['male']]
            colors = ['pink', 'lightblue']

            bars = ax2.bar(genders, probabilities, color=colors, alpha=0.7)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Probability')
            ax2.set_title(f"Prediction: {result['predicted_gender']} ({result['confidence']:.2%})")

            # Add value labels
            for bar, prob in zip(bars, probabilities):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{prob:.2%}', ha='center', va='bottom', fontweight='bold')

            # Save figure
            image_name = os.path.splitext(os.path.basename(result['image_path']))[0]
            output_path = os.path.join(output_dir, f"{image_name}_prediction.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"💾 Result saved to: {output_path}")

        except Exception as e:
            print(f"⚠️  Could not save result: {str(e)}")

    def save_batch_results(self, results, output_file):
        """Save batch prediction results to CSV"""
        try:
            import pandas as pd

            # Convert results to DataFrame
            data = []
            for result in results:
                data.append({
                    'image_path': result['image_path'],
                    'image_name': os.path.basename(result['image_path']),
                    'predicted_gender': result['predicted_gender'],
                    'confidence': result['confidence'],
                    'prob_female': result['probabilities']['female'],
                    'prob_male': result['probabilities']['male']
                })

            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            print(f"💾 Batch results saved to: {output_file}")

        except Exception as e:
            print(f"⚠️  Could not save batch results: {str(e)}")

    def print_batch_summary(self, results):
        """Print summary statistics for batch processing"""
        if not results:
            return

        # Calculate statistics
        total_images = len(results)
        female_predictions = sum(1 for r in results if r['predicted_gender'] == 'Female')
        male_predictions = total_images - female_predictions
        avg_confidence = np.mean([r['confidence'] for r in results])
        high_confidence = sum(1 for r in results if r['confidence'] > 0.9)
        low_confidence = sum(1 for r in results if r['confidence'] < 0.6)

        print(f"\n📊 BATCH PROCESSING SUMMARY")
        print(f"=" * 50)
        print(f"Total images processed: {total_images}")
        print(f"Female predictions: {female_predictions} ({female_predictions/total_images:.1%})")
        print(f"Male predictions: {male_predictions} ({male_predictions/total_images:.1%})")
        print(f"Average confidence: {avg_confidence:.3f} ({avg_confidence:.1%})")
        print(f"High confidence (>90%): {high_confidence} ({high_confidence/total_images:.1%})")
        print(f"Low confidence (<60%): {low_confidence} ({low_confidence/total_images:.1%})")
        print(f"=" * 50)

    def interactive_demo(self):
        """Interactive demo for testing individual images"""
        print(f"\n🎮 INTERACTIVE GENDER CLASSIFICATION DEMO")
        print(f"=" * 60)
        print("Commands:")
        print("  - Enter image path to classify")
        print("  - 'batch <folder_path>' to process all images in folder")
        print("  - 'quit' or 'exit' to stop")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n💭 Enter command or image path: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                elif user_input.lower().startswith('batch '):
                    folder_path = user_input[6:].strip()
                    if os.path.exists(folder_path):
                        output_file = f"batch_results_{int(time.time())}.csv"
                        self.predict_batch(folder_path, output_file)
                    else:
                        print(f"❌ Folder not found: {folder_path}")

                elif os.path.exists(user_input):
                    self.predict_single_image(user_input, show_image=True, save_result=True)

                else:
                    print(f"❌ File not found: {user_input}")
                    print("Please enter a valid image path or command")

            except KeyboardInterrupt:
                print("\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Gender Classification Demo")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--model-type", type=str, choices=['single', 'ensemble', 'deployment'],
                       default='single', help="Type of model to load")
    parser.add_argument("--image", type=str, help="Path to single image for prediction")
    parser.add_argument("--batch", type=str, help="Path to folder containing images for batch prediction")
    parser.add_argument("--output", type=str, help="Output file for batch results (CSV)")
    parser.add_argument("--interactive", action="store_true", help="Start interactive demo")
    parser.add_argument("--no-display", action="store_true", help="Don't display images")
    parser.add_argument("--save-results", action="store_true", help="Save prediction results")

    args = parser.parse_args()

    # Initialize demo
    demo = GenderPredictionDemo(
        model_path=args.model,
        model_type=args.model_type
    )

    if args.interactive:
        # Interactive mode
        demo.interactive_demo()

    elif args.image:
        # Single image prediction
        if os.path.exists(args.image):
            demo.predict_single_image(
                args.image,
                show_image=not args.no_display,
                save_result=args.save_results
            )
        else:
            print(f"❌ Image file not found: {args.image}")

    elif args.batch:
        # Batch prediction
        if os.path.exists(args.batch):
            output_file = args.output or f"batch_results_{int(time.time())}.csv"
            demo.predict_batch(args.batch, output_file)
        else:
            print(f"❌ Batch folder not found: {args.batch}")

    else:
        # No specific mode selected, show help
        print("Gender Classification Demo")
        print("\nUsage examples:")
        print("  python demo_predictions.py --model models/best_model.pth --image path/to/image.jpg")
        print("  python demo_predictions.py --model models/best_model.pth --batch path/to/folder --output results.csv")
        print("  python demo_predictions.py --model models/best_model.pth --interactive")
        print("\nFor full help: python demo_predictions.py --help")

if __name__ == "__main__":
    import time
    main()
