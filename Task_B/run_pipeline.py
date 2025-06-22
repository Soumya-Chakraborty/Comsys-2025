#!/usr/bin/env python3
"""
Face Recognition Pipeline Runner
Comprehensive integration script for end-to-end face recognition system
Handles training, evaluation, and deployment pipeline
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import shutil
from datetime import datetime
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceRecognitionPipeline:
    """End-to-end face recognition pipeline"""

    def __init__(self, config_file: str = "config.json"):
        """
        Initialize pipeline

        Args:
            config_file: Configuration file path
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.project_root = Path(__file__).parent
        self.pipeline_start_time = time.time()

        # Create pipeline directories
        self.pipeline_dir = self.project_root / "pipeline_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Setup pipeline logging
        self._setup_pipeline_logging()

        logger.info(f"Pipeline initialized - Run ID: {self.pipeline_dir.name}")

    def _load_config(self) -> Dict:
        """Load configuration from file"""
        config_path = self.project_root / self.config_file

        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()

        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    import yaml
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)

            logger.info(f"Configuration loaded from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default configuration")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "model": {
                "name": "vit_base_patch16_224",
                "embedding_dim": 512,
                "image_size": 224
            },
            "training": {
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 1e-4,
                "val_split": 0.2
            },
            "paths": {
                "train_dir": "train",
                "output_dir": "outputs"
            },
            "evaluation": {
                "num_test_pairs": 1000,
                "verification_threshold": 0.6
            }
        }

    def _setup_pipeline_logging(self):
        """Setup pipeline-specific logging"""
        # Create log file for this pipeline run
        log_file = self.pipeline_dir / "pipeline.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add to all loggers
        logging.getLogger().addHandler(file_handler)

        # Save pipeline config
        config_copy = self.pipeline_dir / "pipeline_config.json"
        with open(config_copy, 'w') as f:
            json.dump(self.config, f, indent=2)

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        logger.info("Checking prerequisites...")

        checks = []

        # Check Python version
        if sys.version_info >= (3, 8):
            logger.info(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}")
            checks.append(True)
        else:
            logger.error(f"✗ Python version too old: {sys.version_info.major}.{sys.version_info.minor}")
            checks.append(False)

        # Check required files
        required_files = [
            "train_face_recognition.py",
            "inference.py",
            "evaluation_utils.py",
            "batch_processor.py"
        ]

        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                logger.info(f"✓ {file_name}")
                checks.append(True)
            else:
                logger.error(f"✗ {file_name} not found")
                checks.append(False)

        # Check training data
        train_dir = Path(self.config["paths"]["train_dir"])
        if train_dir.exists():
            person_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
            if len(person_dirs) >= 2:
                logger.info(f"✓ Training data: {len(person_dirs)} persons found")
                checks.append(True)
            else:
                logger.warning(f"⚠ Only {len(person_dirs)} person directories found")
                checks.append(False)
        else:
            logger.error(f"✗ Training directory not found: {train_dir}")
            checks.append(False)

        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
                checks.append(True)
            else:
                logger.warning("⚠ No GPU available, training will be slower")
                checks.append(True)  # Not critical
        except ImportError:
            logger.warning("⚠ PyTorch not available for GPU check")
            checks.append(True)

        success = all(checks)
        logger.info(f"Prerequisites check: {'PASSED' if success else 'FAILED'}")
        return success

    def run_training(self, force_retrain: bool = False) -> bool:
        """Run model training"""
        logger.info("=" * 60)
        logger.info("TRAINING PHASE")
        logger.info("=" * 60)

        output_dir = Path(self.config["paths"]["output_dir"])
        model_path = output_dir / "best_face_model.pth"

        # Check if model already exists
        if model_path.exists() and not force_retrain:
            logger.info("Trained model already exists. Use --force-retrain to retrain.")
            return True

        # Prepare training command
        train_cmd = [
            sys.executable, "train_face_recognition.py",
            "--train_dir", self.config["paths"]["train_dir"],
            "--output_dir", str(output_dir),
            "--batch_size", str(self.config["training"]["batch_size"]),
            "--epochs", str(self.config["training"]["epochs"]),
            "--learning_rate", str(self.config["training"]["learning_rate"]),
            "--embedding_dim", str(self.config["model"]["embedding_dim"]),
            "--model_name", self.config["model"]["name"],
            "--image_size", str(self.config["model"]["image_size"])
        ]

        logger.info(f"Training command: {' '.join(train_cmd)}")

        try:
            # Run training
            start_time = time.time()
            result = subprocess.run(
                train_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            training_time = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"Training completed successfully in {training_time/60:.1f} minutes")

                # Copy training outputs to pipeline directory
                if output_dir.exists():
                    pipeline_outputs = self.pipeline_dir / "training_outputs"
                    shutil.copytree(output_dir, pipeline_outputs, dirs_exist_ok=True)

                return True
            else:
                logger.error(f"Training failed with return code: {result.returncode}")
                logger.error(f"Training stderr: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Training timeout expired (2 hours)")
            return False
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False

    def run_evaluation(self) -> Dict:
        """Run comprehensive evaluation"""
        logger.info("=" * 60)
        logger.info("EVALUATION PHASE")
        logger.info("=" * 60)

        output_dir = Path(self.config["paths"]["output_dir"])
        model_path = output_dir / "best_face_model.pth"
        label_encoder_path = output_dir / "label_encoder.json"

        # Check if model exists
        if not model_path.exists():
            logger.error("Trained model not found. Run training first.")
            return {}

        # Prepare evaluation command
        eval_cmd = [
            sys.executable, "inference.py",
            "--model_path", str(model_path),
            "--label_encoder_path", str(label_encoder_path),
            "--mode", "evaluate",
            "--data_dir", self.config["paths"]["train_dir"],
            "--num_pairs", str(self.config["evaluation"]["num_test_pairs"]),
            "--threshold", str(self.config["evaluation"]["verification_threshold"]),
            "--output_file", str(self.pipeline_dir / "evaluation_results.json")
        ]

        logger.info(f"Evaluation command: {' '.join(eval_cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(
                eval_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            evaluation_time = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"Evaluation completed successfully in {evaluation_time/60:.1f} minutes")

                # Load results
                results_file = self.pipeline_dir / "evaluation_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    return results
                else:
                    logger.warning("Evaluation results file not found")
                    return {}
            else:
                logger.error(f"Evaluation failed with return code: {result.returncode}")
                logger.error(f"Evaluation stderr: {result.stderr}")
                return {}

        except subprocess.TimeoutExpired:
            logger.error("Evaluation timeout expired (1 hour)")
            return {}
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {}

    def run_dataset_analysis(self) -> Dict:
        """Run dataset analysis"""
        logger.info("=" * 60)
        logger.info("DATASET ANALYSIS PHASE")
        logger.info("=" * 60)

        output_dir = Path(self.config["paths"]["output_dir"])
        model_path = output_dir / "best_face_model.pth"
        label_encoder_path = output_dir / "label_encoder.json"

        # Check if model exists for batch processing
        if not (model_path.exists() and label_encoder_path.exists()):
            logger.warning("Model not found, running analysis without model")

            # Run simple dataset analysis
            try:
                from batch_processor import BatchProcessor
                processor = BatchProcessor(
                    model_path=str(model_path) if model_path.exists() else None,
                    label_encoder_path=str(label_encoder_path) if label_encoder_path.exists() else None,
                    output_dir=str(self.pipeline_dir / "analysis")
                )

                results = processor.dataset_analysis(self.config["paths"]["train_dir"])
                return results

            except Exception as e:
                logger.error(f"Dataset analysis failed: {e}")
                return {}

        # Run full batch analysis
        analysis_cmd = [
            sys.executable, "batch_processor.py",
            "--model_path", str(model_path),
            "--label_encoder_path", str(label_encoder_path),
            "--mode", "analysis",
            "--data_dir", self.config["paths"]["train_dir"],
            "--output_dir", str(self.pipeline_dir / "analysis")
        ]

        logger.info(f"Analysis command: {' '.join(analysis_cmd)}")

        try:
            result = subprocess.run(
                analysis_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            if result.returncode == 0:
                logger.info("Dataset analysis completed successfully")

                # Try to load results
                analysis_file = self.pipeline_dir / "analysis" / f"dataset_analysis_{Path(self.config['paths']['train_dir']).name}.json"
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        results = json.load(f)
                    return results
                else:
                    return {"status": "completed", "details": "Results saved to analysis directory"}
            else:
                logger.error(f"Dataset analysis failed: {result.stderr}")
                return {}

        except Exception as e:
            logger.error(f"Dataset analysis error: {e}")
            return {}

    def run_demo(self) -> bool:
        """Run demonstration"""
        logger.info("=" * 60)
        logger.info("DEMONSTRATION PHASE")
        logger.info("=" * 60)

        demo_cmd = [
            sys.executable, "demo.py",
            "--data_dir", self.config["paths"]["train_dir"],
            "--mode", "full"
        ]

        logger.info(f"Demo command: {' '.join(demo_cmd)}")

        try:
            result = subprocess.run(
                demo_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            if result.returncode == 0:
                logger.info("Demo completed successfully")

                # Copy demo outputs
                demo_files = ["sample_images.png", "training_curves.png"]
                for demo_file in demo_files:
                    src_file = self.project_root / demo_file
                    if src_file.exists():
                        shutil.copy2(src_file, self.pipeline_dir / demo_file)

                return True
            else:
                logger.error(f"Demo failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Demo error: {e}")
            return False

    def generate_pipeline_report(self, eval_results: Dict, analysis_results: Dict) -> str:
        """Generate comprehensive pipeline report"""
        logger.info("Generating pipeline report...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FACE RECOGNITION PIPELINE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Pipeline Run ID: {self.pipeline_dir.name}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Pipeline Time: {(time.time() - self.pipeline_start_time)/60:.1f} minutes")
        report_lines.append("")

        # Configuration
        report_lines.append("CONFIGURATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Model: {self.config['model']['name']}")
        report_lines.append(f"Embedding Dimension: {self.config['model']['embedding_dim']}")
        report_lines.append(f"Training Epochs: {self.config['training']['epochs']}")
        report_lines.append(f"Batch Size: {self.config['training']['batch_size']}")
        report_lines.append(f"Learning Rate: {self.config['training']['learning_rate']}")
        report_lines.append("")

        # Dataset Information
        if analysis_results:
            report_lines.append("DATASET INFORMATION")
            report_lines.append("-" * 40)

            if 'persons' in analysis_results:
                report_lines.append(f"Number of Persons: {len(analysis_results['persons'])}")

            if 'image_stats' in analysis_results:
                stats = analysis_results['image_stats']
                report_lines.append(f"Total Images: {stats.get('total_images', 'Unknown')}")
                report_lines.append(f"Original Images: {stats.get('original_images', 'Unknown')}")
                report_lines.append(f"Distorted Images: {stats.get('distorted_images', 'Unknown')}")

            if 'distortion_stats' in analysis_results:
                report_lines.append("Distortion Types:")
                for dist_type, count in analysis_results['distortion_stats'].items():
                    report_lines.append(f"  {dist_type}: {count}")

            report_lines.append("")

        # Performance Results
        if eval_results:
            report_lines.append("PERFORMANCE RESULTS")
            report_lines.append("-" * 40)

            if 'verification' in eval_results:
                ver_results = eval_results['verification']
                report_lines.append("Face Verification:")
                report_lines.append(f"  Accuracy: {ver_results.get('accuracy', 'N/A'):.4f}")
                report_lines.append(f"  Precision: {ver_results.get('precision', 'N/A'):.4f}")
                report_lines.append(f"  Recall: {ver_results.get('recall', 'N/A'):.4f}")
                report_lines.append(f"  F1-Score: {ver_results.get('f1_score', 'N/A'):.4f}")
                report_lines.append(f"  AUC: {ver_results.get('auc', 'N/A'):.4f}")

            if 'identification' in eval_results:
                id_results = eval_results['identification']
                report_lines.append("Face Identification:")
                rank_acc = id_results.get('rank_accuracies', {})
                for rank, acc in rank_acc.items():
                    report_lines.append(f"  {rank.replace('_', '-').title()}: {acc:.4f}")

            report_lines.append("")

        # System Information
        report_lines.append("SYSTEM INFORMATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

        try:
            import torch
            report_lines.append(f"PyTorch Version: {torch.__version__}")
            if torch.cuda.is_available():
                report_lines.append(f"CUDA Available: Yes ({torch.cuda.get_device_name(0)})")
            else:
                report_lines.append("CUDA Available: No")
        except ImportError:
            report_lines.append("PyTorch: Not installed")

        report_lines.append("")

        # Files Generated
        report_lines.append("GENERATED FILES")
        report_lines.append("-" * 40)

        generated_files = []
        for file_path in self.pipeline_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(self.pipeline_dir)
                generated_files.append(str(rel_path))

        for file_name in sorted(generated_files):
            report_lines.append(f"  {file_name}")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        report_text = "\n".join(report_lines)
        report_file = self.pipeline_dir / "pipeline_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)

        logger.info(f"Pipeline report saved to {report_file}")
        return report_text

    def run_full_pipeline(self, skip_training: bool = False, skip_evaluation: bool = False,
                         skip_analysis: bool = False, skip_demo: bool = False,
                         force_retrain: bool = False) -> bool:
        """Run the complete pipeline"""
        logger.info("🚀 Starting Face Recognition Pipeline")
        logger.info(f"Pipeline ID: {self.pipeline_dir.name}")

        pipeline_success = True

        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed. Please fix issues and retry.")
            return False

        # Training phase
        if not skip_training:
            training_success = self.run_training(force_retrain=force_retrain)
            if not training_success:
                logger.error("Training phase failed")
                pipeline_success = False
        else:
            logger.info("Skipping training phase")

        # Evaluation phase
        eval_results = {}
        if not skip_evaluation and pipeline_success:
            eval_results = self.run_evaluation()
            if not eval_results:
                logger.warning("Evaluation phase failed or returned no results")
        else:
            logger.info("Skipping evaluation phase")

        # Dataset analysis phase
        analysis_results = {}
        if not skip_analysis:
            analysis_results = self.run_dataset_analysis()
            if not analysis_results:
                logger.warning("Dataset analysis failed or returned no results")
        else:
            logger.info("Skipping dataset analysis phase")

        # Demo phase
        if not skip_demo and pipeline_success:
            demo_success = self.run_demo()
            if not demo_success:
                logger.warning("Demo phase failed")
        else:
            logger.info("Skipping demo phase")

        # Generate final report
        try:
            report = self.generate_pipeline_report(eval_results, analysis_results)
            print("\n" + "="*60)
            print("PIPELINE SUMMARY")
            print("="*60)

            if eval_results and 'verification' in eval_results:
                ver_acc = eval_results['verification'].get('accuracy', 0)
                print(f"🎯 Verification Accuracy: {ver_acc:.1%}")

            if eval_results and 'identification' in eval_results:
                rank1_acc = eval_results['identification'].get('rank_accuracies', {}).get('rank_1', 0)
                print(f"🏆 Rank-1 Identification: {rank1_acc:.1%}")

            if analysis_results and 'persons' in analysis_results:
                num_persons = len(analysis_results['persons'])
                print(f"👥 Persons in Dataset: {num_persons}")

            total_time = (time.time() - self.pipeline_start_time) / 60
            print(f"⏱️  Total Pipeline Time: {total_time:.1f} minutes")
            print(f"📁 Results Directory: {self.pipeline_dir}")
            print("="*60)

        except Exception as e:
            logger.error(f"Error generating final report: {e}")

        if pipeline_success:
            logger.info("🎉 Pipeline completed successfully!")
        else:
            logger.error("❌ Pipeline completed with errors")

        return pipeline_success

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Face Recognition Pipeline Runner')

    parser.add_argument('--config', type=str, default='config.json',
                        help='Configuration file path')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training phase')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip evaluation phase')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip dataset analysis phase')
    parser.add_argument('--skip-demo', action='store_true',
                        help='Skip demo phase')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force retraining even if model exists')
    parser.add_argument('--training-only', action='store_true',
                        help='Run training only')
    parser.add_argument('--evaluation-only', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Run dataset analysis only')
    parser.add_argument('--demo-only', action='store_true',
                        help='Run demo only')

    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()

    # Initialize pipeline
    pipeline = FaceRecognitionPipeline(config_file=args.config)

    # Determine what to run
    if args.training_only:
        success = pipeline.check_prerequisites() and pipeline.run_training(force_retrain=args.force_retrain)
    elif args.evaluation_only:
        success = pipeline.check_prerequisites() and bool(pipeline.run_evaluation())
    elif args.analysis_only:
        success = bool(pipeline.run_dataset_analysis())
    elif args.demo_only:
        success = pipeline.run_demo()
    else:
        # Run full pipeline
        success = pipeline.run_full_pipeline(
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            skip_analysis=args.skip_analysis,
            skip_demo=args.skip_demo,
            force_retrain=args.force_retrain
        )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
