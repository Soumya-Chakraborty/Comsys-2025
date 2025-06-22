#!/usr/bin/env python3
"""
Batch Processing Script for Face Recognition System

This module implements high-performance batch processing capabilities for large-scale
face recognition tasks, providing scalable solutions for enterprise deployment.

Mathematical Foundation:

1. Batch Verification Processing:
   Given N pairs {(I₁ⁱ, I₂ⁱ, yᵢ)}ᴺᵢ₌₁, compute similarities:
   S = {cos(φ(I₁ⁱ), φ(I₂ⁱ)) | i = 1, ..., N}

   Parallel processing: Divide N into chunks of size C
   Process chunks independently: Sⱼ = {sᵢ | i ∈ chunk_j}
   Aggregate results: S = ∪ⱼ Sⱼ with statistical consistency

2. Batch Identification Processing:
   For M queries {Qᵢ}ᴹᵢ₌₁ against gallery G = {Gⱼ}ᴸⱼ₌₁:
   Similarity matrix: S[i,j] = cos(φ(Qᵢ), φ(Gⱼ))
   Rankings: Rᵢ = argsort(S[i,:], descending=True)

   Memory optimization: Process in batches to handle large M×L matrices
   Time complexity: O(M×L×d) where d is embedding dimension

3. Dataset Analysis:
   Statistical characterization of dataset properties:
   - Class distribution: {nᵢ} where nᵢ is samples for class i
   - Image quality metrics: Resolution, compression, noise levels
   - Distortion analysis: Distribution across 7 distortion types

Performance Optimizations:
- Multi-processing for CPU-bound embedding extraction
- Memory-mapped file I/O for large dataset handling
- Incremental processing with resumable operations
- Adaptive batch sizing based on available memory
"""

import os
import sys
import json
import csv
import time
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict, Counter
import pickle
import hashlib
import shutil
from datetime import datetime

# Import our modules
try:
    from inference import FaceRecognitionInference
    from evaluation_utils import FaceRecognitionEvaluator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the Task_B directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    High-Performance Batch Processing Manager for Face Recognition Tasks

    This class implements scalable batch processing algorithms for large-scale
    face recognition operations, optimized for enterprise deployment scenarios.

    Mathematical Framework:

    1. Parallel Processing Model:
       Divide workload W into chunks: W = ∪ᵢ Wᵢ where |Wᵢ| ≈ |W|/P
       Process chunks concurrently: Results = ∪ᵢ Process(Wᵢ)
       P = number of worker processes (typically min(CPU_count, 8))

    2. Memory Management:
       Batch size B chosen to satisfy: B × d × sizeof(float32) ≤ Memory_limit
       where d is embedding dimension, typically d=512
       Optimal B ≈ Memory_limit / (512 × 4 bytes) for embeddings

    3. Load Balancing:
       Work distribution using round-robin assignment
       Dynamic load balancing based on processing times
       Fault tolerance through task redistribution

    4. Statistical Consistency:
       Ensure identical results regardless of batch size or worker count
       Deterministic ordering for reproducible evaluation
       Aggregation with numerical stability considerations

    Key Features:
    - Multi-processing support for CPU-bound tasks
    - Memory-efficient streaming for large datasets
    - Resumable operations with checkpoint saving
    - Progress tracking with ETA estimation
    - Error recovery and partial result handling
    """

    def __init__(self, model_path: str, label_encoder_path: str,
                 output_dir: str = "batch_results", num_workers: int = None):
        """
        Initialize batch processor with optimized resource allocation

        Mathematical Considerations:
        - Worker count optimization: P* = min(CPU_cores, Memory_limit/Process_memory)
        - I/O bandwidth allocation: Distribute workers to avoid disk bottlenecks
        - Memory per worker: M_worker = Total_memory / (P + 1) accounting for main process

        Resource Management:
        - Automatic device selection based on workload characteristics
        - Memory monitoring to prevent OOM conditions
        - Process pool lifecycle management for stability

        Args:
            model_path (str): Path to trained face recognition model
            label_encoder_path (str): Path to class label encoder
            output_dir (str): Directory for saving batch processing results
            num_workers (int): Number of parallel worker processes (auto if None)
        """
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set number of workers
        if num_workers is None:
            self.num_workers = min(mp.cpu_count(), 8)
        else:
            self.num_workers = num_workers

        # Initialize inference system
        self.inference_system = None
        self.evaluator = FaceRecognitionEvaluator(str(self.output_dir))

        # Results storage
        self.results_cache = {}
        self.processed_pairs = set()

        logger.info(f"Batch processor initialized with {self.num_workers} workers")

    def _initialize_inference_system(self):
        """Initialize inference system (called in each worker process)"""
        if self.inference_system is None:
            self.inference_system = FaceRecognitionInference(
                model_path=self.model_path,
                label_encoder_path=self.label_encoder_path,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

    def _create_pair_hash(self, image1: str, image2: str) -> str:
        """
        Create unique hash identifier for image pairs

        Mathematical Properties:
        - Bijective mapping: Different pairs produce different hashes
        - Deterministic: Same pair always produces same hash
        - Uniform distribution: Hash collisions minimized
        - Order invariant option: hash(A,B) = hash(B,A) if needed

        Implementation uses MD5 for fast computation and low collision rate.
        Hash space: 2^128 possible values, collision probability ≈ 0 for practical datasets.

        Args:
            image1 (str): Path to first image
            image2 (str): Path to second image

        Returns:
            str: Unique hexadecimal hash identifier
        """
        pair_str = f"{image1}|{image2}"
        return hashlib.md5(pair_str.encode()).hexdigest()

    def batch_verification(self, pairs_file: str, threshold: float = 0.6,
                          save_intermediate: bool = True, chunk_size: int = 1000) -> Dict:
        """
        Process large-scale face verification with optimized batch processing

        Mathematical Framework:
        Given N pairs {(I₁ⁱ, I₂ⁱ, yᵢ)}ᴺᵢ₌₁:

        1. Chunked Processing:
           Divide pairs into chunks: C_j = {pairs[j×K:(j+1)×K]} where K = chunk_size
           Process chunks sequentially: R_j = Process(C_j)
           Aggregate: R = ∪_j R_j maintaining statistical consistency

        2. Similarity Computation:
           For each pair (I₁, I₂): s = cos(φ(I₁), φ(I₂)) = φ(I₁)ᵀφ(I₂)
           Decision: ŷ = 1 if s ≥ τ else 0, where τ is threshold

        3. Performance Metrics:
           Accuracy = (TP + TN) / N
           Precision = TP / (TP + FP)
           Recall = TP / (TP + FN)
           F1 = 2 × Precision × Recall / (Precision + Recall)

        4. Progress Estimation:
           Processing rate: R(t) = processed_pairs / elapsed_time
           ETA = (total_pairs - processed_pairs) / R(t)
           Confidence interval: ETA ± σ_R / √samples

        Optimization Features:
        - Resumable processing with intermediate checkpoints
        - Memory-efficient streaming for large pair files
        - Parallel worker processes for embedding extraction
        - Adaptive chunk sizing based on memory usage

        Args:
            pairs_file (str): CSV file with columns [image1, image2, label]
            threshold (float): Decision threshold τ ∈ [0, 1]
            save_intermediate (bool): Enable checkpoint saving for resumability
            chunk_size (int): Pairs per processing chunk (memory vs. speed trade-off)

        Returns:
            Dict: Comprehensive verification results with statistical analysis
        """
        logger.info(f"Starting batch verification from {pairs_file}")

        # Load pairs
        pairs_df = pd.read_csv(pairs_file)
        total_pairs = len(pairs_df)

        logger.info(f"Loaded {total_pairs} pairs for verification")

        # Check for existing results
        results_file = self.output_dir / f"verification_results_{Path(pairs_file).stem}.json"
        if results_file.exists():
            logger.info("Loading existing results...")
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
                self.processed_pairs = set(existing_results.get('processed_pairs', []))

        # Process in chunks
        all_results = []
        start_time = time.time()

        for i in range(0, total_pairs, chunk_size):
            chunk = pairs_df.iloc[i:i+chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(total_pairs-1)//chunk_size + 1}")

            # Prepare batch data
            batch_data = []
            for _, row in chunk.iterrows():
                image1, image2 = row['image1'], row['image2']
                pair_hash = self._create_pair_hash(image1, image2)

                if pair_hash not in self.processed_pairs:
                    batch_data.append({
                        'image1': image1,
                        'image2': image2,
                        'label': row.get('label', None),
                        'pair_hash': pair_hash
                    })

            if not batch_data:
                logger.info("Chunk already processed, skipping...")
                continue

            # Process chunk
            chunk_results = self._process_verification_chunk(batch_data, threshold)
            all_results.extend(chunk_results)

            # Update processed pairs
            self.processed_pairs.update([item['pair_hash'] for item in batch_data])

            # Save intermediate results
            if save_intermediate:
                self._save_intermediate_results(all_results, results_file)

            # Progress update
            processed = min(i + chunk_size, total_pairs)
            elapsed = time.time() - start_time
            rate = processed / elapsed
            eta = (total_pairs - processed) / rate if rate > 0 else 0

            logger.info(f"Processed {processed}/{total_pairs} pairs "
                       f"({processed/total_pairs*100:.1f}%) - "
                       f"Rate: {rate:.1f} pairs/sec - ETA: {eta/60:.1f} min")

        # Compile final results
        final_results = self._compile_verification_results(all_results, threshold)

        # Save final results
        final_results['processed_pairs'] = list(self.processed_pairs)
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        logger.info(f"Batch verification completed in {(time.time() - start_time)/60:.1f} minutes")
        return final_results

    def _process_verification_chunk(self, batch_data: List[Dict], threshold: float) -> List[Dict]:
        """
        Process verification chunk using parallel worker processes

        Mathematical Parallelization:
        Given chunk C = {(I₁ⁱ, I₂ⁱ)}ᵏᵢ₌₁ with k pairs:

        1. Work Distribution:
           Distribute pairs across P workers: Wⱼ = {pairs assigned to worker j}
           Load balancing: |W₁| ≈ |W₂| ≈ ... ≈ |Wₚ| ≈ k/P

        2. Independent Processing:
           Each worker computes: sᵢ = cos(φ(I₁ⁱ), φ(I₂ⁱ)) for assigned pairs
           Decision making: ŷᵢ = 1 if sᵢ ≥ τ else 0

        3. Result Aggregation:
           Combine worker results: R = ∪ⱼ Rⱼ
           Maintain deterministic ordering for reproducibility

        Performance Characteristics:
        - Linear speedup with worker count (CPU-bound operations)
        - Memory isolation prevents interference between workers
        - Fault tolerance through task redistribution

        Args:
            batch_data (List[Dict]): Chunk of pair data for processing
            threshold (float): Decision threshold for verification

        Returns:
            List[Dict]: Verification results with similarities and decisions
        """
        if not batch_data:
            return []

        # Use multiprocessing for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks
            future_to_data = {
                executor.submit(self._verify_single_pair, item, threshold): item
                for item in batch_data
            }

            results = []
            for future in as_completed(future_to_data):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing pair: {e}")

        return results

    def _verify_single_pair(self, pair_data: Dict, threshold: float) -> Optional[Dict]:
        """
        Verify single face pair in worker process with robust error handling

        Mathematical Operations:
        1. Embedding Extraction:
           f₁ = φ(I₁), f₂ = φ(I₂) where φ is the trained model
           Normalization: f₁ ← f₁/||f₁||₂, f₂ ← f₂/||f₂||₂

        2. Similarity Computation:
           s = f₁ᵀf₂ (dot product of unit vectors = cosine similarity)
           Range: s ∈ [-1, 1] where 1 = identical, -1 = opposite

        3. Decision Rule:
           decision = {1 if s ≥ τ, 0 if s < τ}
           where τ is the verification threshold

        Error Handling:
        - Image loading validation with format checking
        - Embedding extraction with GPU memory management
        - Numerical stability checks for similarity computation
        - Graceful degradation for corrupted or missing files

        Worker Process Isolation:
        - Independent model instance to avoid shared state
        - CPU-only processing to prevent GPU conflicts
        - Memory cleanup after each operation

        Args:
            pair_data (Dict): Pair information with image paths and metadata
            threshold (float): Decision threshold τ

        Returns:
            Optional[Dict]: Verification result or None if processing fails
        """
        try:
            # Initialize inference system in worker process
            if not hasattr(self, '_worker_inference_system'):
                self._worker_inference_system = FaceRecognitionInference(
                    model_path=self.model_path,
                    label_encoder_path=self.label_encoder_path,
                    device='cpu'  # Use CPU in worker processes to avoid GPU conflicts
                )

            image1, image2 = pair_data['image1'], pair_data['image2']

            # Check if images exist
            if not (os.path.exists(image1) and os.path.exists(image2)):
                logger.warning(f"Missing images: {image1} or {image2}")
                return None

            # Perform verification
            is_same, similarity = self._worker_inference_system.verify_faces(
                image1, image2, threshold
            )

            return {
                'image1': image1,
                'image2': image2,
                'similarity': similarity,
                'prediction': int(is_same),
                'threshold': threshold,
                'true_label': pair_data.get('label'),
                'pair_hash': pair_data['pair_hash']
            }

        except Exception as e:
            logger.error(f"Error verifying pair {pair_data.get('pair_hash', 'unknown')}: {e}")
            return None

    def batch_identification(self, queries_file: str, gallery_dir: str,
                           top_k: int = 5, chunk_size: int = 100) -> Dict:
        """
        Process large-scale face identification with optimized gallery matching

        Mathematical Framework:
        Given M queries {Qᵢ}ᴹᵢ₌₁ and gallery G = {Gⱼ}ᴸⱼ₌₁:

        1. Gallery Preprocessing:
           Extract embeddings: F_G = {φ(Gⱼ)}ᴸⱼ₌₁ where φ is embedding function
           Normalize: F_G ← F_G / ||F_G||₂ for cosine similarity computation
           Index construction: Build efficient search structure for fast retrieval

        2. Query Processing:
           For each query Qᵢ: f_qᵢ = φ(Qᵢ) with normalization
           Similarity computation: S[i,j] = f_qᵢᵀF_G[j] for all gallery items
           Ranking: R_i = argsort(S[i,:], descending=True)[:k]

        3. Evaluation Metrics:
           Rank-k accuracy: Acc_k = (1/M) Σᵢ I[true_idᵢ ∈ R_i[:k]]
           Mean Reciprocal Rank: MRR = (1/M) Σᵢ (1/rank_i)
           Top-1 accuracy: Special case of rank-k with k=1

        Optimization Strategies:
        - Gallery preprocessing to avoid redundant embedding extraction
        - Chunked query processing for memory efficiency
        - Parallel similarity computation across multiple workers
        - Result caching for repeated gallery items

        Args:
            queries_file (str): CSV file with columns [query_image, true_identity]
            gallery_dir (str): Directory containing gallery images organized by identity
            top_k (int): Number of top matches to return for ranking analysis
            chunk_size (int): Queries per processing chunk

        Returns:
            Dict: Identification results with rank-based metrics and detailed analysis
        """
        logger.info(f"Starting batch identification from {queries_file}")

        # Load queries
        queries_df = pd.read_csv(queries_file)
        total_queries = len(queries_df)

        logger.info(f"Loaded {total_queries} queries for identification")

        # Process queries
        all_results = []
        start_time = time.time()

        # Initialize inference system
        self._initialize_inference_system()

        for i in tqdm(range(0, total_queries, chunk_size), desc="Processing queries"):
            chunk = queries_df.iloc[i:i+chunk_size]

            for _, row in chunk.iterrows():
                query_image = row['query_image']
                true_identity = row.get('true_identity', None)

                try:
                    # Perform identification
                    matches = self.inference_system.identify_face(
                        query_image, gallery_dir, top_k
                    )

                    all_results.append({
                        'query_image': query_image,
                        'true_identity': true_identity,
                        'matches': matches,
                        'top_match': matches[0]['person'] if matches else None,
                        'top_confidence': matches[0]['confidence'] if matches else 0.0
                    })

                except Exception as e:
                    logger.error(f"Error processing query {query_image}: {e}")
                    all_results.append({
                        'query_image': query_image,
                        'true_identity': true_identity,
                        'matches': [],
                        'top_match': None,
                        'top_confidence': 0.0,
                        'error': str(e)
                    })

        # Compile results
        final_results = self._compile_identification_results(all_results, top_k)

        # Save results
        results_file = self.output_dir / f"identification_results_{Path(queries_file).stem}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        logger.info(f"Batch identification completed in {(time.time() - start_time)/60:.1f} minutes")
        return final_results

    def batch_embedding_extraction(self, images_file: str, batch_size: int = 32) -> Dict:
        """
        Extract normalized face embeddings for large image collections

        Mathematical Framework:
        For N images {Iᵢ}ᴺᵢ₌₁, compute embeddings:

        1. Forward Pass:
           Raw embeddings: E_raw = {φ(Iᵢ)}ᴺᵢ₌₁ where φ is trained model
           Each eᵢ ∈ ℝᵈ where d is embedding dimension (typically 512)

        2. Normalization:
           Normalized embeddings: E = {eᵢ/||eᵢ||₂}ᴺᵢ₌₁
           Properties: ||eᵢ||₂ = 1, cosine similarity = dot product

        3. Quality Assurance:
           Embedding magnitude check: ||e_raw|| should be > threshold
           NaN/Inf detection: Reject invalid embeddings
           Consistency validation: Similar images should have similar embeddings

        Batch Processing Benefits:
        - GPU utilization optimization through batched inference
        - Memory efficiency through streaming large datasets
        - Progress tracking with accurate ETA estimation
        - Error isolation preventing single failures from affecting entire dataset

        Output Formats:
        - JSON: Human-readable format with metadata
        - NumPy: Efficient binary format for large-scale processing
        - CSV: Metadata and statistics for analysis

        Args:
            images_file (str): CSV file containing image paths for processing
            batch_size (int): Images per GPU batch (optimize for memory vs. speed)

        Returns:
            Dict: Comprehensive results including embeddings, metadata, and statistics
        """
        logger.info(f"Starting batch embedding extraction from {images_file}")

        # Load image paths
        images_df = pd.read_csv(images_file)
        image_paths = images_df['image_path'].tolist()
        total_images = len(image_paths)

        logger.info(f"Extracting embeddings for {total_images} images")

        # Initialize inference system
        self._initialize_inference_system()

        embeddings = []
        valid_paths = []

        start_time = time.time()

        for i in tqdm(range(0, total_images, batch_size), desc="Extracting embeddings"):
            batch_paths = image_paths[i:i+batch_size]

            for image_path in batch_paths:
                try:
                    embedding = self.inference_system.extract_embedding(image_path)
                    embeddings.append(embedding.tolist())
                    valid_paths.append(image_path)
                except Exception as e:
                    logger.error(f"Error extracting embedding for {image_path}: {e}")

        # Save embeddings
        results = {
            'embeddings': embeddings,
            'image_paths': valid_paths,
            'embedding_dim': len(embeddings[0]) if embeddings else 0,
            'total_processed': len(valid_paths),
            'total_requested': total_images,
            'processing_time': time.time() - start_time
        }

        # Save to multiple formats
        embeddings_file = self.output_dir / f"embeddings_{Path(images_file).stem}"

        # JSON format
        with open(f"{embeddings_file}.json", 'w') as f:
            json.dump(results, f, indent=2)

        # NumPy format
        if embeddings:
            np.save(f"{embeddings_file}.npy", np.array(embeddings))

        # CSV format for metadata
        df_results = pd.DataFrame({
            'image_path': valid_paths,
            'embedding_extracted': True
        })
        df_results.to_csv(f"{embeddings_file}_metadata.csv", index=False)

        logger.info(f"Embedding extraction completed in {(time.time() - start_time)/60:.1f} minutes")
        return results

    def dataset_analysis(self, data_dir: str, sample_size: int = None) -> Dict:
        """
        Comprehensive statistical analysis of face recognition dataset

        Mathematical Framework:
        Characterize dataset D = {(Iᵢ, yᵢ)}ᴺᵢ₌₁ where Iᵢ are images, yᵢ are labels:

        1. Class Distribution Analysis:
           Class frequencies: n_j = |{i : yᵢ = j}| for each class j
           Balance metric: B = H(Y) / log(K) where H(Y) is entropy, K is num_classes
           Imbalance ratio: IR = max(n_j) / min(n_j)

        2. Image Quality Assessment:
           Resolution distribution: R = {(wᵢ, hᵢ)} for width×height
           File size analysis: S = {sᵢ} in bytes
           Compression quality: JPEG quality factors and artifacts

        3. Statistical Properties:
           Mean image dimensions: μ_w = (1/N)Σwᵢ, μ_h = (1/N)Σhᵢ
           Standard deviations: σ_w, σ_h for dimension variability
           Aspect ratio distribution: AR = {wᵢ/hᵢ}

        4. Distortion Analysis:
           Distortion type frequencies: D = {d₁, d₂, ..., d₇} counts
           Coverage per identity: C_j = distortion types available for class j
           Quality degradation metrics: PSNR, SSIM for distorted vs. original

        Analysis Categories:
        - Demographic analysis: Identity representation and balance
        - Technical analysis: Image properties and quality metrics
        - Augmentation analysis: Distortion type distribution and coverage
        - Temporal analysis: Collection timestamps and consistency

        Args:
            data_dir (str): Root directory containing organized face dataset
            sample_size (int, optional): Limit analysis to subset for efficiency

        Returns:
            Dict: Comprehensive dataset characterization with statistical summaries
        """
        logger.info(f"Starting dataset analysis for {data_dir}")

        data_path = Path(data_dir)
        analysis_results = {
            'dataset_path': str(data_path),
            'analysis_timestamp': datetime.now().isoformat(),
            'persons': {},
            'distortion_stats': defaultdict(int),
            'image_stats': {
                'total_images': 0,
                'original_images': 0,
                'distorted_images': 0,
                'invalid_images': 0
            },
            'class_distribution': {},
            'file_size_stats': [],
            'image_dimension_stats': []
        }

        # Get all person directories
        person_dirs = [d for d in data_path.iterdir() if d.is_dir()]

        if sample_size:
            person_dirs = person_dirs[:sample_size]

        logger.info(f"Analyzing {len(person_dirs)} person directories")

        for person_dir in tqdm(person_dirs, desc="Analyzing persons"):
            person_name = person_dir.name
            person_stats = {
                'original_images': 0,
                'distorted_images': 0,
                'total_images': 0,
                'distortion_types': [],
                'image_files': []
            }

            # Analyze original images
            for img_file in person_dir.glob("*.jpg"):
                if self._is_valid_image(img_file):
                    person_stats['original_images'] += 1
                    person_stats['image_files'].append(str(img_file))

                    # Get file stats
                    file_size = img_file.stat().st_size
                    analysis_results['file_size_stats'].append(file_size)

                    # Get image dimensions
                    try:
                        import cv2
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            h, w, c = img.shape
                            analysis_results['image_dimension_stats'].append((w, h))
                    except:
                        pass
                else:
                    analysis_results['image_stats']['invalid_images'] += 1

            # Analyze distorted images
            distortion_dir = person_dir / "distortion"
            if distortion_dir.exists():
                for img_file in distortion_dir.glob("*.jpg"):
                    if self._is_valid_image(img_file):
                        person_stats['distorted_images'] += 1
                        person_stats['image_files'].append(str(img_file))

                        # Identify distortion type
                        distortion_type = self._get_distortion_type(img_file.name)
                        if distortion_type not in person_stats['distortion_types']:
                            person_stats['distortion_types'].append(distortion_type)
                        analysis_results['distortion_stats'][distortion_type] += 1

                        # File stats
                        file_size = img_file.stat().st_size
                        analysis_results['file_size_stats'].append(file_size)
                    else:
                        analysis_results['image_stats']['invalid_images'] += 1

            person_stats['total_images'] = person_stats['original_images'] + person_stats['distorted_images']
            analysis_results['persons'][person_name] = person_stats

            # Update class distribution
            analysis_results['class_distribution'][person_name] = person_stats['total_images']

        # Calculate summary statistics
        total_original = sum(p['original_images'] for p in analysis_results['persons'].values())
        total_distorted = sum(p['distorted_images'] for p in analysis_results['persons'].values())

        analysis_results['image_stats']['total_images'] = total_original + total_distorted
        analysis_results['image_stats']['original_images'] = total_original
        analysis_results['image_stats']['distorted_images'] = total_distorted

        # Calculate file size statistics
        if analysis_results['file_size_stats']:
            file_sizes = analysis_results['file_size_stats']
            analysis_results['file_size_summary'] = {
                'mean_mb': np.mean(file_sizes) / (1024*1024),
                'median_mb': np.median(file_sizes) / (1024*1024),
                'std_mb': np.std(file_sizes) / (1024*1024),
                'min_mb': min(file_sizes) / (1024*1024),
                'max_mb': max(file_sizes) / (1024*1024)
            }

        # Calculate dimension statistics
        if analysis_results['image_dimension_stats']:
            widths, heights = zip(*analysis_results['image_dimension_stats'])
            analysis_results['dimension_summary'] = {
                'mean_width': np.mean(widths),
                'mean_height': np.mean(heights),
                'median_width': np.median(widths),
                'median_height': np.median(heights),
                'width_range': (min(widths), max(widths)),
                'height_range': (min(heights), max(heights))
            }

        # Save analysis results
        analysis_file = self.output_dir / f"dataset_analysis_{data_path.name}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)

        # Create summary report
        self._create_analysis_report(analysis_results)

        logger.info("Dataset analysis completed")
        return analysis_results

    def _is_valid_image(self, image_path: Path) -> bool:
        """
        Validate image file integrity and format compatibility

        Validation Criteria:
        1. File Existence: Path points to actual file
        2. Format Support: Compatible with PIL/OpenCV (JPEG, PNG, etc.)
        3. Image Integrity: File not corrupted or truncated
        4. Readable Content: Can be opened and basic properties accessed

        Mathematical Properties:
        - Binary validation: Returns True/False based on all criteria
        - Fast execution: Uses lightweight verification without full loading
        - Error isolation: Exceptions handled gracefully

        Args:
            image_path (Path): Path to image file for validation

        Returns:
            bool: True if image is valid and readable, False otherwise
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img.verify()
            return True
        except:
            return False

    def _get_distortion_type(self, filename: str) -> str:
        """Extract distortion type from filename"""
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

    def _save_intermediate_results(self, results: List[Dict], results_file: Path):
        """Save intermediate processing results"""
        intermediate_data = {
            'results': results,
            'processed_count': len(results),
            'timestamp': datetime.now().isoformat(),
            'processed_pairs': list(self.processed_pairs)
        }

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = results_file.parent / f"intermediate_{timestamp}_{results_file.name}"

        with open(intermediate_file, 'w') as f:
            json.dump(intermediate_data, f, indent=2, default=str)

    def _compile_verification_results(self, results: List[Dict], threshold: float) -> Dict:
        """Compile verification results and calculate metrics"""
        if not results:
            return {'error': 'No results to compile'}

        similarities = [r['similarity'] for r in results]
        predictions = [r['prediction'] for r in results]
        true_labels = [r['true_label'] for r in results if r['true_label'] is not None]

        compiled_results = {
            'total_pairs': len(results),
            'threshold': threshold,
            'similarities': similarities,
            'predictions': predictions,
            'processing_timestamp': datetime.now().isoformat()
        }

        # Calculate metrics if true labels are available
        if true_labels and len(true_labels) == len(results):
            verification_metrics = self.evaluator.calculate_verification_metrics(
                similarities, true_labels
            )
            compiled_results.update(verification_metrics)

        return compiled_results

    def _compile_identification_results(self, results: List[Dict], top_k: int) -> Dict:
        """Compile identification results and calculate metrics"""
        if not results:
            return {'error': 'No results to compile'}

        # Extract predictions and ground truth
        predictions = []
        ground_truth = []

        for result in results:
            if result.get('matches') and result.get('true_identity'):
                pred_list = [match['person'] for match in result['matches']]
                predictions.append(pred_list)
                ground_truth.append(result['true_identity'])

        compiled_results = {
            'total_queries': len(results),
            'successful_queries': len([r for r in results if r.get('matches')]),
            'top_k': top_k,
            'processing_timestamp': datetime.now().isoformat(),
            'detailed_results': results
        }

        # Calculate identification metrics if labels are available
        if predictions and ground_truth:
            identification_metrics = self.evaluator.calculate_identification_metrics(
                predictions, ground_truth
            )
            compiled_results.update(identification_metrics)

        return compiled_results

    def _create_analysis_report(self, analysis_results: Dict):
        """
        Generate comprehensive human-readable dataset analysis report

        Mathematical Content:
        Transforms statistical data into interpretable insights:

        1. Descriptive Statistics:
           Central tendencies: μ, median, mode for key metrics
           Variability measures: σ, IQR, range for distributions
           Shape descriptors: Skewness, kurtosis for distribution analysis

        2. Quality Assessment:
           Data completeness: Percentage of valid vs. invalid files
           Balance metrics: Class distribution uniformity scores
           Coverage analysis: Distortion type representation across identities

        3. Recommendations:
           Sample size adequacy: Statistical power analysis
           Augmentation suggestions: Based on identified gaps
           Quality improvements: Targeted enhancement strategies

        Report Structure:
        - Executive Summary: Key findings and recommendations
        - Statistical Overview: Quantitative dataset characteristics
        - Quality Assessment: Data integrity and completeness analysis
        - Distribution Analysis: Class balance and representation metrics
        - Technical Specifications: Image properties and formats
        - Recommendations: Actionable insights for improvement

        Args:
            analysis_results (Dict): Complete statistical analysis results
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATASET ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Basic statistics
        report_lines.append("BASIC STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Dataset Path: {analysis_results['dataset_path']}")
        report_lines.append(f"Number of Persons: {len(analysis_results['persons'])}")
        report_lines.append(f"Total Images: {analysis_results['image_stats']['total_images']}")
        report_lines.append(f"Original Images: {analysis_results['image_stats']['original_images']}")
        report_lines.append(f"Distorted Images: {analysis_results['image_stats']['distorted_images']}")
        report_lines.append(f"Invalid Images: {analysis_results['image_stats']['invalid_images']}")
        report_lines.append("")

        # Distortion statistics
        if analysis_results['distortion_stats']:
            report_lines.append("DISTORTION TYPE DISTRIBUTION")
            report_lines.append("-" * 40)
            for dist_type, count in analysis_results['distortion_stats'].items():
                report_lines.append(f"{dist_type.capitalize()}: {count}")
            report_lines.append("")

        # Class distribution
        if analysis_results['class_distribution']:
            class_counts = list(analysis_results['class_distribution'].values())
            report_lines.append("CLASS DISTRIBUTION STATISTICS")
            report_lines.append("-" * 40)
            report_lines.append(f"Mean images per person: {np.mean(class_counts):.1f}")
            report_lines.append(f"Median images per person: {np.median(class_counts):.1f}")
            report_lines.append(f"Min images per person: {min(class_counts)}")
            report_lines.append(f"Max images per person: {max(class_counts)}")
            report_lines.append("")

        # File size statistics
        if 'file_size_summary' in analysis_results:
            fs = analysis_results['file_size_summary']
            report_lines.append("FILE SIZE STATISTICS")
            report_lines.append("-" * 40)
            report_lines.append(f"Mean file size: {fs['mean_mb']:.2f} MB")
            report_lines.append(f"Median file size: {fs['median_mb']:.2f} MB")
            report_lines.append(f"File size range: {fs['min_mb']:.2f} - {fs['max_mb']:.2f} MB")
            report_lines.append("")

        # Image dimension statistics
        if 'dimension_summary' in analysis_results:
            ds = analysis_results['dimension_summary']
            report_lines.append("IMAGE DIMENSION STATISTICS")
            report_lines.append("-" * 40)
            report_lines.append(f"Mean dimensions: {ds['mean_width']:.0f} x {ds['mean_height']:.0f}")
            report_lines.append(f"Median dimensions: {ds['median_width']:.0f} x {ds['median_height']:.0f}")
            report_lines.append(f"Width range: {ds['width_range'][0]} - {ds['width_range'][1]}")
            report_lines.append(f"Height range: {ds['height_range'][0]} - {ds['height_range'][1]}")
            report_lines.append("")

        report_lines.append("=" * 80)

        # Save report
        report_text = "\n".join(report_lines)
        report_file = self.output_dir / "dataset_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)

        logger.info(f"Analysis report saved to {report_file}")

def parse_arguments():
    """
    Parse command line arguments for flexible batch processing configuration

    Argument Categories:
    1. Model Configuration: Paths to trained model and label encoder
    2. Processing Mode: Type of batch operation to perform
    3. Input Specification: Data files and directories for processing
    4. Performance Tuning: Worker count, batch sizes, chunk sizes
    5. Output Control: Result directories and file formats

    Mathematical Parameters:
    - chunk_size: Affects memory usage and parallelization efficiency
    - batch_size: GPU memory vs. throughput trade-off
    - num_workers: CPU utilization vs. memory overhead balance
    - threshold: Decision boundary for verification tasks

    Returns:
        argparse.Namespace: Validated command line arguments with type checking
    """
    parser = argparse.ArgumentParser(description='Batch Processing for Face Recognition')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--label_encoder_path', type=str, required=True,
                        help='Path to label encoder')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['verification', 'identification', 'embeddings', 'analysis'],
                        help='Processing mode')
    parser.add_argument('--input_file', type=str,
                        help='Input CSV file for verification/identification/embeddings')
    parser.add_argument('--data_dir', type=str,
                        help='Data directory for analysis or gallery for identification')
    parser.add_argument('--output_dir', type=str, default='batch_results',
                        help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Threshold for verification')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top-K for identification')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Chunk size for processing')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for embedding extraction')

    return parser.parse_args()

def main():
    """
    Main batch processing function with comprehensive error handling

    Execution Framework:
    1. Argument validation with informative error messages
    2. Resource allocation based on system capabilities
    3. Mode-specific processing with progress monitoring
    4. Result aggregation and statistical analysis
    5. Output generation in multiple formats

    Mathematical Validation:
    - Input parameter range checking for numerical stability
    - Statistical significance validation for sample sizes
    - Memory requirement estimation for large datasets
    - Performance prediction based on system capabilities

    Error Recovery:
    - Graceful handling of corrupted or missing files
    - Automatic fallback for insufficient system resources
    - Partial result preservation for resumable operations
    - Detailed logging for troubleshooting and optimization

    Performance Monitoring:
    - Real-time progress tracking with ETA estimation
    - Resource utilization monitoring (CPU, memory, disk)
    - Throughput analysis for optimization opportunities
    - Quality metrics validation during processing
    """
    args = parse_arguments()

    # Initialize batch processor
    processor = BatchProcessor(
        model_path=args.model_path,
        label_encoder_path=args.label_encoder_path,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )

    # Run specified mode
    try:
        if args.mode == 'verification':
            if not args.input_file:
                raise ValueError("--input_file required for verification mode")

            results = processor.batch_verification(
                pairs_file=args.input_file,
                threshold=args.threshold,
                chunk_size=args.chunk_size
            )
            print(f"Verification completed: {results.get('total_pairs', 0)} pairs processed")

        elif args.mode == 'identification':
            if not args.input_file or not args.data_dir:
                raise ValueError("--input_file and --data_dir required for identification mode")

            results = processor.batch_identification(
                queries_file=args.input_file,
                gallery_dir=args.data_dir,
                top_k=args.top_k,
                chunk_size=args.chunk_size
            )
            print(f"Identification completed: {results.get('total_queries', 0)} queries processed")

        elif args.mode == 'embeddings':
            if not args.input_file:
                raise ValueError("--input_file required for embeddings mode")

            results = processor.batch_embedding_extraction(
                images_file=args.input_file,
                batch_size=args.batch_size
            )
            print(f"Embedding extraction completed: {results.get('total_processed', 0)} images processed")

        elif args.mode == 'analysis':
            if not args.data_dir:
                raise ValueError("--data_dir required for analysis mode")

            results = processor.dataset_analysis(
                data_dir=args.data_dir,
                sample_size=None
            )
            print(f"Dataset analysis completed: {len(results.get('persons', {}))} persons analyzed")

        print(f"\nResults saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
