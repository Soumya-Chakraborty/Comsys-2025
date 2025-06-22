#!/bin/bash

# =============================================================================
# Gender Classification System - Task A Complete Runner
# =============================================================================
# This script provides a comprehensive way to run the complete Task A system
# with automatic environment setup, dependency management, and GPU optimization.
#
# Author: AI Assistant
# Date: 2024
#
# Usage:
#   ./run_task_a.sh                    # Run with default settings
#   ./run_task_a.sh --quick            # Quick training (reduced epochs)
#   ./run_task_a.sh --gpu-only         # Skip if no GPU available
#   ./run_task_a.sh --custom-config    # Use custom configuration
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PYTHON_MIN_VERSION="3.8"
LOG_FILE="$SCRIPT_DIR/setup_and_training.log"

# Default parameters
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=1e-4
OUTPUT_DIR="output"
SKIP_ENSEMBLE=false
SKIP_OPTIMIZATION=false
GPU_ONLY=false
QUICK_MODE=false
CUSTOM_CONFIG=false
CONFIG_FILE="config_template.json"

# =============================================================================
# Utility Functions
# =============================================================================

print_header() {
    echo -e "${CYAN}===============================================================================${NC}"
    echo -e "${CYAN}🚀 Gender Classification System - Task A Complete Training${NC}"
    echo -e "${CYAN}===============================================================================${NC}"
    echo ""
}

print_section() {
    echo -e "${BLUE}📋 $1${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS: $1" >> "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: $1" >> "$LOG_FILE"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: $1" >> "$LOG_FILE"
}

print_info() {
    echo -e "${PURPLE}ℹ️  $1${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO: $1" >> "$LOG_FILE"
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --epochs)
                EPOCHS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --lr|--learning-rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --skip-ensemble)
                SKIP_ENSEMBLE=true
                shift
                ;;
            --skip-optimization)
                SKIP_OPTIMIZATION=true
                shift
                ;;
            --gpu-only)
                GPU_ONLY=true
                shift
                ;;
            --quick)
                QUICK_MODE=true
                EPOCHS=20
                BATCH_SIZE=64
                shift
                ;;
            --custom-config)
                CUSTOM_CONFIG=true
                CONFIG_FILE="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    echo "Gender Classification System - Task A Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --epochs N              Number of training epochs (default: 50)"
    echo "  --batch-size N          Batch size for training (default: 32)"
    echo "  --lr, --learning-rate   Learning rate (default: 1e-4)"
    echo "  --output-dir DIR        Output directory (default: output)"
    echo "  --skip-ensemble         Skip ensemble training"
    echo "  --skip-optimization     Skip model optimization"
    echo "  --gpu-only              Exit if no GPU available"
    echo "  --quick                 Quick mode (20 epochs, batch size 64)"
    echo "  --custom-config FILE    Use custom configuration file"
    echo "  --config FILE           Configuration file to use"
    echo "  --help, -h              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Default training"
    echo "  $0 --quick                           # Quick training"
    echo "  $0 --epochs 100 --batch-size 64     # Custom parameters"
    echo "  $0 --gpu-only --skip-ensemble       # GPU-only, no ensemble"
}

# =============================================================================
# System Checks
# =============================================================================

check_python_version() {
    print_section "Checking Python Version"

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    print_info "Found Python $PYTHON_VERSION"

    # Compare versions
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_success "Python version is compatible"
    else
        print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
}

check_gpu_availability() {
    print_section "Checking GPU Availability"

    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "")
        if [[ -n "$GPU_INFO" ]]; then
            print_success "NVIDIA GPU detected:"
            echo "$GPU_INFO" | while read -r line; do
                print_info "  $line"
            done
            return 0
        fi
    fi

    print_warning "No NVIDIA GPU detected"
    if [[ "$GPU_ONLY" == true ]]; then
        print_error "GPU-only mode requested but no GPU available"
        exit 1
    fi

    print_info "Training will proceed on CPU (much slower)"
    return 1
}

check_disk_space() {
    print_section "Checking Disk Space"

    AVAILABLE_GB=$(df "$SCRIPT_DIR" | awk 'NR==2 {printf "%.1f", $4/1024/1024}')
    print_info "Available disk space: ${AVAILABLE_GB}GB"

    if (( $(echo "$AVAILABLE_GB < 5.0" | bc -l) )); then
        print_warning "Low disk space detected. Training may fail if space runs out."
    else
        print_success "Sufficient disk space available"
    fi
}

check_memory() {
    print_section "Checking System Memory"

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        TOTAL_MEM_GB=$(free -g | awk '/^Mem:/ {print $2}')
        AVAIL_MEM_GB=$(free -g | awk '/^Mem:/ {print $7}')
        print_info "Total memory: ${TOTAL_MEM_GB}GB, Available: ${AVAIL_MEM_GB}GB"

        if (( AVAIL_MEM_GB < 4 )); then
            print_warning "Low system memory. Consider reducing batch size."
            BATCH_SIZE=16
            print_info "Automatically reduced batch size to $BATCH_SIZE"
        else
            print_success "Sufficient system memory available"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        TOTAL_MEM_GB=$(system_profiler SPHardwareDataType | awk '/Memory:/ {print $2}')
        print_info "Total memory: ${TOTAL_MEM_GB}"
        print_success "Memory check completed"
    fi
}

# =============================================================================
# Environment Setup Functions
# =============================================================================

setup_virtual_environment() {
    print_section "Setting Up Virtual Environment"

    if [[ -d "$VENV_DIR" ]]; then
        print_info "Virtual environment already exists"
    else
        print_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    print_success "Pip upgraded"
}

install_pytorch() {
    print_section "Installing PyTorch"

    # Check if PyTorch is already installed
    if python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
        print_info "PyTorch already installed:"
        python3 -c "import torch; print('  Version:', torch.__version__)"
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            print_success "PyTorch with CUDA support detected"
            return 0
        else
            print_warning "PyTorch without CUDA detected, reinstalling with CUDA support"
        fi
    fi

    print_info "Installing PyTorch with CUDA support..."

    # Detect CUDA version if available
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        print_info "Detected CUDA version: $CUDA_VERSION"

        if [[ "$CUDA_VERSION" == "12."* ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$CUDA_VERSION" == "11."* ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            print_warning "Unsupported CUDA version, installing CPU version"
            pip install torch torchvision torchaudio
        fi
    else
        print_info "CUDA not detected, installing CPU version"
        pip install torch torchvision torchaudio
    fi

    print_success "PyTorch installation completed"
}

install_dependencies() {
    print_section "Installing Dependencies"

    if [[ ! -f "$SCRIPT_DIR/requirements.txt" ]]; then
        print_error "requirements.txt not found"
        exit 1
    fi

    print_info "Installing packages from requirements.txt..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
    print_success "Dependencies installed successfully"
}

verify_installation() {
    print_section "Verifying Installation"

    # Test basic imports
    python3 -c "
import torch
import torchvision
import timm
import transformers
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

print('✅ All core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
" || {
        print_error "Package verification failed"
        exit 1
    }

    print_success "Installation verification completed"
}

# =============================================================================
# Training Functions
# =============================================================================

prepare_training_environment() {
    print_section "Preparing Training Environment"

    # Create output directory
    mkdir -p "$SCRIPT_DIR/$OUTPUT_DIR"
    mkdir -p "$SCRIPT_DIR/$OUTPUT_DIR/models"
    mkdir -p "$SCRIPT_DIR/$OUTPUT_DIR/plots"
    mkdir -p "$SCRIPT_DIR/$OUTPUT_DIR/logs"

    # Check data directories
    if [[ ! -d "$SCRIPT_DIR/train" ]] || [[ ! -d "$SCRIPT_DIR/val" ]]; then
        print_error "Training data directories not found. Expected: train/ and val/"
        print_info "Please ensure the dataset is properly structured as described in README.md"
        exit 1
    fi

    # Count images
    TRAIN_FEMALE=$(find "$SCRIPT_DIR/train/female" -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l)
    TRAIN_MALE=$(find "$SCRIPT_DIR/train/male" -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l)
    VAL_FEMALE=$(find "$SCRIPT_DIR/val/female" -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l)
    VAL_MALE=$(find "$SCRIPT_DIR/val/male" -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l)

    print_info "Dataset statistics:"
    print_info "  Training: $TRAIN_FEMALE female, $TRAIN_MALE male"
    print_info "  Validation: $VAL_FEMALE female, $VAL_MALE male"

    print_success "Training environment prepared"
}

create_config_file() {
    print_section "Creating Configuration File"

    CONFIG_PATH="$SCRIPT_DIR/runtime_config.json"

    cat > "$CONFIG_PATH" << EOF
{
  "data": {
    "train_dir": "train",
    "val_dir": "val",
    "batch_size": $BATCH_SIZE,
    "num_workers": 4,
    "image_size": 224
  },
  "training": {
    "num_epochs": $EPOCHS,
    "learning_rate": $LEARNING_RATE,
    "weight_decay": 1e-4,
    "dropout_rate": 0.3,
    "warmup_epochs": 5
  },
  "models": {
    "train_single": true,
    "train_ensemble": $([ "$SKIP_ENSEMBLE" = true ] && echo "false" || echo "true"),
    "apply_distillation": $([ "$SKIP_OPTIMIZATION" = true ] && echo "false" || echo "true"),
    "apply_quantization": $([ "$SKIP_OPTIMIZATION" = true ] && echo "false" || echo "true")
  },
  "output": {
    "output_dir": "$OUTPUT_DIR",
    "save_plots": true,
    "save_logs": true
  }
}
EOF

    print_success "Configuration file created: $CONFIG_PATH"
}

run_training() {
    print_section "Starting Training Process"

    # Build training command
    TRAIN_CMD="python3 train_complete_system.py"
    TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
    TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
    TRAIN_CMD="$TRAIN_CMD --lr $LEARNING_RATE"
    TRAIN_CMD="$TRAIN_CMD --output-dir $OUTPUT_DIR"

    if [[ "$SKIP_ENSEMBLE" == true ]]; then
        TRAIN_CMD="$TRAIN_CMD --skip-ensemble"
    fi

    if [[ "$SKIP_OPTIMIZATION" == true ]]; then
        TRAIN_CMD="$TRAIN_CMD --skip-optimization"
    fi

    if [[ "$CUSTOM_CONFIG" == true ]] && [[ -f "$SCRIPT_DIR/$CONFIG_FILE" ]]; then
        TRAIN_CMD="$TRAIN_CMD --config $CONFIG_FILE"
    fi

    print_info "Training command: $TRAIN_CMD"
    print_info "Training started at: $(date)"

    # Run training with output to both console and log
    eval "$TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        print_success "Training completed successfully!"
    else
        print_error "Training failed!"
        exit 1
    fi
}

run_demo() {
    print_section "Running Demonstration"

    # Check if models exist
    if [[ -f "$SCRIPT_DIR/$OUTPUT_DIR/models/best_single_model.pth" ]]; then
        print_info "Running prediction demo with single model..."
        python3 demo_predictions.py \
            --model "$OUTPUT_DIR/models/best_single_model.pth" \
            --demo-mode \
            2>&1 | tee -a "$LOG_FILE"
    elif [[ -f "$SCRIPT_DIR/$OUTPUT_DIR/models/best_ensemble_model.pth" ]]; then
        print_info "Running prediction demo with ensemble model..."
        python3 demo_predictions.py \
            --model "$OUTPUT_DIR/models/best_ensemble_model.pth" \
            --demo-mode \
            2>&1 | tee -a "$LOG_FILE"
    else
        print_warning "No trained models found for demo"
    fi
}

# =============================================================================
# Cleanup and Summary Functions
# =============================================================================

show_results_summary() {
    print_section "Training Results Summary"

    if [[ -f "$SCRIPT_DIR/$OUTPUT_DIR/evaluation_results.json" ]]; then
        print_info "Loading evaluation results..."
        python3 -c "
import json
import sys
try:
    with open('$SCRIPT_DIR/$OUTPUT_DIR/evaluation_results.json', 'r') as f:
        results = json.load(f)

    print('📊 Model Performance:')
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f'  {model_name}: {acc:.2%} accuracy')

    print('')
    print('📁 Generated Files:')
    print('  Models: $OUTPUT_DIR/models/')
    print('  Plots: $OUTPUT_DIR/plots/')
    print('  Logs: $OUTPUT_DIR/logs/')

except Exception as e:
    print(f'Could not load results: {e}')
"
    else
        print_warning "No evaluation results file found"
    fi

    # Show file sizes
    if [[ -d "$SCRIPT_DIR/$OUTPUT_DIR/models" ]]; then
        print_info "Model files:"
        find "$SCRIPT_DIR/$OUTPUT_DIR/models" -name "*.pth" -exec ls -lh {} \; | awk '{print "  " $9 ": " $5}'
    fi
}

cleanup_on_exit() {
    print_section "Cleanup"

    # Deactivate virtual environment if active
    if [[ -n "$VIRTUAL_ENV" ]]; then
        deactivate
        print_info "Virtual environment deactivated"
    fi

    print_info "Training session completed at: $(date)"
    print_info "Full log available at: $LOG_FILE"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    # Initialize log file
    echo "=== Gender Classification System Training Log ===" > "$LOG_FILE"
    echo "Started at: $(date)" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    # Set trap for cleanup
    trap cleanup_on_exit EXIT

    # Print header
    print_header

    # Parse command line arguments
    parse_arguments "$@"

    # Show configuration
    print_info "Training Configuration:"
    print_info "  Epochs: $EPOCHS"
    print_info "  Batch Size: $BATCH_SIZE"
    print_info "  Learning Rate: $LEARNING_RATE"
    print_info "  Output Directory: $OUTPUT_DIR"
    print_info "  Skip Ensemble: $SKIP_ENSEMBLE"
    print_info "  Skip Optimization: $SKIP_OPTIMIZATION"
    print_info "  GPU Only: $GPU_ONLY"
    print_info "  Quick Mode: $QUICK_MODE"
    echo ""

    # System checks
    check_python_version
    check_gpu_availability
    check_disk_space
    check_memory

    # Environment setup
    setup_virtual_environment
    install_pytorch
    install_dependencies
    verify_installation

    # Training preparation
    prepare_training_environment
    create_config_file

    # Run training
    START_TIME=$(date +%s)
    run_training
    END_TIME=$(date +%s)

    # Calculate training time
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))

    print_success "Total training time: ${HOURS}h ${MINUTES}m ${SECONDS}s"

    # Show results and run demo
    show_results_summary
    run_demo

    # Final message
    echo ""
    print_success "🎉 Task A Complete System Training Finished Successfully!"
    print_info "Check the $OUTPUT_DIR directory for all generated files."
    print_info "Use demo_predictions.py to test your trained models."

    exit 0
}

# Run main function with all arguments
main "$@"
