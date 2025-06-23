#!/bin/bash

# Face Recognition System - Environment Activation Script
# This script activates the virtual environment and provides helpful information

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/face_recognition_env"

echo -e "${CYAN}🎭 Face Recognition System - Environment Activation${NC}"
echo -e "${CYAN}=====================================================${NC}"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}❌ Virtual environment not found at: $VENV_DIR${NC}"
    echo -e "${YELLOW}Creating virtual environment...${NC}"

    # Create virtual environment
    python3 -m venv "$VENV_DIR"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Virtual environment created successfully${NC}"
    else
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${BLUE}🔄 Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Virtual environment activated successfully${NC}"
else
    echo -e "${RED}❌ Failed to activate virtual environment${NC}"
    exit 1
fi

# Check if requirements are installed
echo -e "${BLUE}🔍 Checking installed packages...${NC}"
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Required packages not found. Installing requirements...${NC}"
    pip install -r requirements.txt

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Requirements installed successfully${NC}"
    else
        echo -e "${RED}❌ Failed to install requirements${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Required packages are already installed${NC}"
fi

# Display system information
echo -e "\n${PURPLE}📊 System Information:${NC}"
echo -e "${BLUE}Python version:${NC} $(python --version)"
echo -e "${BLUE}PyTorch version:${NC} $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")"
echo -e "${BLUE}CUDA available:${NC} $(python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null || echo "Unknown")"
echo -e "${BLUE}Virtual env path:${NC} $VIRTUAL_ENV"

# Display available commands
echo -e "\n${PURPLE}🚀 Available Commands:${NC}"
echo -e "${GREEN}Training:${NC}"
echo -e "  python train_face_recognition.py                # Full training"
echo -e "  python train_face_recognition.py --epochs 5     # Quick training"
echo -e "  python train_face_recognition.py --help         # See all options"

echo -e "\n${GREEN}Demo & Testing:${NC}"
echo -e "  python demo.py --quick_train                     # Interactive demo with training"
echo -e "  python demo.py --mode verify                     # Verification demo"
echo -e "  python demo.py --mode identify                   # Identification demo"
echo -e "  python test_installation.py                     # Test installation"

echo -e "\n${GREEN}Inference:${NC}"
echo -e "  python inference.py --mode verify --image1 img1.jpg --image2 img2.jpg"
echo -e "  python inference.py --mode identify --query_image query.jpg --gallery_dir train"
echo -e "  python inference.py --mode evaluate --data_dir train"

echo -e "\n${GREEN}Batch Processing:${NC}"
echo -e "  python batch_processor.py --mode verification --input_file pairs.csv"
echo -e "  python batch_processor.py --mode identification --query_dir test --gallery_dir train"

echo -e "\n${GREEN}Pipeline:${NC}"
echo -e "  python run_pipeline.py --config config.json     # Full pipeline"

echo -e "\n${GREEN}Setup & Utilities:${NC}"
echo -e "  python setup.py                                 # Setup system"
echo -e "  python evaluation_utils.py --help               # Evaluation tools"

# Display data information if train directory exists
if [ -d "$SCRIPT_DIR/train" ]; then
    PERSON_COUNT=$(find "$SCRIPT_DIR/train" -maxdepth 1 -type d | wc -l)
    PERSON_COUNT=$((PERSON_COUNT - 1))  # Subtract 1 for the train directory itself
    echo -e "\n${PURPLE}📁 Dataset Information:${NC}"
    echo -e "${BLUE}Training data:${NC} Found $PERSON_COUNT person directories"
    echo -e "${BLUE}Data path:${NC} $SCRIPT_DIR/train"
else
    echo -e "\n${YELLOW}⚠️  Training data not found. Please add your data to the 'train' directory.${NC}"
    echo -e "${BLUE}Expected structure:${NC}"
    echo -e "  train/"
    echo -e "  ├── person1_name/"
    echo -e "  │   ├── person1_image.jpg"
    echo -e "  │   └── distortion/"
    echo -e "  │       ├── person1_image_blurred.jpg"
    echo -e "  │       └── ..."
    echo -e "  └── person2_name/"
    echo -e "      └── ..."
fi

# Check for existing models
if [ -f "$SCRIPT_DIR/best_face_model.pth" ]; then
    echo -e "\n${GREEN}✅ Trained model found: best_face_model.pth${NC}"
else
    echo -e "\n${YELLOW}⚠️  No trained model found. Run training first.${NC}"
fi

echo -e "\n${PURPLE}💡 Quick Start Tips:${NC}"
echo -e "${BLUE}•${NC} For first time: ${GREEN}python demo.py --quick_train${NC}"
echo -e "${BLUE}•${NC} For full training: ${GREEN}python train_face_recognition.py${NC}"
echo -e "${BLUE}•${NC} To test installation: ${GREEN}python test_installation.py${NC}"
echo -e "${BLUE}•${NC} For help: ${GREEN}python <script>.py --help${NC}"

echo -e "\n${GREEN}✨ Environment ready! You can now run face recognition commands.${NC}"
echo -e "${CYAN}=====================================================${NC}"

# Keep the environment activated by starting a new shell
echo -e "${BLUE}Starting new shell with activated environment...${NC}"
exec "$SHELL"
