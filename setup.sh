#!/bin/bash
# Quick start script for Emergency Path Finder

echo "╔════════════════════════════════════════════════════════╗"
echo "║   Emergency Path Finder - Quick Start Setup            ║"
echo "╚════════════════════════════════════════════════════════╝"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python
echo -e "\n${BLUE}[1/5] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check Flutter
echo -e "\n${BLUE}[2/5] Checking Flutter...${NC}"
if command -v flutter &> /dev/null; then
    FLUTTER_VERSION=$(flutter --version | head -1)
    echo -e "${GREEN}✓ $FLUTTER_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ Flutter not found. You can install it from https://flutter.dev${NC}"
fi

# Setup Python environment
echo -e "\n${BLUE}[3/5] Installing Python dependencies...${NC}"
cd training
if python3 -m pip install -r requirements.txt --quiet; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi
cd ..

# Create necessary directories
echo -e "\n${BLUE}[4/5] Creating directories...${NC}"
mkdir -p datasets ml_models flutter_app/assets/models
echo -e "${GREEN}✓ Directories created${NC}"

# Summary
echo -e "\n${BLUE}[5/5] Setup complete!${NC}"

echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}NEXT STEPS:${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"

echo -e "\n1. ${YELLOW}Download Datasets${NC}"
echo "   cd training"
echo "   python3 download_datasets.py"
echo "   (Follow instructions to download from Roboflow)"

echo -e "\n2. ${YELLOW}Train ML Model (Optional)${NC}"
echo "   python3 train_exit_detector.py"
echo "   (Takes 2-4 hours on CPU, 20 mins on GPU)"

echo -e "\n3. ${YELLOW}Test Detection${NC}"
echo "   python3 test_detection.py --camera"
echo "   (Test on webcam)"

echo -e "\n4. ${YELLOW}Build Mobile App${NC}"
echo "   cd ../flutter_app"
echo "   flutter pub get"
echo "   flutter run"

echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "\nDocumentation:"
echo "  - Setup Guide: docs/SETUP.md"
echo "  - Architecture: docs/ARCHITECTURE.md"
echo "  - README: README.md"

echo -e "\n${GREEN}Happy coding! 🚀${NC}\n"
