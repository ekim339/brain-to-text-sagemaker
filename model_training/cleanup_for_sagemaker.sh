#!/bin/bash
# Cleanup script to remove unwanted files before SageMaker packaging
# Run this if you ever need to manually clean up

echo "Cleaning up model_training directory for SageMaker..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"

# Remove checkpoint directories
if [ -d ".ipynb_checkpoints" ]; then
    echo "Removing .ipynb_checkpoints/"
    rm -rf .ipynb_checkpoints/
fi

# Remove Python cache
if [ -d "__pycache__" ]; then
    echo "Removing __pycache__/"
    rm -rf __pycache__/
fi

# Remove .pyc files
if ls *.pyc 1> /dev/null 2>&1; then
    echo "Removing *.pyc files"
    rm -f *.pyc
fi

# Remove .DS_Store
if [ -f ".DS_Store" ]; then
    echo "Removing .DS_Store"
    rm -f .DS_Store
fi

echo "✓ Cleanup complete!"
echo ""
echo "Note: With .sagemakerignore in place, you shouldn't need to run this manually."
echo "SageMaker will automatically exclude these files."

