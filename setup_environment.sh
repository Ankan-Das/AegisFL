#!/bin/bash

# Federated Learning Platform - Environment Setup Script
# This script sets up the complete environment for the federated learning platform

set -e  # Exit on any error

echo "🚀 Setting up Federated Learning Platform Environment..."
echo "=============================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "tff"; then
    echo "🗑️  Removing existing 'tff' environment..."
    conda env remove -n tff -y
fi

# Create environment from yml file
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "🎯 To activate the environment, run:"
echo "   conda activate tff"
echo ""
echo "🧪 To test the setup, run:"
echo "   conda activate tff"
echo "   python -c \"import tensorflow as tf, tensorflow_federated as tff, matplotlib.pyplot as plt; print('✅ All imports successful!')\""
echo ""
echo "📓 To run the notebooks:"
echo "   conda activate tff"
echo "   jupyter notebook"
echo ""
echo "Happy federated learning! 🤖"
