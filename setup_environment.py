#!/usr/bin/env python3
"""
Federated Learning Platform - Environment Setup Script
This script sets up the complete environment for the federated learning platform
"""

import subprocess
import sys
import os

def run_command(cmd, check=True):
    """Run a shell command and handle errors"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Error running command: {cmd}")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    return result

def check_conda():
    """Check if conda is available"""
    result = run_command("conda --version", check=False)
    if result.returncode != 0:
        print("❌ Error: Conda is not installed or not in PATH")
        print("Please install Miniconda or Anaconda first:")
        print("https://docs.conda.io/en/latest/miniconda.html")
        sys.exit(1)
    print(f"✅ Found conda: {result.stdout.strip()}")

def setup_environment():
    """Set up the conda environment"""
    print("🚀 Setting up Federated Learning Platform Environment...")
    print("=" * 50)
    
    # Check conda
    check_conda()
    
    # Check if environment.yml exists
    if not os.path.exists("environment.yml"):
        print("❌ Error: environment.yml not found")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Remove existing environment if it exists
    print("🗑️  Removing existing 'tff' environment if it exists...")
    run_command("conda env remove -n tff -y", check=False)
    
    # Create environment
    print("📦 Creating conda environment from environment.yml...")
    run_command("conda env create -f environment.yml")
    
    # Test the environment
    print("🧪 Testing the environment...")
    test_cmd = """conda run -n tff python -c "
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp
import matplotlib.pyplot as plt
import numpy as np
print('✅ TensorFlow:', tf.__version__)
print('✅ TensorFlow Federated:', tff.__version__)
print('✅ TensorFlow Privacy:', tfp.__version__)
print('✅ NumPy:', np.__version__)
print('✅ All imports successful!')
" """
    run_command(test_cmd)
    
    print("")
    print("✅ Environment setup complete!")
    print("")
    print("🎯 To activate the environment, run:")
    print("   conda activate tff")
    print("")
    print("📓 To run the notebooks:")
    print("   conda activate tff")
    print("   jupyter notebook")
    print("")
    print("Happy federated learning! 🤖")

if __name__ == "__main__":
    setup_environment()
