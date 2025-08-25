# Federated Learning Platform - Setup Guide

This repository contains a comprehensive federated learning platform with TensorFlow Federated (TFF), differential privacy, and secure aggregation examples.

## 🚀 Quick Setup (Recommended)

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Git

### Option 1: Automated Setup (Easiest)

```bash
# Clone the repository
git clone <your-repo-url>
cd federated-learning-platform

# Run the automated setup script
./setup_environment.sh

# Or use the Python version
python setup_environment.py
```

### Option 2: Manual Setup with Conda

```bash
# Create environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate tff

# Verify the installation
python -c "import tensorflow as tf, tensorflow_federated as tff, matplotlib.pyplot as plt; print('✅ Setup successful!')"
```

### Option 3: Using pip only

```bash
# Create a new Python 3.9 environment (using conda, venv, or pyenv)
conda create -n tff python=3.9 -y
conda activate tff

# Install packages
pip install -r requirements-clean.txt
```

## 🧪 Testing the Setup

After installation, test that everything works:

```bash
conda activate tff
python -c "
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp
import matplotlib.pyplot as plt
import numpy as np

print('✅ TensorFlow:', tf.__version__)
print('✅ TensorFlow Federated:', tff.__version__)  
print('✅ TensorFlow Privacy:', tfp.__version__)
print('✅ NumPy:', np.__version__)
print('✅ All core components working!')
"
```

Expected output:
```
✅ TensorFlow: 2.8.0
✅ TensorFlow Federated: 0.31.0
✅ TensorFlow Privacy: 0.8.1
✅ NumPy: 1.26.4
✅ All core components working!
```

## 📓 Running the Notebooks

```bash
conda activate tff
jupyter notebook
```

Then navigate to the notebooks in:
- `federated-learning-playground/` - Main FL experiments
- `differential-privacy-playground/` - DP concepts and examples

## 🔧 Package Versions

This environment uses these key package versions:
- **TensorFlow**: 2.8.0
- **TensorFlow Federated**: 0.31.0
- **TensorFlow Privacy**: 0.8.1
- **NumPy**: 1.26.4
- **Python**: 3.9

These versions are tested to work together without conflicts.

## 🐛 Troubleshooting

### Common Issues

1. **Conda not found**
   ```bash
   # Install Miniconda first
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Import errors**
   ```bash
   # Ensure you're in the right environment
   conda activate tff
   
   # Verify environment
   conda list | grep tensorflow
   ```

3. **Matplotlib issues**
   ```bash
   # The environment includes compatible versions
   # If issues persist, try:
   pip install --upgrade matplotlib
   ```

### Platform-Specific Notes

- **macOS**: The setup works on both Intel and Apple Silicon Macs
- **Linux**: Tested on Ubuntu 20.04+
- **Windows**: Use conda or WSL2 for best compatibility

## 📁 Repository Structure

```
federated-learning-platform/
├── environment.yml              # Conda environment file
├── requirements-clean.txt       # Pip requirements  
├── setup_environment.sh         # Bash setup script
├── setup_environment.py         # Python setup script
├── SETUP.md                    # This file
├── federated-learning-playground/
│   ├── 1_setup.ipynb
│   ├── 2_fedavg_scratch.ipynb
│   ├── 7_experiment_cifar10_1k.ipynb
│   └── ...
└── differential-privacy-playground/
    └── basic-concepts/
        ├── laplace_mechanism.py
        └── sensitivity.py
```

## 🎯 Next Steps

1. **Start with the basics**: Run `1_setup.ipynb` 
2. **Try FedAvg**: Run `2_fedavg_scratch.ipynb`
3. **Scale up**: Run `7_experiment_cifar10_1k.ipynb`
4. **Add privacy**: Explore the differential privacy examples

Happy federated learning! 🤖🚀
