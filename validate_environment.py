#!/usr/bin/env python3
"""
Environment Validation Script
Tests all the key components of the federated learning platform
"""

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import tensorflow_federated as tff
        print(f"✅ TensorFlow Federated: {tff.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow Federated import failed: {e}")
        return False
    
    try:
        import tensorflow_privacy as tfp
        print(f"✅ TensorFlow Privacy: {tfp.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow Privacy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib: Available")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    return True

def test_tensorflow_privacy():
    """Test TensorFlow Privacy functionality"""
    print("\n🔒 Testing TensorFlow Privacy functionality...")
    
    try:
        from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
        print("✅ DP Optimizer: Available")
    except ImportError as e:
        print(f"❌ DP Optimizer import failed: {e}")
        return False
    
    try:
        from tensorflow_privacy.privacy.dp_query.gaussian_query import GaussianSumQuery
        print("✅ Gaussian Query: Available")
    except ImportError as e:
        print(f"❌ Gaussian Query import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic TensorFlow and TFF functionality"""
    print("\n⚙️  Testing basic functionality...")
    
    try:
        import tensorflow as tf
        import tensorflow_federated as tff
        
        # Test TensorFlow
        x = tf.constant([1, 2, 3])
        y = tf.constant([4, 5, 6])
        z = tf.add(x, y)
        print("✅ TensorFlow operations: Working")
        
        # Test TFF
        @tff.tf_computation
        def add_one(x):
            return x + 1
        
        result = add_one(5)
        print("✅ TensorFlow Federated computations: Working")
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False
    
    return True

def main():
    """Run all validation tests"""
    print("🚀 Federated Learning Platform - Environment Validation")
    print("=" * 55)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test TensorFlow Privacy
    if not test_tensorflow_privacy():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    print("\n" + "=" * 55)
    if all_passed:
        print("🎉 All tests passed! Environment is ready for federated learning.")
        print("\n📓 You can now run the notebooks:")
        print("   jupyter notebook")
    else:
        print("❌ Some tests failed. Please check the setup.")
        print("\n🔧 Try running the setup script again:")
        print("   ./setup_environment.sh")
    
    return all_passed

if __name__ == "__main__":
    main()
