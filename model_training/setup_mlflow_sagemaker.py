#!/usr/bin/env python3
"""
MLflow setup script for SageMaker notebook.
Run this to start MLflow tracking server.
"""

import os
import subprocess
import time
import sys

def check_mlflow_installed():
    """Check if MLflow is installed."""
    try:
        import mlflow
        print("✅ MLflow is installed")
        return True
    except ImportError:
        print("❌ MLflow is not installed")
        return False

def install_mlflow():
    """Install MLflow."""
    print("Installing MLflow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow", "-q"])
    print("✅ MLflow installed successfully")

def is_mlflow_running():
    """Check if MLflow server is already running."""
    result = subprocess.run(['pgrep', '-f', 'mlflow server'], capture_output=True)
    return result.returncode == 0

def start_mlflow_server():
    """Start MLflow tracking server."""
    # Create MLflow directory
    mlflow_dir = os.path.expanduser('~/mlruns')
    os.makedirs(mlflow_dir, exist_ok=True)
    
    if is_mlflow_running():
        print("✅ MLflow server is already running")
        return
    
    print("Starting MLflow server...")
    
    # Start server in background
    log_file = os.path.expanduser('~/mlflow.log')
    db_path = os.path.expanduser('~/mlflow.db')
    
    with open(log_file, 'w') as f:
        subprocess.Popen([
            'mlflow', 'server',
            '--backend-store-uri', f'sqlite:///{db_path}',
            '--default-artifact-root', mlflow_dir,
            '--host', '0.0.0.0',
            '--port', '5000'
        ], stdout=f, stderr=f)
    
    # Wait for server to start
    time.sleep(3)
    
    # Check if it's running
    if is_mlflow_running():
        print("✅ MLflow server started successfully")
        print(f"   Log file: {log_file}")
    else:
        print("❌ Failed to start MLflow server")
        print(f"   Check log file: {log_file}")
        return False
    
    return True

def test_mlflow_connection():
    """Test connection to MLflow server."""
    import requests
    
    try:
        response = requests.get('http://localhost:5000/health', timeout=2)
        if response.status_code == 200:
            print("✅ MLflow server is accessible")
            return True
        else:
            print(f"⚠️ MLflow server responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to MLflow server: {e}")
        return False

def print_usage():
    """Print usage instructions."""
    print("\n" + "="*80)
    print("MLflow Setup Complete!")
    print("="*80)
    print("\n📊 Access MLflow UI:")
    print("   http://localhost:5000")
    print("\n🚀 Launch training with MLflow:")
    print("   python train_model_sagemaker_mlflow.py \\")
    print("       --s3-bucket 4k-eugene-btt \\")
    print("       --s3-data-prefix hdf5_data_diphone_encoded \\")
    print("       --config-file rnn_args_diphone_sagemaker.yaml \\")
    print("       --mlflow-tracking-uri http://localhost:5000 \\")
    print("       --mlflow-experiment-name my-experiment")
    print("\n🛑 Stop MLflow server:")
    print("   pkill -f 'mlflow server'")
    print("\n" + "="*80)

def main():
    """Main setup function."""
    print("="*80)
    print("MLflow Setup for SageMaker")
    print("="*80)
    
    # Check/install MLflow
    if not check_mlflow_installed():
        install_mlflow()
    
    # Start server
    if start_mlflow_server():
        # Test connection
        time.sleep(2)
        if test_mlflow_connection():
            print_usage()
        else:
            print("\n⚠️ Server started but not accessible")
            print("   It may take a few more seconds to initialize")
    else:
        print("\n❌ Setup failed")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

