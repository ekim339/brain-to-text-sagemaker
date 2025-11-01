"""
SageMaker training script with MLflow logging integration.
Streams data directly from S3 and logs all metrics to MLflow.
"""
import subprocess, sys

def pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

pip_install(["omegaconf", "h5py", "tqdm", "boto3", "mlflow"])

import os
import sys
import argparse
from omegaconf import OmegaConf
from rnn_trainer_s3_mlflow import BrainToTextDecoder_Trainer_S3_MLflow


def main():
    parser = argparse.ArgumentParser(description='Train Brain-to-Text model on SageMaker with MLflow logging')
    
    # S3 configuration
    parser.add_argument('--s3-bucket', type=str, required=True,
                        help='S3 bucket name (e.g., my-brain-data-bucket)')
    parser.add_argument('--s3-data-prefix', type=str, required=True,
                        help='S3 prefix for data folder (e.g., data/hdf5_data_diphone_encoded)')
    
    # MLflow configuration
    parser.add_argument('--mlflow-tracking-uri', type=str, default='http://localhost:5000',
                        help='MLflow tracking server URI')
    parser.add_argument('--mlflow-experiment-name', type=str, default='brain-to-text-diphone',
                        help='MLflow experiment name')
    
    # Configuration
    parser.add_argument('--config-file', type=str, default='rnn_args_diphone_sagemaker.yaml',
                        help='Config file name (must be in the code directory)')
    
    # SageMaker environment variables (automatically set by SageMaker)
    parser.add_argument('--model-dir', type=str, 
                        default=os.environ.get('SM_MODEL_DIR', './trained_models'))
    parser.add_argument('--output-data-dir', type=str, 
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    
    # Training overrides (optional)
    parser.add_argument('--num-training-batches', type=int, default=None,
                        help='Override number of training batches')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--gpu-number', type=str, default='0',
                        help='GPU number to use')
    
    args = parser.parse_args()
    
    # Set MLflow environment variables
    os.environ['MLFLOW_TRACKING_URI'] = args.mlflow_tracking_uri
    os.environ['MLFLOW_EXPERIMENT_NAME'] = args.mlflow_experiment_name
    
    print("="*80)
    print("NEJM Brain-to-Text Training on AWS SageMaker with MLflow")
    print("="*80)
    print(f"S3 Bucket: {args.s3_bucket}")
    print(f"S3 Data Prefix: {args.s3_data_prefix}")
    print(f"Full S3 Path: s3://{args.s3_bucket}/{args.s3_data_prefix}")
    print(f"MLflow Tracking URI: {args.mlflow_tracking_uri}")
    print(f"MLflow Experiment: {args.mlflow_experiment_name}")
    print("="*80)
    
    # Load config file
    config_path = args.config_file
    if not os.path.exists(config_path):
        # Try to find it in the code directory
        code_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(code_dir, args.config_file)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_file}")
    
    print(f"\nLoading config from: {config_path}")
    training_args = OmegaConf.load(config_path)
    
    # Add S3 configuration to args
    training_args['s3_bucket'] = args.s3_bucket
    training_args['s3_data_prefix'] = args.s3_data_prefix
    
    # Override paths for SageMaker
    training_args['output_dir'] = args.output_data_dir
    training_args['checkpoint_dir'] = os.path.join(args.model_dir, 'checkpoint')
    training_args['gpu_number'] = args.gpu_number
    
    # Apply any command-line overrides
    if args.num_training_batches is not None:
        training_args['num_training_batches'] = args.num_training_batches
        print(f"Override: num_training_batches = {args.num_training_batches}")
    
    if args.batch_size is not None:
        training_args['dataset']['batch_size'] = args.batch_size
        print(f"Override: batch_size = {args.batch_size}")
    
    print(f"\nOutput directory: {training_args['output_dir']}")
    print(f"Checkpoint directory: {training_args['checkpoint_dir']}")
    print(f"Number of classes: {training_args['dataset']['n_classes']}")
    print(f"Batch size: {training_args['dataset']['batch_size']}")
    print(f"Training batches: {training_args['num_training_batches']}")
    
    # Create output directories
    os.makedirs(training_args['output_dir'], exist_ok=True)
    os.makedirs(training_args['checkpoint_dir'], exist_ok=True)
    
    # Initialize trainer with S3 and MLflow support
    print("\n" + "="*80)
    print("Initializing S3-Enabled Trainer with MLflow Logging")
    print("="*80)
    
    trainer = BrainToTextDecoder_Trainer_S3_MLflow(training_args)
    
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    # Train the model (MLflow logging happens automatically)
    metrics = trainer.train()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best validation PER: {trainer.best_val_PER:.4f}")
    
    # Save final metrics
    import json
    metrics_file = os.path.join(training_args['output_dir'], 'final_metrics.json')
    with open(metrics_file, 'w') as f:
        serializable_metrics = {
            'best_val_DER': float(trainer.best_val_PER),  # Note: this is actually DER in diphone mode
            'best_val_loss': float(trainer.best_val_loss),
            'train_losses': [float(x) for x in metrics['train_losses'][-100:]],  # Last 100
            'val_losses': [float(x) for x in metrics['val_losses']],
            'val_DERs': [float(x) for x in metrics.get('val_DERs', [])],
            'val_PERs': [float(x) for x in metrics.get('val_PERs', [])],
        }
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Saved metrics to: {metrics_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()

