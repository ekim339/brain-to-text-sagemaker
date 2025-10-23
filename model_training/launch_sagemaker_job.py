"""
Script to launch a SageMaker training job from your local machine or notebook
"""
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import time


def launch_training_job(
    job_name=None,
    s3_bucket='your-bucket-name',
    s3_data_prefix='data/hdf5_data_diphone_encoded',
    instance_type='ml.g4dn.xlarge',  # GPU instance
    instance_count=1,
    config_file='rnn_args.yaml',
    num_training_batches=None,
    batch_size=None,
    role=None,
):
    """
    Launch a SageMaker training job
    
    Args:
        job_name: Name for the training job (auto-generated if None)
        s3_bucket: S3 bucket containing your data
        s3_data_prefix: Prefix path to your HDF5 data folder
        instance_type: SageMaker instance type (GPU recommended)
            - ml.g4dn.xlarge: 1 GPU, 16GB GPU memory, $0.736/hr
            - ml.g4dn.2xlarge: 1 GPU, 32GB GPU memory, $0.94/hr  
            - ml.g5.xlarge: 1 GPU, 24GB GPU memory, $1.006/hr (newer generation)
            - ml.p3.2xlarge: 1 GPU (V100), 16GB GPU memory, $3.825/hr
        instance_count: Number of instances (1 for single-GPU training)
        config_file: Config file to use (rnn_args.yaml or rnn_args_diphone.yaml)
        num_training_batches: Override training batches (optional)
        batch_size: Override batch size (optional)
        role: IAM role for SageMaker (auto-detected if None)
    """
    
    # Get SageMaker session
    sess = sagemaker.Session()
    
    # Get execution role (either passed or auto-detected)
    if role is None:
        try:
            role = get_execution_role()
        except:
            role = sagemaker.get_execution_role()
    
    print(f"Using role: {role}")
    
    # Generate job name if not provided
    if job_name is None:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        job_name = f"brain-to-text-{timestamp}"
    
    print(f"Launching job: {job_name}")
    print(f"Instance: {instance_type}")
    print(f"S3 Data: s3://{s3_bucket}/{s3_data_prefix}")
    
    # Build hyperparameters
    hyperparameters = {
        's3-bucket': s3_bucket,
        's3-data-prefix': s3_data_prefix,
        'config-file': config_file,
        'gpu-number': '0',
    }
    
    # Add optional overrides
    if num_training_batches is not None:
        hyperparameters['num-training-batches'] = num_training_batches
    if batch_size is not None:
        hyperparameters['batch-size'] = batch_size
    
    # Define the PyTorch estimator
    estimator = PyTorch(
        entry_point='train_model_sagemaker.py',
        source_dir='.',  # Current directory containing all training code
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version='2.1.0',  # PyTorch version
        py_version='py310',
        hyperparameters=hyperparameters,
        output_path=f's3://{s3_bucket}/sagemaker-outputs/{job_name}',
        base_job_name='brain-to-text',
        max_run=7*24*60*60,  # Max 7 days
        keep_alive_period_in_seconds=0,  # No warm pools
        environment={
            'TORCH_COMPILE_DEBUG': '0',
        },
        # Dependencies
        dependencies=[
            'rnn_trainer.py',
            'rnn_trainer_s3.py', 
            'rnn_model.py',
            'dataset.py',
            'dataset_s3.py',
            'data_augmentations.py',
            config_file,
        ]
    )
    
    print("\nStarting training job...")
    print("="*80)
    
    # Start the training job (no input data needed - we stream from S3)
    estimator.fit(wait=False)
    
    print("="*80)
    print(f"Training job launched: {job_name}")
    print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home?region={sess.boto_region_name}#/jobs/{job_name}")
    print(f"\nTo monitor logs from command line:")
    print(f"  aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix {job_name}")
    print(f"\nOutput will be saved to:")
    print(f"  s3://{s3_bucket}/sagemaker-outputs/{job_name}")
    
    return estimator


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch SageMaker training job')
    parser.add_argument('--job-name', type=str, default=None,
                        help='Training job name (auto-generated if not provided)')
    parser.add_argument('--s3-bucket', type=str, required=True,
                        help='S3 bucket containing your data')
    parser.add_argument('--s3-data-prefix', type=str, 
                        default='data/hdf5_data_diphone_encoded',
                        help='S3 prefix for data folder')
    parser.add_argument('--instance-type', type=str, 
                        default='ml.g4dn.xlarge',
                        help='SageMaker instance type')
    parser.add_argument('--config-file', type=str, 
                        default='rnn_args.yaml',
                        help='Config file to use')
    parser.add_argument('--num-training-batches', type=int, default=None,
                        help='Override number of training batches')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    
    args = parser.parse_args()
    
    estimator = launch_training_job(
        job_name=args.job_name,
        s3_bucket=args.s3_bucket,
        s3_data_prefix=args.s3_data_prefix,
        instance_type=args.instance_type,
        config_file=args.config_file,
        num_training_batches=args.num_training_batches,
        batch_size=args.batch_size,
    )

