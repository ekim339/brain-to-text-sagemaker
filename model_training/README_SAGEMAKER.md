# Training on AWS SageMaker with S3 Data Streaming

This guide explains how to train the brain-to-text model on AWS SageMaker while streaming data directly from S3, avoiding local storage limitations.

## Overview

The SageMaker setup includes:
- **`dataset_s3.py`**: Modified dataset that reads HDF5 files directly from S3 using `s3fs`
- **`rnn_trainer_s3.py`**: Extended trainer that uses S3-backed datasets
- **`train_model_sagemaker.py`**: Entry point for SageMaker training jobs
- **`launch_sagemaker_job.py`**: Helper script to launch jobs from your local machine

## Prerequisites

### 1. Data on S3

Upload your diphone-encoded data to S3:

```bash
aws s3 sync ../data/hdf5_data_diphone_encoded s3://your-bucket/data/hdf5_data_diphone_encoded
```

Expected S3 structure:
```
s3://your-bucket/data/hdf5_data_diphone_encoded/
├── t15.2023.08.11/
│   ├── data_train.hdf5
│   └── data_val.hdf5
├── t15.2023.08.13/
│   ├── data_train.hdf5
│   └── data_val.hdf5
└── ... (more session folders)
```

### 2. AWS Credentials

Ensure your AWS credentials are configured:

```bash
aws configure
# Or set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### 3. IAM Role

Create a SageMaker execution role with:
- `AmazonSageMakerFullAccess`
- S3 read access to your data bucket
- S3 write access to your output bucket

## Launching a Training Job

### Option 1: From Your Local Machine (Recommended)

```bash
cd model_training

# Launch with default settings
python launch_sagemaker_job.py \
  --s3-bucket your-bucket-name \
  --s3-data-prefix data/hdf5_data_diphone_encoded \
  --config-file rnn_args.yaml

# Or with custom settings
python launch_sagemaker_job.py \
  --s3-bucket your-bucket-name \
  --s3-data-prefix data/hdf5_data_diphone_encoded \
  --config-file rnn_args.yaml \
  --instance-type ml.g4dn.2xlarge \
  --num-training-batches 150000 \
  --batch-size 48 \
  --job-name my-diphone-training
```

### Option 2: From SageMaker Notebook

```python
from launch_sagemaker_job import launch_training_job

estimator = launch_training_job(
    s3_bucket='your-bucket-name',
    s3_data_prefix='data/hdf5_data_diphone_encoded',
    instance_type='ml.g4dn.xlarge',
    config_file='rnn_args.yaml',
    num_training_batches=150000,
    batch_size=48
)

# Wait for completion
estimator.fit(wait=True)
```

### Option 3: Manual SageMaker Training Job

```bash
# Package your code
tar -czf sourcedir.tar.gz *.py *.yaml

# Upload to S3
aws s3 cp sourcedir.tar.gz s3://your-bucket/code/

# Create training job via AWS CLI or Console
```

## Instance Types

Choose based on your budget and training speed needs:

| Instance Type | GPU | GPU Memory | CPU | RAM | Cost/hr | Use Case |
|---------------|-----|------------|-----|-----|---------|----------|
| `ml.g4dn.xlarge` | 1x T4 | 16GB | 4 | 16GB | $0.736 | Development/small models |
| `ml.g4dn.2xlarge` | 1x T4 | 16GB | 8 | 32GB | $0.94 | Standard training |
| `ml.g5.xlarge` | 1x A10G | 24GB | 4 | 16GB | $1.006 | Better performance |
| `ml.g5.2xlarge` | 1x A10G | 24GB | 8 | 32GB | $1.212 | Recommended |
| `ml.p3.2xlarge` | 1x V100 | 16GB | 8 | 61GB | $3.825 | Maximum performance |

For diphone model (1681 classes), recommend:
- **Development**: `ml.g4dn.xlarge` with `batch_size=32`
- **Production**: `ml.g5.2xlarge` with `batch_size=48`

## Configuration

### Create a Diphone Config for SageMaker

```yaml
# rnn_args_diphone_sagemaker.yaml
model:
  n_units: 1024
  n_classes: 1681  # Diphones
  rnn_dropout: 0.5
  # ... other model params

dataset:
  batch_size: 48  # Adjust based on GPU memory
  n_classes: 1681
  dataset_dir: /will/be/overridden  # Not used with S3
  # ... other dataset params

# SageMaker will override these:
output_dir: /opt/ml/output
checkpoint_dir: /opt/ml/model/checkpoint

num_training_batches: 150000
```

## Monitoring Training

### View Logs in Real-time

```bash
# Get job name from launch output, then:
aws logs tail /aws/sagemaker/TrainingJobs \
  --follow \
  --log-stream-name-prefix brain-to-text-2025-01-15-10-30-00
```

### Check Training Progress

```python
import boto3
client = boto3.client('sagemaker')

response = client.describe_training_job(
    TrainingJobName='brain-to-text-2025-01-15-10-30-00'
)
print(response['TrainingJobStatus'])
print(response['SecondaryStatus'])
```

### Download Results

```bash
# Model checkpoint
aws s3 sync s3://your-bucket/sagemaker-outputs/your-job-name/output/model.tar.gz ./

# Extract
tar -xzf model.tar.gz
```

## Performance Tips

### 1. Optimize Batch Size

Start with these values and adjust based on GPU memory:

```python
# For ml.g4dn.xlarge (16GB GPU)
batch_size = 32

# For ml.g5.2xlarge (24GB GPU) 
batch_size = 48

# For ml.p3.2xlarge (16GB GPU, but faster)
batch_size = 40
```

### 2. Use DataLoader Workers

The S3 dataset supports multiple workers:

```yaml
dataset:
  num_dataloader_workers: 4  # Parallel S3 reads
```

However, if you experience slowdowns, try reducing to `2` or `0`.

### 3. Enable Compilation

The trainer uses `torch.compile` by default for ~20% speedup.

### 4. Mixed Precision Training

Enable AMP for 2-3x speedup:

```yaml
use_amp: true  # Already default
```

## Cost Estimation

Example training run:
- Instance: `ml.g5.2xlarge` ($1.212/hr)
- Training batches: 150,000
- Time per batch: ~0.6 seconds
- Total time: 150,000 × 0.6 = 90,000 seconds ≈ 25 hours
- **Total cost**: 25 × $1.212 = **~$30**

Validation adds ~10% overhead, so budget **~$35 total**.

## Troubleshooting

### "No space left on device"

✅ **Solution**: You're already using S3 streaming, so this shouldn't happen. If it does, check:
- Logs are not filling `/opt/ml/output`
- Checkpoints are not too large

### "Access Denied" S3 errors

Check your SageMaker execution role has:
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:ListBucket"],
  "Resource": [
    "arn:aws:s3:::your-bucket/*",
    "arn:aws:s3:::your-bucket"
  ]
}
```

### Slow training

1. Check `num_dataloader_workers` (try 2-4)
2. Verify S3 bucket is in same region as SageMaker
3. Use faster instance type
4. Reduce batch size if CPU is bottleneck

### Out of Memory (OOM)

Reduce `batch_size`:
```bash
python launch_sagemaker_job.py \
  --s3-bucket your-bucket \
  --batch-size 32  # Down from 48
```

Or reduce model size:
```yaml
model:
  n_units: 768  # Down from 1024
```

## Example: Complete Training Run

```bash
# 1. Upload config (if modified)
aws s3 cp rnn_args.yaml s3://your-bucket/configs/

# 2. Launch training
python launch_sagemaker_job.py \
  --s3-bucket your-bucket \
  --s3-data-prefix data/hdf5_data_diphone_encoded \
  --config-file rnn_args.yaml \
  --instance-type ml.g5.2xlarge \
  --job-name diphone-baseline-v1

# 3. Monitor (in another terminal)
aws logs tail /aws/sagemaker/TrainingJobs \
  --follow \
  --log-stream-name-prefix diphone-baseline-v1

# 4. After completion, download results
aws s3 sync s3://your-bucket/sagemaker-outputs/diphone-baseline-v1 ./results/

# 5. Extract checkpoint
cd results
tar -xzf output/model.tar.gz
# Checkpoint is in: checkpoint/best_checkpoint
```

## Differences from Local Training

| Aspect | Local | SageMaker S3 |
|--------|-------|--------------|
| Data loading | HDF5 from disk | HDF5 from S3 via s3fs |
| Dataset class | `BrainToTextDataset` | `BrainToTextDatasetS3` |
| Trainer | `BrainToTextDecoder_Trainer` | `BrainToTextDecoder_Trainer_S3` |
| Storage needed | ~100GB+ | ~10GB (logs/checkpoints only) |
| Speed | Fastest | ~10-20% slower (S3 latency) |
| Cost | Hardware cost | ~$30-50 per training run |

## Next Steps

1. Start with a short test run:
   ```bash
   --num-training-batches 1000  # Just to verify setup
   ```

2. Once confirmed working, launch full training:
   ```bash
   --num-training-batches 150000
   ```

3. Monitor validation metrics and adjust hyperparameters as needed

4. Download checkpoints and evaluate on test set

## Support

For issues specific to:
- **S3/AWS**: Check AWS documentation or CloudWatch logs
- **Training logic**: Same as local training (see main README)
- **Performance**: Adjust batch size, workers, or instance type

