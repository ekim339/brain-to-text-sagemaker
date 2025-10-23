# AWS SageMaker Setup Summary - Brain-to-Text Diphone Training

## What Was Created

I've created a complete SageMaker training pipeline that streams data directly from S3, avoiding local storage limitations. Here's what you now have:

### Core Files

1. **`model_training/dataset_s3.py`**
   - Modified dataset class that reads HDF5 files from S3 using `s3fs`
   - Works identically to the original `dataset.py` but with S3 support
   - Key feature: Streams data on-demand, no local storage needed

2. **`model_training/rnn_trainer_s3.py`**
   - Extended trainer that uses S3-backed datasets
   - Inherits all functionality from `rnn_trainer.py`
   - Only overrides dataset initialization to use S3

3. **`model_training/train_model_sagemaker.py`**
   - Entry point for SageMaker training jobs
   - Handles command-line arguments and environment setup
   - Automatically configures paths for SageMaker environment

4. **`model_training/launch_sagemaker_job.py`**
   - Helper script to launch training jobs from your local machine
   - Handles all SageMaker API calls and configuration
   - Can be run directly or imported as a module

### Configuration Files

5. **`model_training/rnn_args_diphone_sagemaker.yaml`**
   - Pre-configured for diphone training (1681 classes)
   - Optimized hyperparameters for SageMaker
   - Adjusted batch size, model capacity, and regularization

6. **`model_training/requirements_sagemaker.txt`**
   - All dependencies needed in SageMaker environment
   - Includes critical `s3fs` package for S3 streaming

### Documentation

7. **`model_training/README_SAGEMAKER.md`**
   - Comprehensive guide with all details
   - Troubleshooting tips, cost estimates, instance recommendations
   - Complete examples and monitoring instructions

8. **`model_training/SAGEMAKER_QUICK_START.md`**
   - Quick reference for launching jobs
   - Copy-paste commands to get started immediately

## Key Architecture Changes

### Data Flow Comparison

**Original (Local)**:
```
HDF5 Files (local disk) → dataset.py → DataLoader → Training
```

**New (SageMaker S3)**:
```
HDF5 Files (S3) → s3fs → dataset_s3.py → DataLoader → Training
                    ↑
            (streaming, no download)
```

### How S3 Streaming Works

1. **`s3fs`** creates a filesystem-like interface to S3
2. **`h5py`** can read directly from the s3fs file handles
3. Data is fetched **on-demand** as batches are requested
4. Only the current batch needs to be in memory

### Storage Requirements

- **Local training**: 100+ GB for all HDF5 files
- **SageMaker S3 streaming**: <10 GB (just logs and checkpoints)

## Quick Start (3 Steps)

### Step 1: Upload Data to S3

```bash
# From your local machine
aws s3 sync data/hdf5_data_diphone_encoded s3://YOUR-BUCKET/data/hdf5_data_diphone_encoded
```

### Step 2: Launch Training

```bash
cd model_training

python launch_sagemaker_job.py \
  --s3-bucket YOUR-BUCKET \
  --s3-data-prefix data/hdf5_data_diphone_encoded \
  --config-file rnn_args_diphone_sagemaker.yaml \
  --instance-type ml.g5.2xlarge
```

### Step 3: Monitor and Retrieve

```bash
# Monitor logs (replace JOB-NAME)
aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix JOB-NAME

# After completion, download results
aws s3 sync s3://YOUR-BUCKET/sagemaker-outputs/JOB-NAME ./results/
```

## Configuration for Diphone Training

The key changes from phoneme to diphone training:

| Parameter | Phoneme (41) | Diphone (1681) | Why Changed |
|-----------|--------------|----------------|-------------|
| `n_classes` | 41 | **1681** | Core change |
| `n_units` | 768 | **1024** | More capacity needed |
| `batch_size` | 64 | **48** | Larger output → more memory |
| `rnn_dropout` | 0.4 | **0.5** | More regularization |
| `num_training_batches` | 120K | **150K** | More training needed |
| `lr_max` | 0.005 | **0.004** | Conservative for stability |
| `weight_decay` | 0.001 | **0.002** | Prevent overfitting |

## Cost Estimation

**Recommended setup** (ml.g5.2xlarge):
- GPU: 1x NVIDIA A10G (24GB)
- Cost: $1.212/hour
- Training time: ~25 hours (150K batches)
- **Total: ~$35**

**Budget option** (ml.g4dn.xlarge):
- GPU: 1x NVIDIA T4 (16GB)  
- Cost: $0.736/hour
- Need to reduce batch_size to 32
- Training time: ~30 hours
- **Total: ~$25**

## AWS Prerequisites

### 1. IAM Role

Your SageMaker execution role needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR-BUCKET/*",
        "arn:aws:s3:::YOUR-BUCKET"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR-BUCKET/sagemaker-outputs/*"
      ]
    }
  ]
}
```

### 2. AWS Credentials

On your local machine:
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Set region (e.g., us-east-1)
```

### 3. Install SageMaker SDK

```bash
pip install sagemaker boto3 s3fs
```

## Expected Training Output

### During Training (CloudWatch Logs)

```
2025-01-15 10:30:00: Using device: cuda:0
2025-01-15 10:30:05: Initialized RNN decoding model
2025-01-15 10:30:05: Model has 12,345,678 parameters
2025-01-15 10:30:10: Loading data from: your-bucket/data/...
2025-01-15 10:30:15: Successfully initialized S3-backed datasets
2025-01-15 10:30:15: Train batches: 150000
2025-01-15 10:30:15: Val batches: 1234

Train batch 0: loss: 7.43 grad norm: 9.87 time: 0.542
Train batch 200: loss: 5.21 grad norm: 8.43 time: 0.538
...
Running test after training batch: 1500
Val batch 1500: DER (avg): 0.7654 CTC Loss (avg): 4.32 time: 12.345
New best test DER 0.9876 --> 0.7654
Checkpointing model
...
```

### After Training (S3 Output)

```
s3://YOUR-BUCKET/sagemaker-outputs/JOB-NAME/
├── output/
│   └── model.tar.gz          # Contains checkpoint/best_checkpoint
└── logs/
    └── training_log
```

Extract checkpoint:
```bash
tar -xzf model.tar.gz
# Files:
# - checkpoint/best_checkpoint
# - checkpoint/args.yaml
# - checkpoint/val_metrics.pkl
# - training_log
```

## Monitoring Performance

### Key Metrics to Watch

1. **Training Loss**: Should decrease steadily
   - Initial: ~7-8
   - After 1K batches: ~5-6
   - After 50K batches: ~2-3
   - Final: ~1-2

2. **Validation DER (Diphone Error Rate)**:
   - Initial: ~0.9-1.0 (90-100% error)
   - After 10K batches: ~0.6-0.7
   - After 50K batches: ~0.4-0.5
   - Final target: <0.35

3. **Training Speed**:
   - Target: 0.5-0.7 seconds per batch
   - If slower: Check num_dataloader_workers or S3 region

## Troubleshooting

### Issue: Out of Memory

**Solution 1**: Reduce batch size
```bash
python launch_sagemaker_job.py ... --batch-size 32
```

**Solution 2**: Reduce model size in config
```yaml
model:
  n_units: 768  # Down from 1024
```

### Issue: S3 Access Denied

**Check**: IAM role permissions (see above)
**Test**: 
```bash
aws s3 ls s3://YOUR-BUCKET/data/hdf5_data_diphone_encoded/
```

### Issue: Very Slow Training

**Causes**:
1. S3 bucket in different region than SageMaker
2. Too many dataloader workers
3. Network bottleneck

**Solutions**:
- Use S3 bucket in same region as training job
- Reduce `num_dataloader_workers` to 2
- Use faster instance type (g5 instead of g4)

### Issue: Job Fails Immediately

**Check logs**:
```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix JOB-NAME
```

**Common causes**:
- Wrong S3 paths
- Missing dependencies
- Config file errors

## Next Steps

1. **Test with short run** (5 minutes, ~$0.10):
   ```bash
   python launch_sagemaker_job.py \
     --s3-bucket YOUR-BUCKET \
     --num-training-batches 1000 \
     --instance-type ml.g4dn.xlarge
   ```

2. **Full training run** (25 hours, ~$35):
   ```bash
   python launch_sagemaker_job.py \
     --s3-bucket YOUR-BUCKET \
     --config-file rnn_args_diphone_sagemaker.yaml \
     --instance-type ml.g5.2xlarge
   ```

3. **Evaluate results**: Download checkpoint and run evaluation script

## Summary

✅ **What you have**:
- Complete S3-streaming training pipeline
- No local storage requirements
- Production-ready SageMaker integration
- Optimized for 1681-class diphone training

✅ **What you need**:
- AWS account with SageMaker access
- Data uploaded to S3
- 5 minutes to launch first job

✅ **What it costs**:
- ~$35 for full training run
- ~$0.10 for test run

**You're ready to train!** 🚀

## Getting Help

- **SageMaker docs**: https://docs.aws.amazon.com/sagemaker/
- **S3FS docs**: https://s3fs.readthedocs.io/
- **Detailed guide**: See `model_training/README_SAGEMAKER.md`
- **Quick commands**: See `model_training/SAGEMAKER_QUICK_START.md`

