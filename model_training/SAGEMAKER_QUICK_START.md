# SageMaker Quick Start Guide

## 1. Upload Data to S3

```bash
aws s3 sync ../data/hdf5_data_diphone_encoded s3://YOUR-BUCKET/data/hdf5_data_diphone_encoded
```

## 2. Install Dependencies Locally

```bash
pip install sagemaker boto3 s3fs
```

## 3. Launch Training Job

```bash
python launch_sagemaker_job.py \
  --s3-bucket YOUR-BUCKET \
  --s3-data-prefix data/hdf5_data_diphone_encoded \
  --instance-type ml.g5.2xlarge \
  --config-file rnn_args.yaml
```

## 4. Monitor Progress

```bash
# View logs (replace JOB-NAME with actual job name from step 3)
aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix JOB-NAME
```

## 5. Download Results

```bash
# After training completes
aws s3 sync s3://YOUR-BUCKET/sagemaker-outputs/JOB-NAME ./results/
cd results
tar -xzf output/model.tar.gz
```

## Configuration for Diphone Training

Use `rnn_args.yaml` and modify these key parameters:

```yaml
model:
  n_units: 1024        # Increased from 768
  rnn_dropout: 0.5     # Increased from 0.4
  
dataset:
  n_classes: 1681      # Changed from 41
  batch_size: 48       # Reduced from 64 (adjust based on GPU)
  dataset_dir: /dummy  # Will be overridden by S3 paths

num_training_batches: 150000  # Increased from 120000
lr_max: 0.004                 # Slightly reduced from 0.005
weight_decay: 0.002           # Increased from 0.001
```

## Cost Estimate

- **Instance**: ml.g5.2xlarge @ $1.21/hour
- **Duration**: ~25 hours for 150K batches
- **Total**: ~$35

## Troubleshooting

### Out of Memory
```bash
--batch-size 32  # Add to launch command
```

### S3 Access Denied
Check SageMaker execution role has S3 read/write permissions.

### Slow Training
- Ensure S3 bucket is in same region as SageMaker
- Try reducing `num_dataloader_workers` to 2

## Key Files

- `dataset_s3.py` - S3-enabled dataset
- `rnn_trainer_s3.py` - S3-enabled trainer  
- `train_model_sagemaker.py` - Training entry point
- `launch_sagemaker_job.py` - Job launcher
- `requirements_sagemaker.txt` - Dependencies

That's it! 🚀

