# MLflow with SageMaker Training Jobs

## Quick Answer

**By default, NO** - `launch_sagemaker_job.py` does NOT use MLflow integration. 

But now you have **3 options** to choose from:

---

## Option 1: Local Training with MLflow ⭐ (Easiest)

**Best for:** Quick experiments, debugging, small-scale training

Run training directly on your SageMaker notebook instance with MLflow tracking:

```bash
# 1. Start MLflow server (one-time)
cd /home/ec2-user/SageMaker/brain-to-text-sagemaker/model_training
python setup_mlflow_sagemaker.py

# 2. Run training locally
python train_model_sagemaker_mlflow.py \
    --s3-bucket 4k-eugene-btt \
    --s3-data-prefix hdf5_data_diphone_encoded \
    --config-file rnn_args_diphone_sagemaker.yaml \
    --mlflow-tracking-uri http://localhost:5000 \
    --mlflow-experiment-name diphone-local-v1 \
    --gpu-number 0

# 3. View results in MLflow UI
# Open: http://localhost:5000
```

**Pros:**
- ✅ Easy setup (5 minutes)
- ✅ Real-time monitoring
- ✅ No extra infrastructure needed
- ✅ Perfect for experimentation

**Cons:**
- ❌ Limited to notebook instance resources
- ❌ Training stops if you close notebook
- ❌ Can't scale to multiple GPUs

**Cost:** Uses your existing notebook instance (no additional charges)

---

## Option 2: SageMaker Job WITHOUT MLflow (Current Default)

**Best for:** Production training, long-running jobs, maximum scalability

Standard SageMaker training job without MLflow:

```bash
python launch_sagemaker_job.py \
    --s3-bucket 4k-eugene-btt \
    --s3-data-prefix hdf5_data_diphone_encoded \
    --config-file rnn_args_diphone_sagemaker.yaml \
    --instance-type ml.g4dn.2xlarge
```

**Pros:**
- ✅ Scales to any instance size
- ✅ Runs independently (can close notebook)
- ✅ Managed infrastructure
- ✅ Job-level logging via CloudWatch

**Cons:**
- ❌ No MLflow experiment tracking
- ❌ Manual log parsing
- ❌ Harder to compare runs

**Cost:** SageMaker training instance charges (~$0.94/hr for ml.g4dn.2xlarge)

---

## Option 3: SageMaker Job WITH MLflow (Advanced) 🚀

**Best for:** Production experiments, team collaboration, long-term tracking

SageMaker training job that logs to a remote MLflow server:

### Prerequisites

1. **Set up a remote MLflow server** (one-time setup):

#### Option A: Run MLflow on your notebook instance (accessible to SageMaker jobs)

```bash
# On your SageMaker notebook terminal

# Start MLflow with public IP binding
mlflow server \
    --backend-store-uri sqlite:///~/mlflow.db \
    --default-artifact-root ~/mlruns \
    --host 0.0.0.0 \
    --port 5000 &

# Get your notebook instance IP
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
NOTEBOOK_IP=`curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/local-ipv4`
echo "MLflow URI: http://$NOTEBOOK_IP:5000"
```

#### Option B: Deploy dedicated MLflow server on EC2 (recommended for production)

See detailed guide in `mlflow_setup.md`

2. **Configure security groups** to allow SageMaker training instances to reach your MLflow server on port 5000.

### Launch Training with MLflow

```bash
python launch_sagemaker_job.py \
    --s3-bucket 4k-eugene-btt \
    --s3-data-prefix hdf5_data_diphone_encoded \
    --config-file rnn_args_diphone_sagemaker.yaml \
    --instance-type ml.g4dn.2xlarge \
    --use-mlflow \
    --mlflow-tracking-uri http://YOUR_MLFLOW_SERVER_IP:5000 \
    --mlflow-experiment-name diphone-sagemaker-v1
```

**Pros:**
- ✅ Scalable training (any instance size)
- ✅ Centralized experiment tracking
- ✅ Runs independently
- ✅ Team can view results in real-time

**Cons:**
- ❌ Requires MLflow server setup
- ❌ Need to configure networking/security groups
- ❌ More complex infrastructure

**Cost:** 
- SageMaker training: ~$0.94/hr (ml.g4dn.2xlarge)
- MLflow server (if on EC2): ~$0.10-0.50/hr depending on instance

---

## Comparison Table

| Feature | Option 1: Local w/ MLflow | Option 2: SageMaker (no MLflow) | Option 3: SageMaker w/ MLflow |
|---------|---------------------------|----------------------------------|-------------------------------|
| **Experiment Tracking** | ✅ Yes | ❌ No | ✅ Yes |
| **Scalability** | ❌ Limited | ✅ High | ✅ High |
| **Setup Complexity** | 🟢 Easy | 🟢 Easy | 🟠 Medium |
| **Cost** | $ | $$ | $$$ |
| **Best For** | Experimentation | Production training | Production + tracking |
| **Infrastructure** | Notebook only | SageMaker only | SageMaker + MLflow server |

---

## Recommendations

### For Your Current Stage (Diphone Training Experiments)

**Start with Option 1 (Local with MLflow)**:
1. Quick to set up
2. Perfect for hyperparameter tuning
3. Real-time monitoring
4. Easy to iterate

**Then move to Option 2** for final long training runs:
1. Full scalability
2. Can use larger instances
3. Doesn't tie up your notebook

**Consider Option 3** when:
1. You have multiple team members
2. Running many long experiments
3. Need centralized tracking for production

---

## Example Workflow

```bash
# Phase 1: Quick experiments with MLflow (2-3 hours each)
python setup_mlflow_sagemaker.py  # One-time
python train_model_sagemaker_mlflow.py \
    --s3-bucket 4k-eugene-btt \
    --s3-data-prefix hdf5_data_diphone_encoded \
    --config-file rnn_args_diphone_sagemaker.yaml \
    --mlflow-experiment-name diphone-debug \
    --num-training-batches 1000  # Short runs

# View results, tune hyperparameters
# Open http://localhost:5000

# Phase 2: Final training on SageMaker (12+ hours)
python launch_sagemaker_job.py \
    --s3-bucket 4k-eugene-btt \
    --s3-data-prefix hdf5_data_diphone_encoded \
    --config-file rnn_args_diphone_sagemaker.yaml \
    --instance-type ml.g4dn.2xlarge
    # No MLflow, full training
```

---

## Current Status

- ✅ **Updated** `launch_sagemaker_job.py` to support `--use-mlflow` flag
- ✅ Created `train_model_sagemaker_mlflow.py` for MLflow integration
- ✅ Created `setup_mlflow_sagemaker.py` for easy MLflow setup
- ⚠️ **By default, MLflow is NOT used** when running `launch_sagemaker_job.py`

---

## Next Steps

Choose your approach:

**Option 1 (Recommended to start):**
```bash
python setup_mlflow_sagemaker.py
python train_model_sagemaker_mlflow.py --s3-bucket 4k-eugene-btt --s3-data-prefix hdf5_data_diphone_encoded --config-file rnn_args_diphone_sagemaker.yaml
```

**Option 2 (Current behavior):**
```bash
python launch_sagemaker_job.py --s3-bucket 4k-eugene-btt --s3-data-prefix hdf5_data_diphone_encoded
```

