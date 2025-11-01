# MLflow Integration for SageMaker Training

This guide shows you how to track your brain-to-text training experiments with MLflow on SageMaker.

---

## Quick Start (3 steps)

### 1️⃣ Setup MLflow (One-time)

```bash
cd /home/ec2-user/SageMaker/brain-to-text-sagemaker/model_training
python setup_mlflow_sagemaker.py
```

This will:
- Install MLflow
- Start MLflow tracking server on port 5000
- Verify it's running

### 2️⃣ Access MLflow UI

Open in your browser:
- **http://localhost:5000** (if on SageMaker notebook)

### 3️⃣ Run Training with MLflow

**Option A: Using the MLflow-enabled script**

```bash
python train_model_sagemaker_mlflow.py \
    --s3-bucket 4k-eugene-btt \
    --s3-data-prefix hdf5_data_diphone_encoded \
    --config-file rnn_args_diphone_sagemaker.yaml \
    --mlflow-tracking-uri http://localhost:5000 \
    --mlflow-experiment-name diphone-v1 \
    --gpu-number 0
```

**Option B: Run from Jupyter Notebook**

```python
import os

# Set MLflow environment variables
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'diphone-training-v1'

# Import and run trainer
from rnn_trainer_s3_mlflow import BrainToTextDecoder_Trainer_S3_MLflow
from omegaconf import OmegaConf

# Load config
args = OmegaConf.load('rnn_args_diphone_sagemaker.yaml')
args.s3_bucket = '4k-eugene-btt'
args.s3_data_prefix = 'hdf5_data_diphone_encoded'

# Train with MLflow logging
trainer = BrainToTextDecoder_Trainer_S3_MLflow(args)
metrics = trainer.train()
```

---

## What Gets Logged

MLflow automatically tracks:

### 📊 Metrics (Real-time)
- **Training loss** - every 10 batches
- **Validation DER** (Diphone Error Rate) - every validation step
- **Validation PER** (Phoneme Error Rate) - every validation step  
- **Validation loss** - every validation step
- **Learning rate** - tracks schedule
- **Gradient norm** - monitors stability

### ⚙️ Parameters (Hyperparameters)
- Model architecture (hidden_dim, n_layers, dropout)
- Learning rates (lr_max, lr_min, warmup_steps)
- Optimizer settings (weight_decay, epsilon)
- Batch size, number of classes (1681 for diphones)
- Data augmentation settings
- All config file parameters

### 📦 Artifacts
- Best model checkpoint
- Training configuration (YAML)
- Final metrics (JSON)

---

## Viewing Results

Once training is running:

1. **Open MLflow UI**: http://localhost:5000
2. **Select experiment**: e.g., "diphone-training-v1"
3. **View runs**: See all training runs with their metrics
4. **Compare runs**: Side-by-side comparison of hyperparameters and results
5. **Download models**: Get the best checkpoint from any run

### Example UI Views

**Experiments List**
- See all your experiments
- Compare different approaches (e.g., learning rates, batch sizes)

**Run Details**
- Full parameter list
- Interactive metric plots (loss vs. training steps)
- Per-session DER/PER breakdown
- Model checkpoints

**Compare Runs**
- Parallel coordinates plot
- Scatter plot of PER vs. batch size
- Table view of all hyperparameters

---

## Advanced Usage

### Multiple Experiments

Organize different approaches into separate experiments:

```bash
# Baseline experiment
python train_model_sagemaker_mlflow.py \
    --mlflow-experiment-name diphone-baseline \
    ...

# With data augmentation
python train_model_sagemaker_mlflow.py \
    --mlflow-experiment-name diphone-augmented \
    ...

# Different architecture
python train_model_sagemaker_mlflow.py \
    --mlflow-experiment-name diphone-large-model \
    ...
```

### Query Runs Programmatically

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri('http://localhost:5000')

# Get all runs from an experiment
runs = mlflow.search_runs(experiment_names=['diphone-training-v1'])

# Find best run
best_run = runs.loc[runs['metrics.best_val_PER'].idxmin()]
print(f"Best PER: {best_run['metrics.best_val_PER']:.4f}")
print(f"Best run ID: {best_run['run_id']}")

# Load best model
model_uri = f"runs:/{best_run['run_id']}/best_checkpoint"
# mlflow.pytorch.load_model(model_uri)  # If model was logged with mlflow.pytorch
```

### Remote MLflow Server (Production)

For production use, set up a dedicated MLflow server:

1. **Launch EC2 instance** with MLflow server
2. **Configure security group** to allow port 5000
3. **Use remote URI**:

```python
mlflow.set_tracking_uri('http://your-mlflow-server.com:5000')
```

---

## Troubleshooting

### MLflow server not starting

```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill existing MLflow processes
pkill -f "mlflow server"

# Restart
python setup_mlflow_sagemaker.py
```

### Can't access UI

```bash
# Check server is running
ps aux | grep mlflow

# Check logs
tail -f ~/mlflow.log

# Test connection
curl http://localhost:5000/health
```

### Metrics not showing up

- Make sure you're using `train_model_sagemaker_mlflow.py` (not the regular script)
- Check MLflow server is running
- Verify MLFLOW_TRACKING_URI environment variable is set

---

## Stop MLflow Server

When you're done:

```bash
pkill -f "mlflow server"
```

---

## Comparison: With vs. Without MLflow

| Feature | Without MLflow | With MLflow |
|---------|---------------|-------------|
| Track experiments | Manual log files | Automatic, centralized |
| Compare runs | Parse logs manually | Interactive UI |
| Find best model | Remember file paths | Query by metrics |
| Share results | Send log files | Share MLflow URL |
| Reproduce runs | Remember all settings | All params logged automatically |

---

## Files Created

- `rnn_trainer_s3_mlflow.py` - Trainer with MLflow logging
- `train_model_sagemaker_mlflow.py` - Training script with MLflow
- `setup_mlflow_sagemaker.py` - One-command setup
- `mlflow_setup.md` - Detailed setup guide
- `MLFLOW_README.md` - This file

---

## Next Steps

1. ✅ Run `python setup_mlflow_sagemaker.py`
2. ✅ Open http://localhost:5000 to see MLflow UI
3. ✅ Launch a training run
4. ✅ Watch metrics update in real-time
5. ✅ Compare different hyperparameters
6. ✅ Find your best model!

Happy tracking! 📊🚀

