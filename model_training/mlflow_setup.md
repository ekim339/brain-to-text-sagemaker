# MLflow Integration for SageMaker Training

## Setup on SageMaker Notebook Instance

### 1. Install MLflow (Run in notebook terminal)

```bash
pip install mlflow
```

### 2. Start MLflow Tracking Server (Run in notebook terminal)

```bash
# Create directory for MLflow artifacts
mkdir -p ~/mlruns

# Start MLflow server (runs in background)
nohup mlflow server \
    --backend-store-uri sqlite:///~/mlflow.db \
    --default-artifact-root ~/mlruns \
    --host 0.0.0.0 \
    --port 5000 > ~/mlflow.log 2>&1 &

# Check it's running
ps aux | grep mlflow
```

### 3. Access MLflow UI

Open in browser:
- **Local**: `http://localhost:5000`
- **Remote**: `http://<your-sagemaker-instance-ip>:5000`
  - You may need to configure security groups to allow port 5000

### 4. Stop MLflow Server (when needed)

```bash
pkill -f "mlflow server"
```

---

## What Gets Logged to MLflow

### Parameters (Hyperparameters)
- Learning rates (max, min, warmup steps)
- Batch size
- Model architecture (hidden dims, layers, dropout)
- Number of classes (diphones)
- Optimizer settings (weight decay, epsilon)
- Data augmentation settings

### Metrics (Per Training Step)
- Training loss
- Validation loss
- Validation DER (Diphone Error Rate)
- Validation PER (Phoneme Error Rate)
- Learning rate
- Gradient norm

### Artifacts
- Best model checkpoint
- Final model checkpoint
- Training configuration (YAML)
- Final metrics (JSON)

---

## Viewing Results

Once training is running, you can:

1. **Open MLflow UI**: Navigate to `http://localhost:5000`
2. **Compare runs**: View all experiments side-by-side
3. **Plot metrics**: Interactive charts for loss/PER over time
4. **Download models**: Download best checkpoints from any run

---

## Tips

### Run Multiple Experiments
Each SageMaker job will create a new run in MLflow with a unique run ID.

### Custom Experiment Names
Set environment variable before launching:
```bash
export MLFLOW_EXPERIMENT_NAME="diphone-training-v2"
```

### Remote Tracking Server
For production, consider setting up a dedicated MLflow server on EC2 and setting:
```python
mlflow.set_tracking_uri("http://your-mlflow-server:5000")
```

