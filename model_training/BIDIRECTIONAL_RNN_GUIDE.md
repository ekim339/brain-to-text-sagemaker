# Bidirectional RNN Guide

## Overview

The model now supports **bidirectional RNN** mode. A bidirectional RNN processes sequences in both forward (left-to-right) and backward (right-to-left) directions, allowing the model to capture both past and future context.

## How to Enable

### In `rnn_args_diphone_sagemaker.yaml`:

```yaml
model:
  bidirectional: true  # Enable bidirectional RNN
```

Set to `false` for unidirectional (default).

## What Changes

### Unidirectional (bidirectional: false)
- Processes sequence **left-to-right only**
- At each timestep, model sees only **past** information
- **Suitable for real-time/streaming** applications
- Output size: `n_units`
- Parameters: ~standard

### Bidirectional (bidirectional: true)
- Processes sequence **both directions** (forward + backward)
- At each timestep, model sees **both past and future** information
- **Better accuracy**, but **NOT suitable for real-time use**
- Output size: `2 × n_units` (concatenated forward + backward)
- Parameters: **~2× more** (doubled)

## Architecture Changes

When `bidirectional=true`:

1. **GRU Layer**: 
   - Creates both forward and backward GRU paths
   - Doubles the number of parameters

2. **Output Layer**: 
   - Input size changes from `n_units` to `2 × n_units`
   - Automatically handles concatenated forward+backward states

3. **Hidden States**: 
   - Shape changes from `(n_layers, batch, n_units)` 
   - To: `(n_layers × 2, batch, n_units)`

## Performance Impact

### Computational Cost
- **Training time**: ~1.5-2× slower (two passes through sequence)
- **Memory**: ~2× more (doubled parameters + activations)
- **Inference**: Same slowdown applies

### Accuracy
- **DER**: Typically improves by 2-5%
- **PER**: Typically improves by 3-8%
- **Reason**: Model can "look ahead" to future context for better predictions

## When to Use

### ✅ Use Bidirectional When:
- You have **complete sequences** before decoding
- **Offline analysis** is acceptable
- **Maximum accuracy** is more important than speed
- You're doing **post-hoc research** on recorded data
- Training on **full trial data** where you can see the entire sequence

### ❌ Don't Use Bidirectional When:
- You need **real-time decoding** (streaming)
- **Low latency** is critical
- You're building a **BCI for live use**
- **Memory is limited** (bidirectional uses 2× parameters)
- You want faster training

## Example Configurations

### Configuration 1: Unidirectional (Default - Real-time Ready)

```yaml
model:
  n_units: 1024
  n_layers: 5
  bidirectional: false  # Unidirectional
  # Total parameters: ~standard
  # Output size: 1024
  # Suitable for: Real-time BCI applications
```

### Configuration 2: Bidirectional (Max Accuracy)

```yaml
model:
  n_units: 1024
  n_layers: 5
  bidirectional: true  # Bidirectional
  # Total parameters: ~2× standard
  # Output size: 2048 (1024 forward + 1024 backward)
  # Suitable for: Offline analysis, research
```

### Configuration 3: Smaller Bidirectional (Balanced)

```yaml
model:
  n_units: 512  # Halved to keep similar param count
  n_layers: 5
  bidirectional: true
  # Total parameters: ~similar to n_units=1024 unidirectional
  # Output size: 1024 (512 forward + 512 backward)
  # Suitable for: Better accuracy with similar param count
```

## Training Examples

### Train with Bidirectional

```bash
# Edit rnn_args_diphone_sagemaker.yaml:
# model:
#   bidirectional: true

python launch_sagemaker_job.py \
    --s3-data-prefix hdf5_data_encoded \
    --config-file rnn_args_diphone_sagemaker.yaml \
    --instance-type ml.g4dn.xlarge
```

### Compare Unidirectional vs Bidirectional

```bash
# Run 1: Unidirectional baseline
# bidirectional: false
python launch_sagemaker_job.py ... --job-name brain-to-text-unidirectional

# Run 2: Bidirectional for comparison
# bidirectional: true
python launch_sagemaker_job.py ... --job-name brain-to-text-bidirectional
```

## Monitoring

During training, you won't see explicit "bidirectional" labels in logs, but you can verify:

1. **Model Summary**: Check parameter count (should be ~2× if bidirectional)
2. **Memory Usage**: Should be higher for bidirectional
3. **Training Speed**: Should be slower (more time per batch)

## Expected Results

Based on typical NLP/speech recognition tasks:

| Metric | Unidirectional | Bidirectional | Improvement |
|--------|---------------|---------------|-------------|
| DER | 0.45 | 0.42 | ~7% better |
| PER | 0.38 | 0.33 | ~13% better |
| Training Speed | 1.0× | 0.5-0.6× | ~2× slower |
| Parameters | 1.0× | ~2.0× | Doubled |

*Note: Actual results will vary based on your specific dataset and configuration.*

## Common Issues

### Issue 1: Out of Memory
**Problem**: Bidirectional uses more memory  
**Solution**: 
- Reduce `batch_size` in config
- Reduce `n_units` (e.g., from 1024 to 512)
- Use a larger instance type

### Issue 2: Training Too Slow
**Problem**: Bidirectional is 2× slower  
**Solution**: 
- This is expected behavior
- Consider if the accuracy gain is worth the time
- Use a faster instance type (ml.p3 or ml.g5)

### Issue 3: Checkpoint Loading Fails
**Problem**: Loading unidirectional checkpoint into bidirectional model (or vice versa)  
**Solution**: 
- Models with different `bidirectional` settings are incompatible
- Train from scratch or use matching checkpoint

## Technical Details

### Forward Pass (Bidirectional)

```
Input: (batch, time, features)
  ↓
Day-specific layers
  ↓
Forward GRU: t=0 → t=T  →  Forward hidden states
Backward GRU: t=T → t=0  →  Backward hidden states
  ↓
Concatenate: [forward_hidden; backward_hidden]
  ↓  (batch, time, 2*n_units)
Output Linear Layer
  ↓
Logits: (batch, time, n_classes)
```

### Parameter Count

```
Unidirectional:
- GRU params: 3 × (input_size × n_units + n_units × n_units) × n_layers
- Output params: n_units × n_classes

Bidirectional:
- GRU params: 2 × [3 × (input_size × n_units + n_units × n_units) × n_layers]
- Output params: (2 × n_units) × n_classes
```

## Recommendations

### For Research/Analysis (Offline)
✅ **Use bidirectional = true**
- You want maximum accuracy
- You have complete trials
- Computational cost is acceptable

### For Real-time BCI Application
✅ **Use bidirectional = false**
- You need streaming/online decoding
- Low latency is critical
- You're building a deployable system

### For Exploration
✅ **Try both and compare**
- Run experiments with both settings
- Measure the accuracy-speed tradeoff
- Choose based on your specific needs

## Summary

Bidirectional RNNs offer **better accuracy** at the cost of:
- 2× more parameters
- 2× slower training/inference
- Cannot be used for real-time streaming

Enable with `bidirectional: true` in your config when offline accuracy is more important than real-time performance.

