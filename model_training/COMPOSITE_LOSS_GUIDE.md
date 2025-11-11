# Composite Loss for Diphone and Phoneme Training

## Overview

This implementation allows you to balance between **diphone loss** and **phoneme loss** during training using a weighted combination:

```
L = α × (phoneme_loss) + (1 - α) × (diphone_loss)
```

Where:
- `α` (alpha) controls the balance between the two losses
- `α = 0.0`: Only diphone loss (original behavior)
- `α = 0.5`: Equal weight to both losses
- `α = 1.0`: Only phoneme loss

## How It Works

### 1. **Diphone Loss** (DER-focused)
- Uses the model's direct diphone logits output (1681 classes)
- Calculates CTC loss against ground truth diphone sequences
- Optimizes for accurate diphone predictions

### 2. **Phoneme Loss** (PER-focused)
- Marginalizes diphone logits to phoneme logits (41 classes) using log-sum-exp
- Calculates CTC loss against ground truth phoneme sequences
- Optimizes for accurate phoneme predictions

### 3. **Marginalization Process**
For each phoneme P:
```python
phoneme_logit[P] = log_sum_exp(all diphone_logits ending in P)
```

This properly combines all diphone probabilities that end in the same phoneme.

## Configuration

### Option 1: Fixed Alpha (Simple)

```yaml
# Composite loss: L = alpha * (phoneme_loss) + (1 - alpha) * (diphone_loss)
use_composite_loss: true  # Enable composite loss
composite_loss_alpha: 0.5  # Weight for phoneme loss (0.0-1.0)
use_alpha_schedule: false  # Use fixed alpha
```

### Option 2: Dynamic Alpha Schedule (Curriculum Learning - Recommended!)

```yaml
# Composite loss with dynamic alpha schedule
use_composite_loss: true
composite_loss_alpha: 0.6  # Final target (ignored if using schedule)
use_alpha_schedule: true  # Enable dynamic schedule
alpha_schedule_start: 0.0  # Start with diphone-only
alpha_schedule_end: 0.6  # Gradually increase to 60% phoneme
alpha_schedule_step_size: 0.1  # Increase by 0.1
alpha_schedule_step_interval: 3000  # Every 3000 batches
```

### Parameters:

**Basic Parameters:**
- **`use_composite_loss`**: Set to `true` to enable, `false` for diphone-only
- **`composite_loss_alpha`**: Float between 0.0 and 1.0 (used if schedule is disabled)

**Schedule Parameters (Curriculum Learning):**
- **`use_alpha_schedule`**: Enable dynamic alpha that increases over training
- **`alpha_schedule_start`**: Starting alpha value (typically 0.0)
- **`alpha_schedule_end`**: Maximum alpha value (e.g., 0.6)
- **`alpha_schedule_step_size`**: How much to increase alpha at each step (e.g., 0.1)
- **`alpha_schedule_step_interval`**: How many batches between increases (e.g., 3000)

## Training Logs

### With Dynamic Alpha Schedule

```
Train batch 0: loss: 152.45 (diphone: 152.45, phoneme: 138.01, α: 0.00) grad norm: 25.34 lr: 0.000250 time: 0.982
Train batch 3000: loss: 147.23 (diphone: 152.45, phoneme: 138.01, α: 0.10) grad norm: 23.12 lr: 0.000400 time: 0.965
Train batch 6000: loss: 145.67 (diphone: 152.45, phoneme: 138.01, α: 0.20) grad norm: 22.45 lr: 0.000450 time: 0.978
Train batch 18000: loss: 142.34 (diphone: 152.45, phoneme: 138.01, α: 0.60) grad norm: 21.89 lr: 0.000480 time: 0.971
```

### Log Fields:
- **`loss`**: Combined loss = α × phoneme + (1 - α) × diphone
- **`diphone`**: Diphone-only CTC loss component
- **`phoneme`**: Phoneme-only CTC loss component
- **`α`**: Current alpha value (shows schedule progression)

## Benefits

### Why Use Composite Loss?

1. **Balanced Optimization**: Directly optimize for both DER and PER simultaneously
2. **Better Phoneme Accuracy**: Explicitly trains the model to produce phoneme-consistent predictions
3. **Flexible Trade-offs**: Adjust `α` to prioritize diphone or phoneme accuracy based on your needs
4. **No Architecture Changes**: Uses the same model architecture, just changes the training objective

### Recommended Alpha Values

**For Fixed Alpha:**
- **α = 0.3-0.4**: Prioritize diphone accuracy while still improving phonemes
- **α = 0.5**: Equal balance (good starting point)
- **α = 0.6-0.7**: Prioritize phoneme accuracy
- **α = 0.0**: Baseline (diphone-only, original behavior)

**For Dynamic Schedule (Curriculum Learning - RECOMMENDED):**
- Start at **α = 0.0** (learn diphone patterns first)
- End at **α = 0.6** (gradually shift to phoneme optimization)
- This allows the model to:
  1. First learn the easier diphone task (more classes but direct supervision)
  2. Then refine phoneme predictions (harder marginalization task)
  3. Build phoneme accuracy on top of solid diphone foundation

## Alpha Schedule Visualization

With default settings (start=0.0, end=0.6, step=0.1, interval=3000):

```
Batch Range    | Alpha | Loss Composition
---------------|-------|----------------------------------
0 - 2,999      | 0.0   | 100% diphone, 0% phoneme
3,000 - 5,999  | 0.1   | 90% diphone, 10% phoneme
6,000 - 8,999  | 0.2   | 80% diphone, 20% phoneme
9,000 - 11,999 | 0.3   | 70% diphone, 30% phoneme
12,000 - 14,999| 0.4   | 60% diphone, 40% phoneme
15,000 - 17,999| 0.5   | 50% diphone, 50% phoneme
18,000+        | 0.6   | 40% diphone, 60% phoneme (capped)
```

This curriculum learning approach helps the model learn progressively from easier (diphone) to harder (phoneme) tasks.

## Implementation Details

### Code Changes

1. **Config** (`rnn_args_diphone_sagemaker.yaml`):
   - Added `use_composite_loss` and `composite_loss_alpha` parameters

2. **Trainer** (`rnn_trainer.py`):
   - Added `marginalize_diphone_logits_to_phoneme_logits()` method
   - Modified training loop to load phoneme labels
   - Calculate both diphone and phoneme losses
   - Combine losses with weighted sum
   - Enhanced logging to show loss components

3. **Datasets** (`dataset.py`, `dataset_s3.py`):
   - Load `seq_class_ids_phoneme` (ground truth phoneme sequences)
   - Load `phone_seq_lens_phoneme` (phoneme sequence lengths)

### Marginalization Algorithm

```python
def marginalize_diphone_logits_to_phoneme_logits(diphone_logits):
    # For each phoneme (0-40)
    for curr_phoneme in range(41):
        # Find all diphones ending in this phoneme
        # diphone_idx = prev_phoneme * 41 + curr_phoneme
        diphone_indices = [prev * 41 + curr_phoneme for prev in range(41)]
        
        # Log-sum-exp marginalization
        phoneme_logits[:, :, curr_phoneme] = logsumexp(
            diphone_logits[:, :, diphone_indices], dim=-1
        )
```

## Usage Example

### Training with Composite Loss (Equal Balance)

```bash
python launch_sagemaker_job.py \
    --s3-data-prefix hdf5_data_encoded \
    --config-file rnn_args_diphone_sagemaker.yaml \
    --instance-type ml.g4dn.xlarge
```

In your config:
```yaml
use_composite_loss: true
composite_loss_alpha: 0.5  # 50% phoneme, 50% diphone
```

### Training with Phoneme Priority

```yaml
use_composite_loss: true
composite_loss_alpha: 0.7  # 70% phoneme, 30% diphone
```

### Training with Diphone Priority (Baseline)

```yaml
use_composite_loss: false  # Or set alpha to 0.0
```

## Experiments to Try

1. **Alpha Sweep**: Train with α = 0.0, 0.3, 0.5, 0.7, 1.0
2. **Dynamic Alpha**: Start with high α (phoneme-focused) and decrease over training
3. **Task-Specific**: Use α = 0.7 if PER is more important for your application

## Expected Results

With composite loss enabled:
- **DER**: May slightly decrease compared to diphone-only training
- **PER**: Should significantly improve compared to diphone-only training
- **Total**: Better overall phoneme recognition with minimal diphone accuracy trade-off

## Monitoring

Track these metrics during training:
- Combined loss (total)
- Diphone loss component
- Phoneme loss component
- Validation DER
- Validation PER

This helps you understand:
- Which loss component is harder to optimize
- Whether the model is balancing both objectives
- If you need to adjust α

## Notes

- The model architecture remains unchanged (still outputs 1681 diphone classes)
- Phoneme logits are computed on-the-fly during training via marginalization
- Validation metrics (DER and PER) are computed the same way regardless of loss type
- You can switch between composite and diphone-only loss at any time by changing the config

