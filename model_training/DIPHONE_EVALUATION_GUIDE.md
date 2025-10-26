# Diphone Model Evaluation Guide

## Overview

Your diphone model has **1681 classes** instead of 41, representing all possible pairs of consecutive phonemes.

## Key Differences: Phonemes vs Diphones

| Aspect | Phoneme Model | Diphone Model |
|--------|---------------|---------------|
| **Classes** | 41 | 1681 (41 × 41) |
| **Represents** | Single phoneme | Phoneme pair |
| **Example** | `['HH', 'EH', 'L', 'OW']` | `['HH-EH', 'EH-L', 'L-OW']` |
| **Advantages** | Simpler, faster | Captures coarticulation |
| **Disadvantages** | Ignores context | More classes, harder to train |

## Class Structure

### Phoneme Classes (41 total)
```python
Classes 0-40:
  0: BLANK
  1: AA
  2: AE
  3: AH
  ...
  40: |  (silence)
```

### Diphone Classes (1681 total)
```python
Classes 0-1680:
  0: BLANK
  1: AA-AA
  2: AA-AE
  3: AA-AH
  ...
  41: AA-|
  42: AE-AA
  43: AE-AE
  ...
  1680: |-|
```

## Model Output

```python
# Phoneme model
logits.shape = [batch_size, time_steps, 41]

# Diphone model
logits.shape = [batch_size, time_steps, 1681]
```

## Evaluation Pipeline

### Step 1: Model Inference (Same Process)

```python
# Both models work the same way
logits = model(neural_features, day_indicies)

# Greedy decoding
predictions = argmax(logits, dim=-1)
predictions = unique_consecutive(predictions)
predictions = remove_blanks(predictions)
```

### Step 2: Decode to Symbols

**Phoneme Model:**
```python
pred_phonemes = [LOGIT_TO_PHONEME[p] for p in predictions]
# Output: ['HH', 'EH', 'L', 'OW']
```

**Diphone Model:**
```python
pred_diphones = [LOGIT_TO_DIPHONE[p] for p in predictions]
# Output: ['HH-EH', 'EH-L', 'L-OW']

# Convert to phonemes for language model
pred_phonemes = diphone_sequence_to_phonemes(pred_diphones)
# Output: ['HH', 'EH', 'L', 'OW']
```

### Step 3: Language Model (Needs Phonemes)

The existing language model expects **phoneme** sequences, so we convert:

```python
# Convert diphones to phonemes
diphones = ['AA-AE', 'AE-HH', 'HH-EH']
phonemes = diphone_sequence_to_phonemes(diphones)
# Result: ['AA', 'AE', 'HH', 'EH']

# Then pass to language model
text = language_model(phonemes)
# Result: "hello"
```

## Metrics

### During Training

**Phoneme Model:**
- **PER (Phoneme Error Rate)**: Edit distance at phoneme level

**Diphone Model:**
- **DER (Diphone Error Rate)**: Edit distance at diphone level
- Can also compute PER after converting to phonemes

### After Language Model

Both models report the same final metric:
- **WER (Word Error Rate)**: Edit distance at word level

## Running Evaluation

### For Phoneme Model (41 classes)

```bash
python evaluate_model.py \
  --model_path trained_models/baseline_rnn \
  --data_dir ../data/hdf5_data_final \
  --eval_type val \
  --gpu_number 0
```

### For Diphone Model (1681 classes)

```bash
python evaluate_model_diphone.py \
  --model_path trained_models/diphone_rnn \
  --data_dir ../data/hdf5_data_diphone_encoded \
  --eval_type val \
  --gpu_number 0
```

## Example: Complete Flow

### Phoneme Model
```
Neural Data
    ↓
RNN Model (41 classes)
    ↓
Logits [batch, time, 41]
    ↓
Greedy Decode
    ↓
Phonemes: ['HH', 'EH', 'L', 'OW']
    ↓
Language Model
    ↓
Text: "hello"
```

### Diphone Model
```
Neural Data
    ↓
RNN Model (1681 classes)
    ↓
Logits [batch, time, 1681]
    ↓
Greedy Decode
    ↓
Diphones: ['HH-EH', 'EH-L', 'L-OW']
    ↓
Convert to Phonemes: ['HH', 'EH', 'L', 'OW']
    ↓
Language Model
    ↓
Text: "hello"
```

## Diphone → Phoneme Conversion

The conversion algorithm:

```python
def diphone_sequence_to_phonemes(diphone_seq):
    """
    Example: ['AA-AE', 'AE-HH', 'HH-EH'] -> ['AA', 'AE', 'HH', 'EH']
    
    Algorithm:
    1. Take first phoneme from first diphone: 'AA'
    2. For each diphone, take the second phoneme: 'AE', 'HH', 'EH'
    """
    if len(diphone_seq) == 0:
        return []
    
    phonemes = []
    
    # First phoneme from first diphone
    phonemes.append(diphone_seq[0].split('-')[0])
    
    # Second phoneme from each diphone
    for diphone in diphone_seq:
        phonemes.append(diphone.split('-')[1])
    
    return phonemes
```

## Expected Performance

### Diphone Error Rate (DER)
- **Initial**: ~95% (random)
- **Well-trained**: 30-40%
- **Target**: <25%

### Phoneme Error Rate (PER) 
After converting diphones to phonemes:
- **Expected**: Slightly better than pure phoneme model
- **Reason**: Diphones capture coarticulation

### Word Error Rate (WER)
Final metric after language model:
- **Target**: <15% (with good language model)
- **Clinical goal**: <5%

## Files Created for Diphone Evaluation

1. **`evaluate_model_helpers_diphone.py`**
   - Contains `LOGIT_TO_DIPHONE` (1681 classes)
   - Contains `diphone_sequence_to_phonemes()` converter
   - All helper functions adapted for diphones

2. **`evaluate_model_diphone.py`**
   - Main evaluation script for diphone models
   - Computes DER (Diphone Error Rate)
   - Computes PER (after conversion)
   - Saves both diphone and phoneme predictions

## Output Files

### Predictions CSV
```csv
session,block,trial,pred_diphones,pred_phonemes
t15.2023.08.13,1,0,HH-EH EH-L L-OW,HH EH L OW
t15.2023.08.13,1,1,W-ER ER-L L-D,W ER L D
...
```

### Metrics
```
Diphone Error Rate (DER): 35.42%
Phoneme Error Rate (PER): 32.18%
```

## Advantages of Diphones

1. **Coarticulation**: Captures how phonemes blend together
2. **Context**: Each unit includes information about neighboring phonemes
3. **Potentially Better**: May achieve lower WER with proper training

## Disadvantages

1. **More Classes**: 1681 vs 41 (harder to train)
2. **More Data**: Needs more training data to learn all combinations
3. **Conversion Needed**: Extra step to convert to phonemes for language model
4. **Sparse Classes**: Some diphone combinations may be very rare

## Next Steps

1. **Train diphone model** on SageMaker with 1681 classes
2. **Evaluate using** `evaluate_model_diphone.py`
3. **Compare metrics** with baseline phoneme model
4. **Integrate with language model** using phoneme conversion
5. **Measure final WER** to see if diphones help

## Summary

✅ **Key Point**: Your diphone model has 1681 classes (not 41)

✅ **Model structure**: Exactly the same as phoneme model, just different output dimension

✅ **Evaluation**: Use `evaluate_model_diphone.py` which handles the conversion

✅ **Final goal**: Lower WER through better coarticulation modeling

