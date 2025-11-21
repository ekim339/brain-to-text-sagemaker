"""
Example: How to use binTensor preprocessing in your data pipeline

There are three main approaches:
1. Add binning as a configurable option in the dataset class
2. Apply binning during data loading (in __getitem__)
3. Preprocess data offline before training
"""

import torch
from preprocessing_utils import binTensor, apply_binning_to_batch, bin_trial_data


# ============================================================================
# APPROACH 1: Add binning to dataset class (recommended for training)
# ============================================================================

# In your dataset.py, modify __init__ to accept bin_size parameter:
"""
class BrainToTextDataset(Dataset):
    def __init__(
        self,
        trial_indicies,
        n_batches,
        split='train',
        batch_size=64,
        days_per_batch=1,
        random_seed=-1,
        must_include_days=None,
        feature_subset=None,
        bin_size=1,  # NEW: Add this parameter
    ):
        # ... existing code ...
        self.bin_size = bin_size  # Store bin size
"""

# Then in __getitem__, apply binning after padding:
"""
def __getitem__(self, idx):
    # ... existing code to load and pad data ...
    
    # After padding:
    batch['input_features'] = pad_sequence(batch['input_features'], batch_first=True, padding_value=0)
    
    # NEW: Apply binning if enabled
    if self.bin_size > 1:
        batch = apply_binning_to_batch(batch, self.bin_size, update_time_steps=True)
    
    return batch
"""


# ============================================================================
# APPROACH 2: Apply binning to individual trials during loading
# ============================================================================

def load_and_bin_trial(session_path, trial_idx, bin_size=2):
    """Load a single trial and bin it before adding to batch."""
    import h5py
    
    with h5py.File(session_path, 'r') as f:
        g = f[f'trial_{trial_idx:04d}']
        input_features = torch.from_numpy(g['input_features'][:])  # (T, N)
        
        # Bin the trial data
        if bin_size > 1:
            input_features = bin_trial_data(input_features, bin_size)
        
        return input_features


# ============================================================================
# APPROACH 3: Preprocess entire dataset offline (for one-time preprocessing)
# ============================================================================

def preprocess_hdf5_file(input_path, output_path, bin_size=2):
    """
    Preprocess an entire HDF5 file by binning all trials.
    This creates a new HDF5 file with binned data.
    """
    import h5py
    import numpy as np
    
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # Copy metadata
        for key, value in f_in.attrs.items():
            f_out.attrs[key] = value
        
        # Process each trial
        for trial_name in f_in.keys():
            if not trial_name.startswith('trial_'):
                continue
            
            g_in = f_in[trial_name]
            g_out = f_out.create_group(trial_name)
            
            # Load original data
            input_features = g_in['input_features'][:]  # (T, N)
            
            # Bin the data
            if bin_size > 1:
                # Add batch dimension, bin, remove batch dimension
                input_features_batched = input_features[np.newaxis, :, :]  # (1, T, N)
                input_features_binned = binTensor(input_features_batched, bin_size)  # (1, T_binned, N)
                input_features = input_features_binned[0]  # (T_binned, N)
                
                # Update time steps attribute
                original_time_steps = g_in.attrs.get('n_time_steps', input_features.shape[0])
                g_out.attrs['n_time_steps'] = int(np.floor(original_time_steps / bin_size))
            else:
                g_out.attrs['n_time_steps'] = g_in.attrs.get('n_time_steps', input_features.shape[0])
            
            # Copy other data unchanged
            g_out.create_dataset('input_features', data=input_features)
            g_out.create_dataset('seq_class_ids', data=g_in['seq_class_ids'][:])
            if 'seq_class_ids_phoneme' in g_in:
                g_out.create_dataset('seq_class_ids_phoneme', data=g_in['seq_class_ids_phoneme'][:])
            g_out.create_dataset('transcription', data=g_in['transcription'][:])
            
            # Copy other attributes
            for attr_key in g_in.attrs:
                if attr_key != 'n_time_steps':  # Already updated
                    g_out.attrs[attr_key] = g_in.attrs[attr_key]
    
    print(f"Preprocessed {input_path} -> {output_path} with bin_size={bin_size}")


# ============================================================================
# APPROACH 4: Apply binning in the trainer (after batch is loaded)
# ============================================================================

def apply_binning_in_trainer(batch, bin_size):
    """
    Apply binning in the trainer's training loop.
    This is less efficient but gives you flexibility.
    """
    if bin_size > 1:
        batch = apply_binning_to_batch(batch, bin_size, update_time_steps=True)
    return batch


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Bin a single tensor
    data = torch.randn(32, 100, 512)  # (batch_size, time_steps, features)
    binned = binTensor(data, bin_size=2)  # (32, 50, 512)
    print(f"Original shape: {data.shape}, Binned shape: {binned.shape}")
    
    # Example 2: Bin a batch dictionary
    batch = {
        'input_features': torch.randn(32, 100, 512),
        'n_time_steps': torch.tensor([100] * 32),
    }
    binned_batch = apply_binning_to_batch(batch, bin_size=2)
    print(f"Binned batch time steps: {binned_batch['n_time_steps']}")
    
    # Example 3: Compare binning vs patching
    print("\n" + "="*60)
    print("BINNING vs PATCHING:")
    print("="*60)
    print("BINNING (preprocessing):")
    print("  - Reduces sequence length BEFORE model")
    print("  - Averages time steps (mean pooling)")
    print("  - Applied to raw neural data")
    print("  - Example: 100 timesteps -> 50 timesteps (bin_size=2)")
    print()
    print("PATCHING (in-model):")
    print("  - Reduces sequence length INSIDE model")
    print("  - Concatenates time steps (no averaging)")
    print("  - Applied after day-specific layers")
    print("  - Example: 100 timesteps -> 25 patches (patch_size=14, stride=4)")
    print()
    print("You can use BOTH:")
    print("  - Bin first (e.g., bin_size=2) to reduce from 100->50")
    print("  - Then patch (e.g., patch_size=14, stride=4) to reduce 50->12")

