"""
Preprocessing utilities for neural data.
Includes binTensor function from NeuralDecoder for time-binning.
"""
import numpy as np
import torch


def binTensor(data, binSize):
    """
    A simple utility function to bin a 3d numpy tensor along axis 1 (the time axis here). 
    Data is binned by taking the mean across a window of time steps. 
    
    Args:
        data (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        binSize (int): The bin size in # of time steps
        
    Returns:
        binnedTensor (tensor : B x S x N): A 3d tensor with batch size B, time bins S, and number of features N.
                                           S = floor(T/binSize)
    
    Note: This is ported from NeuralDecoder/neuralDecoder/utils/preprocessing.py
    """
    # Convert torch tensor to numpy if needed
    is_torch = isinstance(data, torch.Tensor)
    if is_torch:
        data_np = data.cpu().numpy()
    else:
        data_np = np.array(data)
    
    nBins = np.floor(data_np.shape[1] / binSize).astype(int)
    
    sh = np.array(data_np.shape)
    sh[1] = nBins
    binnedTensor = np.zeros(sh)
    
    binIdx = np.arange(0, binSize).astype(int)
    for t in range(nBins):
        binnedTensor[:, t, :] = np.mean(data_np[:, binIdx, :], axis=1)
        binIdx += binSize
    
    # Convert back to torch tensor if input was torch
    if is_torch:
        return torch.from_numpy(binnedTensor).to(data.device)
    return binnedTensor


def apply_binning_to_batch(batch, bin_size, update_time_steps=True):
    """
    Apply time-binning to a batch of neural data.
    
    Args:
        batch (dict): Batch dictionary with 'input_features' key
                     Shape: (batch_size, time_steps, n_features)
        bin_size (int): Number of time steps to average together
        update_time_steps (bool): If True, update 'n_time_steps' to reflect new length
        
    Returns:
        batch (dict): Modified batch with binned input_features
    """
    if bin_size <= 1:
        return batch  # No binning needed
    
    # Bin the input features
    batch['input_features'] = binTensor(batch['input_features'], bin_size)
    
    # Update time steps if requested
    if update_time_steps and 'n_time_steps' in batch:
        # Calculate new time steps: floor(original / bin_size)
        batch['n_time_steps'] = (batch['n_time_steps'] / bin_size).long()
    
    return batch


def bin_trial_data(input_features, bin_size):
    """
    Bin a single trial's neural data.
    
    Args:
        input_features (torch.Tensor or np.ndarray): Shape (time_steps, n_features)
        bin_size (int): Number of time steps to average together
        
    Returns:
        binned_features (torch.Tensor or np.ndarray): Shape (binned_time_steps, n_features)
    """
    if bin_size <= 1:
        return input_features
    
    # Add batch dimension for binTensor (expects B x T x N)
    if len(input_features.shape) == 2:
        input_features = input_features.unsqueeze(0) if isinstance(input_features, torch.Tensor) else input_features[np.newaxis, :, :]
    
    # Apply binning
    binned = binTensor(input_features, bin_size)
    
    # Remove batch dimension
    return binned.squeeze(0)

