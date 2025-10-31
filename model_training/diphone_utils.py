"""
Utility functions for diphone encoding and decoding.
"""

import numpy as np


NUM_PHONEMES = 41  # Including BLANK (0)

def marginalize_diphone_probabilities(diphone_probs):
    """
    Marginalize the log probabilities over the diphone dimension.
    """
    r, c, p = diphone_probs.shape
    phoneme_probs = np.zeros((r, c, NUM_PHONEMES))

    for i in range(NUM_PHONEMES):
        phoneme_probs[:, :, i] = np.sum(diphone_probs[:, :, i::NUM_PHONEMES], axis=-1)
    return phoneme_probs


def diphone_sequence_to_phonemes(diphone_sequence):
    """
    Convert a sequence of diphone IDs back to phoneme IDs.
    
    Diphone encoding format:
    - Values < 41: Single phoneme (used for first phoneme)
    - Values >= 41: Transition diphone = prev_phoneme * 41 + curr_phoneme
    
    Args:
        diphone_sequence: np.array of diphone IDs (after CTC decoding)
        
    Returns:
        np.array of phoneme IDs
        
    Example:
        >>> diphone_seq = [1, 42, 43, 84, 85, 126, 123]  # Encoded: [AA, AA_AA, AA_B, B_B, B_IY, IY_IY, IY_0]
        >>> phoneme_seq = diphone_sequence_to_phonemes(diphone_seq)
        >>> print(phoneme_seq)  # [1, 2, 3]  # Decoded: [AA, B, IY]
    """
    if len(diphone_sequence) == 0:
        return np.array([], dtype=np.int64)
    
    phonemes = []
    
    for diphone_id in diphone_sequence:
        if diphone_id < NUM_PHONEMES:
            # Single phoneme encoding (used for first phoneme)
            phonemes.append(int(diphone_id))
        else:
            # Diphone encoding: extract the second (current) phoneme
            # diphone_id = prev_phoneme * NUM_PHONEMES + curr_phoneme
            curr_phoneme = int(diphone_id % NUM_PHONEMES)
            phonemes.append(curr_phoneme)
    
    if len(phonemes) == 0:
        return np.array([], dtype=np.int64)
    
    # Remove consecutive duplicate phonemes
    unique_phonemes = [phonemes[0]]
    for p in phonemes[1:]:
        if p != unique_phonemes[-1]:
            unique_phonemes.append(p)
    
    # Remove blanks (phoneme ID 0) and silence markers
    unique_phonemes = [p for p in unique_phonemes if p != 0]
    
    return np.array(unique_phonemes, dtype=np.int64)


def test_diphone_conversion():
    """Test the conversion function."""
    # Test case 1: Simple sequence
    # Phonemes: [1, 2, 3] (AA, AE, AH)
    # Diphones: [1, 42, 43, 84, 85, 126, 123]
    diphones = np.array([1, 42, 43, 84, 85, 126, 123])
    phonemes = diphone_sequence_to_phonemes(diphones)
    expected = np.array([1, 2, 3])
    assert np.array_equal(phonemes, expected), f"Expected {expected}, got {phonemes}"
    print("✅ Test 1 passed")
    
    # Test case 2: Empty sequence
    diphones = np.array([])
    phonemes = diphone_sequence_to_phonemes(diphones)
    assert len(phonemes) == 0
    print("✅ Test 2 passed")
    
    # Test case 3: With blanks
    diphones = np.array([0, 1, 0, 42, 43])  # Contains blanks
    phonemes = diphone_sequence_to_phonemes(diphones)
    expected = np.array([1, 2])  # Blanks removed
    assert np.array_equal(phonemes, expected), f"Expected {expected}, got {phonemes}"
    print("✅ Test 3 passed")
    
    print("All tests passed! ✅")


if __name__ == "__main__":
    test_diphone_conversion()

