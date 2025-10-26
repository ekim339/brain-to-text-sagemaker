"""
Evaluation helpers for DIPHONE model (1681 classes)
"""
import torch
import numpy as np
import h5py
import time
import re

from data_augmentations import gauss_smooth

# Original 41 phonemes (including silence)
PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    '|',  # Silence marker
]

# Generate all 1681 diphone combinations (41 x 41)
# Class 0 is BLANK (for CTC)
# Classes 1-1680 are diphones
LOGIT_TO_DIPHONE = ['BLANK']
DIPHONE_TO_LOGIT = {'BLANK': 0}

idx = 1
for p1 in PHONEMES:
    for p2 in PHONEMES:
        diphone = f'{p1}-{p2}'
        LOGIT_TO_DIPHONE.append(diphone)
        DIPHONE_TO_LOGIT[diphone] = idx
        idx += 1

assert len(LOGIT_TO_DIPHONE) == 1681, f"Expected 1681 diphones, got {len(LOGIT_TO_DIPHONE)}"

print(f"Created {len(LOGIT_TO_DIPHONE)} diphone classes")
print(f"Example diphones: {LOGIT_TO_DIPHONE[1:6]}")  # First few diphones


def diphone_sequence_to_phonemes(diphone_seq):
    """
    Convert a sequence of diphones back to phonemes
    
    Example:
        ['AA-AE', 'AE-HH', 'HH-EH'] -> ['AA', 'AE', 'HH', 'EH']
    
    Args:
        diphone_seq: List of diphone strings (e.g., ['AA-AE', 'AE-HH'])
    
    Returns:
        List of phoneme strings
    """
    if len(diphone_seq) == 0:
        return []
    
    phonemes = []
    
    # Add first phoneme from first diphone
    first_diphone = diphone_seq[0].split('-')
    phonemes.append(first_diphone[0])
    
    # For each diphone, add the second phoneme
    for diphone in diphone_seq:
        parts = diphone.split('-')
        if len(parts) == 2:
            phonemes.append(parts[1])
    
    return phonemes


def diphone_indices_to_phonemes(diphone_indices):
    """
    Convert diphone indices to phoneme sequence
    
    Args:
        diphone_indices: List of diphone class indices (e.g., [1, 42, 83])
    
    Returns:
        List of phoneme strings
    """
    # Convert indices to diphone strings
    diphones = [LOGIT_TO_DIPHONE[idx] for idx in diphone_indices]
    
    # Convert diphones to phonemes
    return diphone_sequence_to_phonemes(diphones)


def _extract_transcription(input):
    endIdx = np.argwhere(input == 0)[0, 0]
    trans = ''
    for c in range(endIdx):
        trans += chr(input[c])
    return trans


def load_h5py_file(file_path, b2txt_csv_df):
    """
    Load HDF5 file with diphone-encoded data
    """
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],  # Now contains diphone indices (0-1680)
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
        'corpus': [],
    }
    
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())

        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            # Match with CSV to get corpus name
            year, month, day = session.split('.')[1:]
            date = f'{year}-{month}-{day}'
            row = b2txt_csv_df[(b2txt_csv_df['Date'] == date) & (b2txt_csv_df['Block number'] == block_num)]
            corpus_name = row['Corpus'].values[0] if len(row) > 0 else 'unknown'

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
            data['corpus'].append(corpus_name)
    
    return data


def rearrange_speech_logits_diphone(logits):
    """
    Rearrange diphone logits for language model
    
    For diphones, we may need to convert back to phoneme space
    or adjust the order for the language model
    
    Args:
        logits: [batch, time, 1681] diphone logits
    
    Returns:
        Rearranged logits
    """
    # For now, keep the same structure
    # You may need to modify this based on your language model requirements
    return logits


def runSingleDecodingStep(x, input_layer, model, model_args, device):
    """
    Single decoding step for diphone model
    """
    # Use autocast for efficiency
    with torch.autocast(device_type="cuda", enabled=model_args['use_amp'], dtype=torch.bfloat16):

        x = gauss_smooth(
            inputs=x, 
            device=device,
            smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
            smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
            padding='valid',
        )

        with torch.no_grad():
            logits, _ = model(
                x=x,
                day_idx=torch.tensor([input_layer], device=device),
                states=None,
                return_state=True,
            )

    # Convert logits from bfloat16 to float32
    logits = logits.float().cpu().numpy()

    return logits


def remove_punctuation(sentence):
    """Remove punctuation from sentence"""
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])
    return sentence


def get_current_redis_time_ms(redis_conn):
    """Get current Redis time in milliseconds"""
    t = redis_conn.time()
    return int(t[0]*1000 + t[1]/1000)


######### Language Model Helper Functions ##########

def reset_remote_language_model(r, remote_lm_done_resetting_lastEntrySeen):
    r.xadd('remote_lm_reset', {'done': 0})
    time.sleep(0.001)
    
    remote_lm_done_resetting = []
    while len(remote_lm_done_resetting) == 0:
        remote_lm_done_resetting = r.xread(
            {'remote_lm_done_resetting': remote_lm_done_resetting_lastEntrySeen},
            count=1,
            block=10000,
        )
        if len(remote_lm_done_resetting) == 0:
            print(f'Still waiting for remote lm reset from ts {remote_lm_done_resetting_lastEntrySeen}...')
    
    for entry_id, entry_data in remote_lm_done_resetting[0][1]:
        remote_lm_done_resetting_lastEntrySeen = entry_id
    
    return remote_lm_done_resetting_lastEntrySeen


def update_remote_lm_params(r, remote_lm_done_updating_lastEntrySeen, 
                            acoustic_scale=0.35, blank_penalty=90.0, alpha=0.55):
    
    entry_dict = {
        'acoustic_scale': acoustic_scale,
        'blank_penalty': blank_penalty,
        'alpha': alpha,
    }

    r.xadd('remote_lm_update_params', entry_dict)
    time.sleep(0.001)
    
    remote_lm_done_updating = []
    while len(remote_lm_done_updating) == 0:
        remote_lm_done_updating = r.xread(
            {'remote_lm_done_updating_params': remote_lm_done_updating_lastEntrySeen},
            block=10000,
            count=1,
        )
        if len(remote_lm_done_updating) == 0:
            print(f'Still waiting for remote lm to update parameters...')
    
    for entry_id, entry_data in remote_lm_done_updating[0][1]:
        remote_lm_done_updating_lastEntrySeen = entry_id
    
    return remote_lm_done_updating_lastEntrySeen


def send_logits_to_remote_lm(r, remote_lm_input_stream, remote_lm_output_partial_stream,
                             remote_lm_output_partial_lastEntrySeen, logits):
    
    r.xadd(remote_lm_input_stream, {'logits': np.float32(logits).tobytes()})
    
    remote_lm_output = []
    while len(remote_lm_output) == 0:
        remote_lm_output = r.xread(
            {remote_lm_output_partial_stream: remote_lm_output_partial_lastEntrySeen},
            block=10000,
            count=1,
        )
        if len(remote_lm_output) == 0:
            print(f'Still waiting for remote lm partial output...')
    
    for entry_id, entry_data in remote_lm_output[0][1]:
        remote_lm_output_partial_lastEntrySeen = entry_id
        decoded = entry_data[b'lm_response_partial'].decode()

    return remote_lm_output_partial_lastEntrySeen, decoded


def finalize_remote_lm(r, remote_lm_output_final_stream, remote_lm_output_final_lastEntrySeen):
    
    r.xadd('remote_lm_finalize', {'done': 0})
    time.sleep(0.005)
    
    remote_lm_output = []
    while len(remote_lm_output) == 0:
        remote_lm_output = r.xread(
            {remote_lm_output_final_stream: remote_lm_output_final_lastEntrySeen},
            block=10000,
            count=1,
        )
        if len(remote_lm_output) == 0:
            print(f'Still waiting for remote lm final output...')

    for entry_id, entry_data in remote_lm_output[0][1]:
        remote_lm_output_final_lastEntrySeen = entry_id

        candidate_sentences = [str(c) for c in entry_data[b'scoring'].decode().split(';')[::5]]
        candidate_acoustic_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[1::5]]
        candidate_ngram_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[2::5]]
        candidate_llm_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[3::5]]
        candidate_total_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[4::5]]

    # Handle edge case with no candidates
    if len(candidate_sentences) == 0 or len(candidate_total_scores) == 0:
        print('No candidate sentences from language model.')
        candidate_sentences = ['']
        candidate_acoustic_scores = [0]
        candidate_ngram_scores = [0]
        candidate_llm_scores = [0]
        candidate_total_scores = [0]
    else:
        # Sort by total score
        sort_order = np.argsort(candidate_total_scores)[::-1]
        candidate_sentences = [candidate_sentences[i] for i in sort_order]
        candidate_acoustic_scores = [candidate_acoustic_scores[i] for i in sort_order]
        candidate_ngram_scores = [candidate_ngram_scores[i] for i in sort_order]
        candidate_llm_scores = [candidate_llm_scores[i] for i in sort_order]
        candidate_total_scores = [candidate_total_scores[i] for i in sort_order]

    # Remove duplicates
    for i in range(len(candidate_sentences)-1, 0, -1):
        if candidate_sentences[i] in candidate_sentences[:i]:
            candidate_sentences.pop(i)
            candidate_acoustic_scores.pop(i)
            candidate_ngram_scores.pop(i)
            candidate_llm_scores.pop(i)
            candidate_total_scores.pop(i)

    lm_out = {
        'candidate_sentences': candidate_sentences,
        'candidate_acoustic_scores': candidate_acoustic_scores,
        'candidate_ngram_scores': candidate_ngram_scores,
        'candidate_llm_scores': candidate_llm_scores,
        'candidate_total_scores': candidate_total_scores,
    }

    return remote_lm_output_final_lastEntrySeen, lm_out

