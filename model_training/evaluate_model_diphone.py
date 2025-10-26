"""
Evaluation script for DIPHONE model (1681 classes)

Usage:
    python evaluate_model_diphone.py \
        --model_path trained_models/diphone_rnn \
        --data_dir ../data/hdf5_data_diphone_encoded \
        --eval_type val \
        --gpu_number 0
"""
import os
import torch
import numpy as np
import pandas as pd
import redis
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

from rnn_model import GRUDecoder
from evaluate_model_helpers_diphone import *

# Argument parser
parser = argparse.ArgumentParser(description='Evaluate a pretrained DIPHONE RNN model.')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to the pretrained model directory')
parser.add_argument('--data_dir', type=str, default='../data/hdf5_data_diphone_encoded',
                    help='Path to the diphone-encoded dataset directory')
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                    help='Evaluation type: "val" or "test"')
parser.add_argument('--csv_path', type=str, default='../data/t15_copyTaskData_description.csv',
                    help='Path to the CSV metadata file')
parser.add_argument('--gpu_number', type=int, default=0,
                    help='GPU number to use. Set to -1 for CPU.')
parser.add_argument('--convert_to_phonemes', action='store_true',
                    help='Convert diphone predictions to phonemes before language model')
args = parser.parse_args()

print("="*80)
print("DIPHONE MODEL EVALUATION")
print("="*80)
print(f"Model path: {args.model_path}")
print(f"Data directory: {args.data_dir}")
print(f"Evaluation type: {args.eval_type}")
print(f"Number of classes: 1681 (diphones)")
print("="*80)

# Load CSV metadata
b2txt_csv_df = pd.read_csv(args.csv_path)

# Load model args
model_args = OmegaConf.load(os.path.join(args.model_path, 'checkpoint/args.yaml'))

# Verify this is a diphone model
if model_args['dataset']['n_classes'] != 1681:
    print(f"WARNING: Model has {model_args['dataset']['n_classes']} classes, expected 1681 for diphones")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        exit(1)

# Set up GPU device
gpu_number = args.gpu_number
if torch.cuda.is_available() and gpu_number >= 0:
    if gpu_number >= torch.cuda.device_count():
        raise ValueError(f'GPU {gpu_number} out of range. Available: {torch.cuda.device_count()}')
    device = torch.device(f'cuda:{gpu_number}')
    print(f'Using {device} for model inference.')
else:
    if gpu_number >= 0:
        print(f'GPU {gpu_number} requested but not available.')
    print('Using CPU for model inference.')
    device = torch.device('cpu')

# Define model
print("\nInitializing model...")
model = GRUDecoder(
    neural_dim=model_args['model']['n_input_features'],
    n_units=model_args['model']['n_units'], 
    n_days=len(model_args['dataset']['sessions']),
    n_classes=model_args['dataset']['n_classes'],  # 1681 for diphones
    rnn_dropout=model_args['model']['rnn_dropout'],
    input_dropout=model_args['model']['input_network']['input_layer_dropout'],
    n_layers=model_args['model']['n_layers'],
    patch_size=model_args['model']['patch_size'],
    patch_stride=model_args['model']['patch_stride'],
)

# Load model weights
print("Loading checkpoint...")
checkpoint = torch.load(
    os.path.join(args.model_path, 'checkpoint/best_checkpoint'), 
    weights_only=False
)

# Clean up state dict keys
for key in list(checkpoint['model_state_dict'].keys()):
    checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
    checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Model loaded successfully!")
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

# Load test data
print(f"\nLoading {args.eval_type} data...")
test_data = {}
total_test_trials = 0

for session in model_args['dataset']['sessions']:
    files = [f for f in os.listdir(os.path.join(args.data_dir, session)) if f.endswith('.hdf5')]
    
    if f'data_{args.eval_type}.hdf5' in files:
        eval_file = os.path.join(args.data_dir, session, f'data_{args.eval_type}.hdf5')
        data = load_h5py_file(eval_file, b2txt_csv_df)
        test_data[session] = data
        
        total_test_trials += len(test_data[session]["neural_features"])
        print(f'  Loaded {len(test_data[session]["neural_features"])} trials for {session}')

print(f'\nTotal {args.eval_type} trials: {total_test_trials}')
print()

# Run inference
print("Running RNN inference...")
with tqdm(total=total_test_trials, desc='Predicting diphone sequences', unit='trial') as pbar:
    for session, data in test_data.items():
        data['logits'] = []
        data['pred_seq'] = []
        data['pred_seq_phonemes'] = []  # Store phoneme version
        
        input_layer = model_args['dataset']['sessions'].index(session)
        
        for trial in range(len(data['neural_features'])):
            # Get neural input
            neural_input = data['neural_features'][trial]
            neural_input = np.expand_dims(neural_input, axis=0)
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)
            
            # Run decoding
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data['logits'].append(logits)
            
            pbar.update(1)
pbar.close()

# Convert logits to diphone sequences
print("\nDecoding diphone sequences...")
for session, data in test_data.items():
    for trial in range(len(data['logits'])):
        logits = data['logits'][trial][0]
        
        # Decode to diphone indices
        pred_seq = np.argmax(logits, axis=-1)
        # Remove blanks (0)
        pred_seq = [int(p) for p in pred_seq if p != 0]
        # Remove consecutive duplicates
        pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
        
        # Convert to diphone strings
        pred_seq_diphones = [LOGIT_TO_DIPHONE[p] for p in pred_seq]
        data['pred_seq'].append(pred_seq_diphones)
        
        # Convert diphones to phonemes
        pred_seq_phonemes = diphone_sequence_to_phonemes(pred_seq_diphones)
        data['pred_seq_phonemes'].append(pred_seq_phonemes)
        
        # Print predictions
        block_num = data['block_num'][trial]
        trial_num = data['trial_num'][trial]
        
        print(f'\nSession: {session}, Block: {block_num}, Trial: {trial_num}')
        
        if args.eval_type == 'val':
            sentence_label = data['sentence_label'][trial]
            true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
            true_seq_diphones = [LOGIT_TO_DIPHONE[p] for p in true_seq]
            true_seq_phonemes = diphone_sequence_to_phonemes(true_seq_diphones)
            
            print(f'Sentence label:       {sentence_label}')
            print(f'True diphones:        {" ".join(true_seq_diphones[:10])}...')  # Show first 10
            print(f'True phonemes:        {" ".join(true_seq_phonemes)}')
        
        print(f'Predicted diphones:   {" ".join(pred_seq_diphones[:10])}...')  # Show first 10
        print(f'Predicted phonemes:   {" ".join(pred_seq_phonemes)}')

print("\n" + "="*80)
print("NOTE: Language model integration for diphones requires additional setup.")
print("The diphone→phoneme conversion is done, but you may need to:")
print("  1. Retrain the language model on phoneme sequences from diphone predictions")
print("  2. Or convert diphone logits to phoneme logits for the existing LM")
print("="*80)

# For now, skip language model inference and just save the predictions
print("\nSkipping language model for diphone evaluation.")
print("Raw phoneme sequences have been extracted from diphones.")

# Calculate Diphone Error Rate if validation set
if args.eval_type == 'val':
    print("\nCalculating Diphone Error Rate (DER)...")
    total_edit_distance = 0
    total_seq_length = 0
    
    for session, data in test_data.items():
        for trial in range(len(data['pred_seq'])):
            true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
            pred_seq = [DIPHONE_TO_LOGIT.get(d, 0) for d in data['pred_seq'][trial]]
            
            ed = editdistance.eval(pred_seq, true_seq.tolist())
            total_edit_distance += ed
            total_seq_length += len(true_seq)
    
    DER = total_edit_distance / total_seq_length
    print(f"Diphone Error Rate (DER): {DER*100:.2f}%")
    
    # Also calculate Phoneme Error Rate
    print("\nCalculating Phoneme Error Rate (PER) from diphone predictions...")
    total_phoneme_ed = 0
    total_phoneme_len = 0
    
    for session, data in test_data.items():
        for trial in range(len(data['pred_seq_phonemes'])):
            # Get true phonemes from diphones
            true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
            true_diphones = [LOGIT_TO_DIPHONE[p] for p in true_seq]
            true_phonemes = diphone_sequence_to_phonemes(true_diphones)
            
            pred_phonemes = data['pred_seq_phonemes'][trial]
            
            ed = editdistance.eval(pred_phonemes, true_phonemes)
            total_phoneme_ed += ed
            total_phoneme_len += len(true_phonemes)
    
    PER = total_phoneme_ed / total_phoneme_len
    print(f"Phoneme Error Rate (PER): {PER*100:.2f}%")

# Save predictions
output_file = os.path.join(
    args.model_path, 
    f'diphone_{args.eval_type}_predictions_{time.strftime("%Y%m%d_%H%M%S")}.csv'
)

# Save both diphone and phoneme sequences
df_out = pd.DataFrame({
    'session': [s for s, d in test_data.items() for _ in range(len(d['pred_seq']))],
    'block': [d['block_num'][i] for d in test_data.values() for i in range(len(d['pred_seq']))],
    'trial': [d['trial_num'][i] for d in test_data.values() for i in range(len(d['pred_seq']))],
    'pred_diphones': [' '.join(d['pred_seq'][i]) for d in test_data.values() for i in range(len(d['pred_seq']))],
    'pred_phonemes': [' '.join(d['pred_seq_phonemes'][i]) for d in test_data.values() for i in range(len(d['pred_seq']))],
})

df_out.to_csv(output_file, index=False)
print(f"\nPredictions saved to: {output_file}")
print("="*80)
print("Evaluation complete!")
print("="*80)

