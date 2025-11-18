"""
Quick script to compare the two model architectures.
Shows parameter counts, layer structures, and key differences.
"""
import torch
from rnn_model import GRUDecoder
from neural_decoder_model import GRUDecoder_NeuralDecoder_WithInputNet


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model, name):
    """Print a detailed summary of the model."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Size in MB (float32): {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print(f"\nModel structure:")
    print(model)


def compare_architectures():
    """Compare the two architectures side by side."""
    
    # Common config
    n_days = 4
    n_classes = 1667
    n_layers = 5
    neural_dim = 512
    
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*80)
    
    # Original model
    print("\n[1] Original GRUDecoder")
    print("-" * 80)
    model1 = GRUDecoder(
        neural_dim=neural_dim,
        n_units=1024,
        n_days=n_days,
        n_classes=n_classes,
        rnn_dropout=0.4,
        input_dropout=0.3,
        n_layers=n_layers,
        bidirectional=False,
        patch_size=4,
        patch_stride=2,
    )
    
    total1, trainable1 = count_parameters(model1)
    print(f"GRU units: 1024")
    print(f"Day layers: {neural_dim} → {neural_dim}")
    print(f"Patching: kernel=4, stride=2")
    print(f"Dropout: RNN=40%, Input=30%")
    print(f"Total parameters: {total1:,}")
    print(f"Memory: {total1 * 4 / 1024 / 1024:.2f} MB")
    
    # NeuralDecoder model
    print("\n[2] NeuralDecoder (Converted from TensorFlow)")
    print("-" * 80)
    model2 = GRUDecoder_NeuralDecoder_WithInputNet(
        neural_dim=neural_dim,
        n_days=n_days,
        input_layer_sizes=[256],
        gru_units=512,
        n_classes=n_classes,
        n_gru_layers=n_layers,
        input_activation='softsign',
        input_dropout=0.2,
        rnn_dropout=0.4,
        bidirectional=False,
        stack_kwargs={'kernel_size': 14, 'strides': 4},
    )
    
    total2, trainable2 = count_parameters(model2)
    print(f"GRU units: 512")
    print(f"Day layers: {neural_dim} → 256")
    print(f"Patching: kernel=14, stride=4")
    print(f"Dropout: RNN=40%, Input=20%")
    print(f"Total parameters: {total2:,}")
    print(f"Memory: {total2 * 4 / 1024 / 1024:.2f} MB")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Parameter reduction: {(1 - total2/total1)*100:.1f}%")
    print(f"Memory reduction: {(1 - total2/total1)*100:.1f}%")
    print(f"Speed improvement (estimated): {(total1/total2):.2f}x faster")
    
    # Sequence length comparison
    print("\n" + "="*80)
    print("SEQUENCE LENGTH AFTER PATCHING (example: 1000 timesteps)")
    print("="*80)
    
    input_len = 1000
    
    # Model 1 patching
    patch1_len = (input_len - 4) // 2 + 1
    print(f"Original model: {input_len} → {patch1_len} timesteps")
    print(f"  Compression ratio: {input_len/patch1_len:.2f}x")
    
    # Model 2 patching
    patch2_len = (input_len - 14) // 4 + 1
    print(f"NeuralDecoder: {input_len} → {patch2_len} timesteps")
    print(f"  Compression ratio: {input_len/patch2_len:.2f}x")
    
    print(f"\nDifference: NeuralDecoder produces {patch2_len - patch1_len} fewer timesteps")
    print(f"This is {(1 - patch2_len/patch1_len)*100:.1f}% shorter sequence for GRU to process")
    
    # CTC constraint check
    print("\n" + "="*80)
    print("CTC CONSTRAINT CHECK")
    print("="*80)
    print("For CTC to work: input_length >= target_length")
    print(f"\nOriginal model output length: {patch1_len}")
    print(f"NeuralDecoder output length: {patch2_len}")
    print(f"\nMax target length (safe):")
    print(f"  Original: {patch1_len} diphones")
    print(f"  NeuralDecoder: {patch2_len} diphones")
    print(f"\n⚠️  NeuralDecoder can handle {patch2_len/patch1_len:.2f}x shorter targets")
    print(f"    This may cause issues with longer sentences!")
    
    # Forward pass test
    print("\n" + "="*80)
    print("FORWARD PASS TEST")
    print("="*80)
    
    batch_size = 2
    time_steps = 100
    x = torch.randn(batch_size, time_steps, neural_dim)
    day_idx = torch.tensor([0, 1])
    
    print(f"Input shape: {x.shape}")
    
    # Model 1
    with torch.no_grad():
        out1 = model1(x, day_idx)
    print(f"Original model output: {out1.shape}")
    
    # Model 2
    with torch.no_grad():
        out2 = model2(x, day_idx)
    print(f"NeuralDecoder output: {out2.shape}")
    
    print("\n✅ Both models forward pass successful!")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nOriginal Model:")
    print("  ✅ Larger capacity (1024 units)")
    print("  ✅ Gentler sequence compression")
    print("  ✅ Can handle longer target sequences")
    print("  ❌ 3x more parameters")
    print("  ❌ Slower training")
    
    print("\nNeuralDecoder Model:")
    print("  ✅ 3x fewer parameters")
    print("  ✅ Faster training and inference")
    print("  ✅ Proven architecture from Stanford paper")
    print("  ✅ Better regularization")
    print("  ❌ Smaller capacity (512 units)")
    print("  ❌ More aggressive compression (may violate CTC on long sequences)")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
Try NeuralDecoder if:
  - You want faster training
  - You're seeing overfitting
  - You want to match the paper baseline

Stick with Original if:
  - You need maximum model capacity
  - You have very long sequences
  - Current model is working well
""")


if __name__ == "__main__":
    compare_architectures()


