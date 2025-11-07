"""
Convert PyTorch models to TorchScript format.

TorchScript provides JIT compilation for PyTorch models, enabling:
- Faster inference through graph optimization
- Deployment without Python dependency
- Compatibility with TorchServe and mobile platforms

Usage:
    # Convert LeNet-5
    python conversion/to_torchscript.py --model lenet \\
        --checkpoint checkpoints/pytorch/lenet_mnist.pth \\
        --output checkpoints/torchscript/lenet_mnist.pt

    # Convert with scripting mode (instead of tracing)
    python conversion/to_torchscript.py --model lenet \\
        --checkpoint checkpoints/pytorch/lenet_mnist.pth \\
        --mode script \\
        --output checkpoints/torchscript/lenet_mnist_scripted.pt
"""

import argparse
import torch
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import model classes
from models.cnn.lenet import LeNet5
from models.cnn.resnet import ResNet18
from models.rnn.lstm_sentiment import LSTMSentimentClassifier
from models.rnn.gru_lm import GRULanguageModel

# Import conversion utilities
from conversion.utils import (
    load_model_from_checkpoint,
    validate_outputs,
    create_output_directory
)


# Model configurations
MODEL_CONFIGS = {
    'lenet': {
        'class': LeNet5,
        'kwargs': {'num_classes': 10, 'in_channels': 1},
        'example_input': torch.randn(1, 1, 28, 28),
        'checkpoint': 'checkpoints/pytorch/lenet_mnist.pth'
    },
    'resnet18': {
        'class': ResNet18,
        'kwargs': {'num_classes': 10},
        'example_input': torch.randn(1, 3, 32, 32),
        'checkpoint': 'checkpoints/pytorch/resnet18_cifar10.pth'
    },
    'lstm': {
        'class': LSTMSentimentClassifier,
        'kwargs': {
            'vocab_size': 25000,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.5
        },
        'example_input': torch.randint(0, 25000, (1, 256)),
        'checkpoint': 'checkpoints/pytorch/lstm_imdb.pth'
    },
    'gru': {
        'class': GRULanguageModel,
        'kwargs': {
            'vocab_size': 29573,
            'embedding_dim': 200,
            'hidden_dim': 200,
            'num_layers': 2,
            'dropout': 0.2
        },
        'example_input': torch.randint(0, 29573, (32, 35)),
        'checkpoint': 'checkpoints/pytorch/gru_wikitext.pth'
    }
}


def convert_to_torchscript(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    mode: str = 'trace',
    device: str = 'cpu'
) -> torch.jit.ScriptModule:
    """
    Convert PyTorch model to TorchScript.
    
    Args:
        model: PyTorch model in eval mode
        example_input: Example input tensor for tracing
        mode: 'trace' or 'script'
        device: Device to run conversion on
        
    Returns:
        TorchScript model
    """
    model = model.to(device)
    example_input = example_input.to(device)
    
    print(f"Converting using {mode} mode...")
    start_time = time.time()
    
    if mode == 'trace':
        # Tracing: Records operations during example forward pass
        scripted_model = torch.jit.trace(model, example_input)
    elif mode == 'script':
        # Scripting: Compiles Python code directly
        scripted_model = torch.jit.script(model)
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'trace' or 'script'")
    
    conversion_time = time.time() - start_time
    print(f"Conversion completed in {conversion_time:.3f}s")
    
    return scripted_model


def validate_conversion(
    original_model: torch.nn.Module,
    scripted_model: torch.jit.ScriptModule,
    example_input: torch.Tensor,
    device: str = 'cpu',
    num_samples: int = 10
) -> bool:
    """
    Validate that TorchScript model produces identical outputs.
    
    Args:
        original_model: Original PyTorch model
        scripted_model: Converted TorchScript model
        example_input: Example input tensor
        device: Device to run on
        num_samples: Number of random samples to test
        
    Returns:
        True if validation passes
    """
    print("\nValidating conversion...")
    original_model.eval()
    scripted_model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate random input with same shape
            if example_input.dim() == 2:  # For RNNs (batch, seq_len)
                test_input = torch.randint_like(example_input, 0, 1000)
            else:  # For CNNs
                test_input = torch.randn_like(example_input)
            
            test_input = test_input.to(device)
            
            # Get outputs
            original_output = original_model(test_input)
            scripted_output = scripted_model(test_input)
            
            # Handle tuple outputs (RNNs return (output, hidden))
            if isinstance(original_output, tuple):
                original_output = original_output[0]
            if isinstance(scripted_output, tuple):
                scripted_output = scripted_output[0]
            
            # Validate
            is_valid, max_diff = validate_outputs(
                original_output,
                scripted_output,
                rtol=1e-3,
                atol=1e-4
            )
            
            if not is_valid:
                print(f"  ✗ Sample {i+1}: FAILED (max diff: {max_diff:.2e})")
                return False
            
            print(f"  ✓ Sample {i+1}: PASSED (max diff: {max_diff:.2e})")
    
    print("\n✓ All validation tests passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch models to TorchScript'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['lenet', 'resnet18', 'lstm', 'gru'],
        help='Model to convert'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (default: use MODEL_CONFIGS)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for TorchScript model'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='trace',
        choices=['trace', 'script'],
        help='Conversion mode: trace or script (default: trace)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for conversion'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Validate converted model (default: True)'
    )
    
    args = parser.parse_args()
    
    # Get model configuration
    config = MODEL_CONFIGS[args.model]
    checkpoint_path = args.checkpoint or config['checkpoint']
    
    print(f"\n{'='*60}")
    print(f"Converting {args.model.upper()} to TorchScript")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {args.output}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print()
    
    # Create output directory
    create_output_directory(args.output)
    
    # Load original model
    print("Loading PyTorch model...")
    model, checkpoint = load_model_from_checkpoint(
        model_class=config['class'],
        checkpoint_path=checkpoint_path,
        device=args.device,
        **config['kwargs']
    )
    print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Convert to TorchScript
    scripted_model = convert_to_torchscript(
        model=model,
        example_input=config['example_input'],
        mode=args.mode,
        device=args.device
    )
    
    # Validate conversion
    if args.validate:
        is_valid = validate_conversion(
            original_model=model,
            scripted_model=scripted_model,
            example_input=config['example_input'],
            device=args.device
        )
        
        if not is_valid:
            print("\n✗ Validation failed! Not saving model.")
            return
    
    # Save TorchScript model
    print(f"\nSaving TorchScript model to {args.output}...")
    scripted_model.save(args.output)
    
    # Get file size
    file_size = Path(args.output).stat().st_size / (1024 * 1024)  # MB
    print(f"✓ Model saved ({file_size:.2f} MB)")
    
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
