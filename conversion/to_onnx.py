"""
Convert PyTorch models to ONNX format.

ONNX (Open Neural Network Exchange) provides:
- Cross-platform model interoperability
- AOT compilation with graph optimization
- Support for multiple execution providers (CPU, CUDA, TensorRT)
- Efficient deployment on edge devices

Usage:
    # Convert LeNet-5
    python conversion/to_onnx.py --model lenet \\
        --checkpoint checkpoints/pytorch/lenet_mnist.pth \\
        --output checkpoints/onnx/lenet_mnist.onnx

    # Convert with dynamic batch size
    python conversion/to_onnx.py --model resnet18 \\
        --checkpoint checkpoints/pytorch/resnet18_cifar10.pth \\
        --output checkpoints/onnx/resnet18.onnx \\
        --dynamic-batch
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

# Try to import ONNX Runtime for validation
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("Warning: onnxruntime not available. Validation will be skipped.")


# Model configurations
MODEL_CONFIGS = {
    'lenet': {
        'class': LeNet5,
        'kwargs': {'num_classes': 10, 'in_channels': 1},
        'example_input': torch.randn(1, 1, 28, 28),
        'checkpoint': 'checkpoints/pytorch/lenet_mnist.pth',
        'input_names': ['input'],
        'output_names': ['output']
    },
    'resnet18': {
        'class': ResNet18,
        'kwargs': {'num_classes': 10},
        'example_input': torch.randn(1, 3, 32, 32),
        'checkpoint': 'checkpoints/pytorch/resnet18_cifar10.pth',
        'input_names': ['input'],
        'output_names': ['output']
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
        'checkpoint': 'checkpoints/pytorch/lstm_imdb.pth',
        'input_names': ['input'],
        'output_names': ['output']
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
        'checkpoint': 'checkpoints/pytorch/gru_wikitext.pth',
        'input_names': ['input'],
        'output_names': ['output']
    }
}


def convert_to_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_path: str,
    input_names: list = None,
    output_names: list = None,
    dynamic_axes: dict = None,
    opset_version: int = 13,
    device: str = 'cpu',
    verbose: bool = False
) -> None:
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model in eval mode
        example_input: Example input tensor for tracing
        output_path: Path to save ONNX model
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dictionary specifying dynamic axes (for variable batch/sequence lengths)
        opset_version: ONNX opset version to use
        device: Device to run conversion on
        verbose: Whether to print verbose output
    """
    model = model.to(device)
    example_input = example_input.to(device)
    
    print(f"Converting to ONNX (opset {opset_version})...")
    start_time = time.time()
    
    # Default input/output names
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # Export to ONNX
    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,  # Optimize constants
        export_params=True,  # Export trained parameters
        verbose=verbose,
        keep_initializers_as_inputs=False
    )
    
    conversion_time = time.time() - start_time
    print(f"Conversion completed in {conversion_time:.3f}s")


def validate_onnx_model(
    original_model: torch.nn.Module,
    onnx_path: str,
    example_input: torch.Tensor,
    device: str = 'cpu',
    num_samples: int = 10,
    execution_provider: str = None
) -> bool:
    """
    Validate that ONNX model produces identical outputs to PyTorch model.
    
    Args:
        original_model: Original PyTorch model
        onnx_path: Path to ONNX model
        example_input: Example input tensor
        device: Device to run on
        num_samples: Number of random samples to test
        execution_provider: ONNX Runtime execution provider ('CPUExecutionProvider', 'CUDAExecutionProvider', etc.)
        
    Returns:
        True if validation passes
    """
    if not ONNXRUNTIME_AVAILABLE:
        print("\n⚠ Skipping validation (onnxruntime not available)")
        return True
    
    print("\nValidating ONNX model...")
    original_model.eval()
    
    # Setup ONNX Runtime session
    providers = []
    if execution_provider:
        providers = [execution_provider]
    else:
        # Auto-detect: prefer CUDA if available, else CPU
        if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"Using execution providers: {session.get_providers()}")
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate random input with same shape
            if example_input.dim() == 2:  # For RNNs (batch, seq_len)
                test_input = torch.randint_like(example_input, 0, 1000)
            else:  # For CNNs
                test_input = torch.randn_like(example_input)
            
            test_input_np = test_input.cpu().numpy()
            
            # Get PyTorch output
            original_output = original_model(test_input)
            
            # Handle tuple outputs (RNNs return (output, hidden))
            if isinstance(original_output, tuple):
                original_output = original_output[0]
            
            original_output_np = original_output.detach().cpu().numpy()
            
            # Get ONNX Runtime output
            onnx_output = session.run([output_name], {input_name: test_input_np})[0]
            
            # Validate
            is_valid, max_diff = validate_outputs(
                torch.from_numpy(original_output_np),
                torch.from_numpy(onnx_output),
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
        description='Convert PyTorch models to ONNX format'
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
        help='Output path for ONNX model'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=13,
        help='ONNX opset version (default: 13)'
    )
    parser.add_argument(
        '--dynamic-batch',
        action='store_true',
        help='Enable dynamic batch size'
    )
    parser.add_argument(
        '--dynamic-shapes',
        action='store_true',
        help='Enable dynamic input shapes (for RNNs with variable sequence lengths)'
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
    parser.add_argument(
        '--execution-provider',
        type=str,
        choices=['CPUExecutionProvider', 'CUDAExecutionProvider'],
        help='ONNX Runtime execution provider for validation'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose ONNX export information'
    )
    
    args = parser.parse_args()
    
    # Get model configuration
    config = MODEL_CONFIGS[args.model]
    checkpoint_path = args.checkpoint or config['checkpoint']
    
    print(f"\n{'='*60}")
    print(f"Converting {args.model.upper()} to ONNX")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {args.output}")
    print(f"Opset Version: {args.opset}")
    print(f"Device: {args.device}")
    print(f"Dynamic Batch: {args.dynamic_batch}")
    print(f"Dynamic Shapes: {args.dynamic_shapes}")
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
    
    # Setup dynamic axes if requested
    dynamic_axes = None
    if args.dynamic_batch or args.dynamic_shapes:
        dynamic_axes = {
            config['input_names'][0]: {0: 'batch_size'}
        }
        if args.dynamic_shapes and args.model in ['lstm', 'gru']:
            # For RNNs, also make sequence length dynamic
            dynamic_axes[config['input_names'][0]][1] = 'sequence_length'
    
    # Convert to ONNX
    convert_to_onnx(
        model=model,
        example_input=config['example_input'],
        output_path=args.output,
        input_names=config['input_names'],
        output_names=config['output_names'],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        device=args.device,
        verbose=args.verbose
    )
    
    # Validate conversion
    if args.validate:
        is_valid = validate_onnx_model(
            original_model=model,
            onnx_path=args.output,
            example_input=config['example_input'],
            device=args.device,
            execution_provider=args.execution_provider
        )
        
        if not is_valid:
            print("\n✗ Validation failed! Model saved but outputs may differ.")
            return
    
    # Get file size
    file_size = Path(args.output).stat().st_size / (1024 * 1024)  # MB
    print(f"\n✓ Model saved ({file_size:.2f} MB)")
    
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

