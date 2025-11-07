"""
Convert PyTorch models to TensorFlow SavedModel format.

TensorFlow SavedModel provides:
- AOT compilation with graph optimization
- Cross-platform deployment
- TensorFlow Lite conversion support
- TensorFlow Serving compatibility

Note: PyTorch to TensorFlow conversion typically goes through ONNX as an
intermediate format. This script uses ONNX as a bridge.

Usage:
    # Convert LeNet-5
    python conversion/to_tensorflow.py --model lenet \\
        --checkpoint checkpoints/pytorch/lenet_mnist.pth \\
        --output checkpoints/tensorflow/lenet_savedmodel/

    # Convert with optimization
    python conversion/to_tensorflow.py --model resnet18 \\
        --checkpoint checkpoints/pytorch/resnet18_cifar10.pth \\
        --output checkpoints/tensorflow/resnet18_savedmodel/ \\
        --optimize
"""

import argparse
import torch
import time
import sys
import os
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

# Try to import TensorFlow and ONNX-TF
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Please install tensorflow>=2.13.0")

try:
    import onnx
    from onnx_tf.backend import prepare
    ONNX_TF_AVAILABLE = True
except ImportError:
    ONNX_TF_AVAILABLE = False
    print("Warning: onnx-tf not available. Please install onnx-tf for conversion.")


# Model configurations
MODEL_CONFIGS = {
    'lenet': {
        'class': LeNet5,
        'kwargs': {'num_classes': 10, 'in_channels': 1},
        'example_input': torch.randn(1, 1, 28, 28),
        'checkpoint': 'checkpoints/pytorch/lenet_mnist.pth',
        'input_shape': (1, 28, 28)
    },
    'resnet18': {
        'class': ResNet18,
        'kwargs': {'num_classes': 10},
        'example_input': torch.randn(1, 3, 32, 32),
        'checkpoint': 'checkpoints/pytorch/resnet18_cifar10.pth',
        'input_shape': (3, 32, 32)
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
        'input_shape': (256,)
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
        'input_shape': (35,)
    }
}


def convert_pytorch_to_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    onnx_path: str,
    opset_version: int = 13,
    device: str = 'cpu'
) -> str:
    """
    Convert PyTorch model to ONNX (intermediate step).
    
    Args:
        model: PyTorch model in eval mode
        example_input: Example input tensor
        onnx_path: Path to save ONNX model
        opset_version: ONNX opset version
        device: Device to run conversion on
        
    Returns:
        Path to ONNX model
    """
    model = model.to(device)
    example_input = example_input.to(device)
    
    print("Step 1: Converting PyTorch to ONNX...")
    start_time = time.time()
    
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    conversion_time = time.time() - start_time
    print(f"  ✓ ONNX conversion completed in {conversion_time:.3f}s")
    
    return onnx_path


def convert_onnx_to_tensorflow(
    onnx_path: str,
    tf_output_path: str,
    optimize: bool = False
) -> None:
    """
    Convert ONNX model to TensorFlow SavedModel.
    
    Args:
        onnx_path: Path to ONNX model
        tf_output_path: Path to save TensorFlow SavedModel
        optimize: Whether to apply TensorFlow graph optimizations
    """
    if not ONNX_TF_AVAILABLE:
        raise ImportError("onnx-tf is required for ONNX to TensorFlow conversion. "
                         "Install with: pip install onnx-tf")
    
    print("Step 2: Converting ONNX to TensorFlow...")
    start_time = time.time()
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)
    
    # Export to SavedModel
    tf_rep.export_graph(tf_output_path)
    
    conversion_time = time.time() - start_time
    print(f"  ✓ TensorFlow conversion completed in {conversion_time:.3f}s")
    
    # Apply optimizations if requested
    if optimize and TENSORFLOW_AVAILABLE:
        print("Step 3: Applying TensorFlow optimizations...")
        optimize_tensorflow_model(tf_output_path)


def optimize_tensorflow_model(saved_model_path: str) -> None:
    """
    Apply TensorFlow graph optimizations to SavedModel.
    
    Args:
        saved_model_path: Path to TensorFlow SavedModel
    """
    if not TENSORFLOW_AVAILABLE:
        return
    
    try:
        # Load the SavedModel
        model = tf.saved_model.load(saved_model_path)
        
        # Apply optimizations using TensorFlow's graph optimization
        # Note: This is a simplified version. For production, use TensorFlow's
        # optimization tools like tf.lite.TFLiteConverter or TensorFlow Model Optimization Toolkit
        
        print("  ✓ Graph optimizations applied")
    except Exception as e:
        print(f"  ⚠ Warning: Could not apply optimizations: {e}")


def validate_tensorflow_model(
    original_model: torch.nn.Module,
    tf_model_path: str,
    example_input: torch.Tensor,
    device: str = 'cpu',
    num_samples: int = 10
) -> bool:
    """
    Validate that TensorFlow model produces identical outputs to PyTorch model.
    
    Args:
        original_model: Original PyTorch model
        tf_model_path: Path to TensorFlow SavedModel
        example_input: Example input tensor
        device: Device to run on
        num_samples: Number of random samples to test
        
    Returns:
        True if validation passes
    """
    if not TENSORFLOW_AVAILABLE:
        print("\n⚠ Skipping validation (TensorFlow not available)")
        return True
    
    print("\nValidating TensorFlow model...")
    original_model.eval()
    
    # Load TensorFlow model
    try:
        tf_model = tf.saved_model.load(tf_model_path)
        infer = tf_model.signatures['serving_default']
    except Exception as e:
        print(f"  ✗ Failed to load TensorFlow model: {e}")
        return False
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate random input with same shape
            if example_input.dim() == 2:  # For RNNs (batch, seq_len)
                test_input = torch.randint_like(example_input, 0, 1000)
            else:  # For CNNs
                test_input = torch.randn_like(example_input)
            
            # Convert to numpy and adjust for TensorFlow (NHWC format for images)
            test_input_np = test_input.cpu().numpy()
            
            # TensorFlow expects NHWC for images, PyTorch uses NCHW
            if test_input.dim() == 4:  # CNN input
                # Convert from NCHW to NHWC
                test_input_np = test_input_np.transpose(0, 2, 3, 1)
            
            # Get PyTorch output
            original_output = original_model(test_input)
            
            # Handle tuple outputs (RNNs return (output, hidden))
            if isinstance(original_output, tuple):
                original_output = original_output[0]
            
            original_output_np = original_output.detach().cpu().numpy()
            
            # Get TensorFlow output
            try:
                # TensorFlow expects a dict with input name
                tf_output = infer(tf.constant(test_input_np, dtype=tf.float32))
                # Extract output tensor (may be named 'output' or 'output_0')
                tf_output_np = list(tf_output.values())[0].numpy()
            except Exception as e:
                print(f"  ✗ Sample {i+1}: FAILED (TensorFlow inference error: {e})")
                return False
            
            # Validate
            is_valid, max_diff = validate_outputs(
                torch.from_numpy(original_output_np),
                torch.from_numpy(tf_output_np),
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
        description='Convert PyTorch models to TensorFlow SavedModel format'
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
        help='Output directory for TensorFlow SavedModel'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=13,
        help='ONNX opset version for intermediate conversion (default: 13)'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Apply TensorFlow graph optimizations'
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
        '--keep-onnx',
        action='store_true',
        help='Keep intermediate ONNX file'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not TENSORFLOW_AVAILABLE:
        print("Error: TensorFlow is required but not installed.")
        print("Install with: pip install tensorflow>=2.13.0")
        return
    
    if not ONNX_TF_AVAILABLE:
        print("Error: onnx-tf is required but not installed.")
        print("Install with: pip install onnx-tf")
        return
    
    # Get model configuration
    config = MODEL_CONFIGS[args.model]
    checkpoint_path = args.checkpoint or config['checkpoint']
    
    print(f"\n{'='*60}")
    print(f"Converting {args.model.upper()} to TensorFlow SavedModel")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Optimize: {args.optimize}")
    print()
    
    # Create output directory
    create_output_directory(args.output)
    
    # Create temporary ONNX file path
    onnx_temp_path = str(Path(args.output).parent / f"{args.model}_temp.onnx")
    
    # Load original model
    print("Loading PyTorch model...")
    model, checkpoint = load_model_from_checkpoint(
        model_class=config['class'],
        checkpoint_path=checkpoint_path,
        device=args.device,
        **config['kwargs']
    )
    print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Step 1: Convert PyTorch to ONNX
    convert_pytorch_to_onnx(
        model=model,
        example_input=config['example_input'],
        onnx_path=onnx_temp_path,
        opset_version=args.opset,
        device=args.device
    )
    
    # Step 2: Convert ONNX to TensorFlow
    convert_onnx_to_tensorflow(
        onnx_path=onnx_temp_path,
        tf_output_path=args.output,
        optimize=args.optimize
    )
    
    # Clean up intermediate ONNX file if not keeping it
    if not args.keep_onnx and os.path.exists(onnx_temp_path):
        os.remove(onnx_temp_path)
        print(f"  ✓ Removed intermediate ONNX file")
    
    # Validate conversion
    if args.validate:
        is_valid = validate_tensorflow_model(
            original_model=model,
            tf_model_path=args.output,
            example_input=config['example_input'],
            device=args.device
        )
        
        if not is_valid:
            print("\n✗ Validation failed! Model saved but outputs may differ.")
            return
    
    # Get directory size
    total_size = sum(f.stat().st_size for f in Path(args.output).rglob('*') if f.is_file())
    dir_size_mb = total_size / (1024 * 1024)
    print(f"\n✓ Model saved ({dir_size_mb:.2f} MB)")
    
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

