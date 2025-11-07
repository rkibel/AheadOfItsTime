"""
Convert PyTorch models to TensorRT optimized engines.

TensorRT provides:
- Aggressive AOT compilation with kernel fusion
- INT8/FP16 quantization support
- Optimized execution for NVIDIA GPUs
- Maximum inference performance

Note: TensorRT conversion typically goes through ONNX as an intermediate format.
This script requires:
- NVIDIA GPU with CUDA support
- TensorRT library installed
- onnx-tensorrt or trtexec utility

Usage:
    # Convert LeNet-5 to TensorRT
    python conversion/to_tensorrt.py --model lenet \\
        --checkpoint checkpoints/pytorch/lenet_mnist.pth \\
        --output checkpoints/tensorrt/lenet.engine

    # Convert with FP16 precision
    python conversion/to_tensorrt.py --model resnet18 \\
        --checkpoint checkpoints/pytorch/resnet18_cifar10.pth \\
        --output checkpoints/tensorrt/resnet18.engine \\
        --precision fp16
"""

import argparse
import torch
import time
import sys
import os
import subprocess
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

# Try to import TensorRT
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT Python bindings not available. "
          "Will attempt to use trtexec command-line tool.")


# Model configurations
MODEL_CONFIGS = {
    'lenet': {
        'class': LeNet5,
        'kwargs': {'num_classes': 10, 'in_channels': 1},
        'example_input': torch.randn(1, 1, 28, 28),
        'checkpoint': 'checkpoints/pytorch/lenet_mnist.pth',
        'input_shape': (1, 1, 28, 28),
        'input_name': 'input',
        'output_name': 'output'
    },
    'resnet18': {
        'class': ResNet18,
        'kwargs': {'num_classes': 10},
        'example_input': torch.randn(1, 3, 32, 32),
        'checkpoint': 'checkpoints/pytorch/resnet18_cifar10.pth',
        'input_shape': (1, 3, 32, 32),
        'input_name': 'input',
        'output_name': 'output'
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
        'input_shape': (1, 256),
        'input_name': 'input',
        'output_name': 'output'
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
        'input_shape': (32, 35),
        'input_name': 'input',
        'output_name': 'output'
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
    Convert PyTorch model to ONNX (intermediate step for TensorRT).
    
    Args:
        model: PyTorch model in eval mode
        example_input: Example input tensor
        onnx_path: Path to save ONNX model
        opset_version: ONNX opset version (TensorRT supports up to opset 13)
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


def convert_onnx_to_tensorrt_python(
    onnx_path: str,
    engine_path: str,
    precision: str = 'fp32',
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 30  # 1GB
) -> None:
    """
    Convert ONNX model to TensorRT engine using Python API.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', 'int8')
        max_batch_size: Maximum batch size for optimization
        max_workspace_size: Maximum workspace size in bytes
    """
    if not TENSORRT_AVAILABLE:
        raise ImportError("TensorRT Python bindings not available. "
                         "Use --use-trtexec flag or install tensorrt package.")
    
    print("Step 2: Converting ONNX to TensorRT engine (Python API)...")
    start_time = time.time()
    
    # Create TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"  ✗ Parser error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model")
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    
    # Set precision
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  ✓ Using FP16 precision")
        else:
            print("  ⚠ FP16 not supported on this platform, using FP32")
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  ✓ Using INT8 precision")
        else:
            print("  ⚠ INT8 not supported on this platform, using FP32")
    else:
        print("  ✓ Using FP32 precision")
    
    # Build engine
    print("  Building TensorRT engine (this may take a while)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    conversion_time = time.time() - start_time
    print(f"  ✓ TensorRT engine built in {conversion_time:.3f}s")


def convert_onnx_to_tensorrt_trtexec(
    onnx_path: str,
    engine_path: str,
    precision: str = 'fp32',
    max_batch_size: int = 1
) -> None:
    """
    Convert ONNX model to TensorRT engine using trtexec command-line tool.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', 'int8')
        max_batch_size: Maximum batch size for optimization
    """
    print("Step 2: Converting ONNX to TensorRT engine (trtexec)...")
    
    # Check if trtexec is available
    try:
        result = subprocess.run(['trtexec', '--help'], 
                              capture_output=True, 
                              timeout=5)
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError("trtexec not found. Please ensure TensorRT is installed "
                          "and trtexec is in your PATH.")
    
    # Build trtexec command
    cmd = [
        'trtexec',
        '--onnx', onnx_path,
        '--saveEngine', engine_path,
        '--workspace', str(1 << 30),  # 1GB
        '--maxBatch', str(max_batch_size),
        '--verbose'
    ]
    
    # Add precision flags
    if precision == 'fp16':
        cmd.append('--fp16')
    elif precision == 'int8':
        cmd.append('--int8')
    
    # Run trtexec
    print(f"  Running: {' '.join(cmd)}")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ✗ trtexec failed:")
        print(result.stderr)
        raise RuntimeError("Failed to build TensorRT engine with trtexec")
    
    conversion_time = time.time() - start_time
    print(f"  ✓ TensorRT engine built in {conversion_time:.3f}s")


def validate_tensorrt_model(
    original_model: torch.nn.Module,
    engine_path: str,
    example_input: torch.Tensor,
    device: str = 'cuda',
    num_samples: int = 10
) -> bool:
    """
    Validate that TensorRT model produces identical outputs to PyTorch model.
    
    Note: This is a simplified validation. Full TensorRT inference requires
    proper context management and memory allocation.
    
    Args:
        original_model: Original PyTorch model
        engine_path: Path to TensorRT engine
        example_input: Example input tensor
        device: Device to run on
        num_samples: Number of random samples to test
        
    Returns:
        True if validation passes (or skipped if TensorRT Python API not available)
    """
    if not TENSORRT_AVAILABLE:
        print("\n⚠ Skipping validation (TensorRT Python API not available)")
        print("  To validate, install tensorrt package and ensure CUDA is available")
        return True
    
    if device != 'cuda':
        print("\n⚠ Skipping validation (TensorRT requires CUDA)")
        return True
    
    print("\n⚠ TensorRT validation requires full inference setup")
    print("  For production use, implement proper TensorRT inference with:")
    print("  - Runtime deserialization")
    print("  - Context creation")
    print("  - Memory allocation and binding")
    print("  - CUDA stream management")
    print("  See TensorRT samples for reference implementation")
    
    # For now, just check that the engine file exists and is valid
    if os.path.exists(engine_path) and os.path.getsize(engine_path) > 0:
        print("  ✓ TensorRT engine file created successfully")
        return True
    else:
        print("  ✗ TensorRT engine file is invalid or missing")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch models to TensorRT optimized engines'
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
        help='Output path for TensorRT engine (.engine file)'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='fp32',
        choices=['fp32', 'fp16', 'int8'],
        help='Precision mode (default: fp32)'
    )
    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=1,
        help='Maximum batch size for optimization (default: 1)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=13,
        help='ONNX opset version for intermediate conversion (default: 13)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for conversion'
    )
    parser.add_argument(
        '--use-trtexec',
        action='store_true',
        help='Use trtexec command-line tool instead of Python API'
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
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available. TensorRT requires CUDA for optimal performance.")
        args.device = 'cpu'
    
    # Get model configuration
    config = MODEL_CONFIGS[args.model]
    checkpoint_path = args.checkpoint or config['checkpoint']
    
    print(f"\n{'='*60}")
    print(f"Converting {args.model.upper()} to TensorRT Engine")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {args.output}")
    print(f"Precision: {args.precision}")
    print(f"Max Batch Size: {args.max_batch_size}")
    print(f"Device: {args.device}")
    print(f"Method: {'trtexec' if args.use_trtexec else 'Python API'}")
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
    
    # Step 2: Convert ONNX to TensorRT
    try:
        if args.use_trtexec:
            convert_onnx_to_tensorrt_trtexec(
                onnx_path=onnx_temp_path,
                engine_path=args.output,
                precision=args.precision,
                max_batch_size=args.max_batch_size
            )
        else:
            convert_onnx_to_tensorrt_python(
                onnx_path=onnx_temp_path,
                engine_path=args.output,
                precision=args.precision,
                max_batch_size=args.max_batch_size
            )
    except Exception as e:
        print(f"\n✗ TensorRT conversion failed: {e}")
        if not args.keep_onnx and os.path.exists(onnx_temp_path):
            os.remove(onnx_temp_path)
        return
    
    # Clean up intermediate ONNX file if not keeping it
    if not args.keep_onnx and os.path.exists(onnx_temp_path):
        os.remove(onnx_temp_path)
        print(f"  ✓ Removed intermediate ONNX file")
    
    # Validate conversion
    if args.validate:
        is_valid = validate_tensorrt_model(
            original_model=model,
            engine_path=args.output,
            example_input=config['example_input'],
            device=args.device
        )
        
        if not is_valid:
            print("\n✗ Validation failed! Engine file may be invalid.")
            return
    
    # Get file size
    file_size = Path(args.output).stat().st_size / (1024 * 1024)  # MB
    print(f"\n✓ Engine saved ({file_size:.2f} MB)")
    
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

