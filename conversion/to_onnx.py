"""
Convert PyTorch models to ONNX format.

ONNX (Open Neural Network Exchange) provides:
- Cross-platform model interoperability
- AOT compilation with graph optimization
- Support for multiple execution providers (CPU, CUDA, TensorRT)
- Efficient deployment on edge devices

Usage:
    python conversion/to_onnx.py --model lenet
    python conversion/to_onnx.py --model all
    python conversion/to_onnx.py --model resnet18 --opset 14
"""

import argparse
import torch
import time
import sys
import onnx
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import centralized model configuration
from models import get_model_config, get_model_class, is_rnn_model

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


def validate_onnx_model(
    original_model: torch.nn.Module,
    onnx_path: str,
    example_input: torch.Tensor,
    device: str = 'cpu',
    num_samples: int = 10,
    rtol: float = 1e-3,
    atol: float = 1e-4
) -> bool:
    """
    Validate that ONNX model produces identical outputs to PyTorch model.
    """
    if not ONNXRUNTIME_AVAILABLE:
        print("  ⚠ Skipping validation (onnxruntime not available)")
        return True
    
    print("  Validating ONNX model...")
    
    # Verify ONNX model structure
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX structure check passed")
    except Exception as e:
        print(f"  ✗ ONNX structure check failed: {e}")
        return False

    original_model.eval()
    
    # Setup ONNX Runtime session
    providers = ['CPUExecutionProvider']
    if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        print(f"  ✗ Failed to create ONNX Runtime session: {e}")
        return False
        
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate random input
            if example_input.dtype in [torch.int64, torch.int32]:
                # For RNNs/Embeddings
                # Vary sequence length for RNNs to test dynamic shapes
                if is_rnn_model(original_model.__class__.__name__.lower()) or 'lstm' in str(type(original_model)).lower() or 'gru' in str(type(original_model)).lower():
                    seq_len = torch.randint(10, 100, (1,)).item()
                    # Keep batch size same as example_input for simplicity, or 1
                    batch_size = example_input.shape[0]
                    test_input = torch.randint(0, 1000, (batch_size, seq_len), dtype=example_input.dtype).to(device)
                else:
                    test_input = torch.randint_like(example_input, 0, 1000).to(device)
            else:
                # For CNNs
                test_input = torch.randn_like(example_input).to(device)
            
            test_input_np = test_input.cpu().numpy()
            
            # Get PyTorch output
            original_output = original_model(test_input)
            
            # Handle tuple outputs (RNNs return (output, hidden))
            if isinstance(original_output, tuple):
                original_output = original_output[0]
            
            original_output_np = original_output.detach().cpu().numpy()
            
            # Get ONNX Runtime output
            try:
                onnx_output = session.run([output_name], {input_name: test_input_np})[0]
            except Exception as e:
                print(f"  ✗ Runtime execution failed: {e}")
                return False
            
            # Validate
            is_valid, max_diff = validate_outputs(
                torch.from_numpy(original_output_np),
                torch.from_numpy(onnx_output),
                rtol=rtol,
                atol=atol
            )
            
            if not is_valid:
                print(f"    ✗ Sample {i+1}: FAILED (max diff: {max_diff:.2e})")
                return False
            
            print(f"    ✓ Sample {i+1}: PASSED (max diff: {max_diff:.2e})")
    
    print("  ✓ All validation tests passed!")
    return True


def convert_model(
    model_name: str,
    opset_version: int = 13,
    validate: bool = True,
    dynamic_batch: bool = True,
    verbose: bool = False
) -> bool:
    """
    Convert a specific model to ONNX format.
    """
    config = get_model_config(model_name)
    checkpoint_path = config['checkpoints']['pytorch']
    output_path = config['checkpoints']['onnx']
    
    print(f"\n{'='*70}")
    print(f"Converting {model_name.upper()} to ONNX")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    
    # Create output directory
    create_output_directory(output_path)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    try:
        # Load model
        print("Loading PyTorch model...")
        model_class = get_model_class(model_name)
        model, checkpoint = load_model_from_checkpoint(
            model_class=model_class,
            checkpoint_path=checkpoint_path,
            device=device,
            **config['kwargs']
        )
        print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Prepare example input
        example_input = config['example_input'].to(device)
        
        # Setup dynamic axes
        dynamic_axes = None
        if dynamic_batch:
            input_name = config['input_names'][0]
            output_name = config['output_names'][0]
            
            dynamic_axes = {
                input_name: {0: 'batch_size'},
                output_name: {0: 'batch_size'}
            }
            
            # For RNNs, sequence length should also be dynamic
            if is_rnn_model(model_name):
                dynamic_axes[input_name][1] = 'sequence_length'
                # Only GRU (Language Model) has sequence length in output
                # LSTM (Sentiment) has fixed output shape (batch, 1)
                if model_name == 'gru':
                    dynamic_axes[output_name][1] = 'sequence_length'
                
        print(f"Dynamic axes: {dynamic_axes}")
        
        # Export to ONNX
        print(f"Exporting to ONNX (opset {opset_version})...")
        start_time = time.time()
        
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=config['input_names'],
            output_names=config['output_names'],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=verbose,
            keep_initializers_as_inputs=False
        )
        
        conversion_time = time.time() - start_time
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"✓ Export completed in {conversion_time:.3f}s ({file_size:.2f} MB)")
        
        # Validate
        if validate:
            # Set tolerance based on model type
            if is_rnn_model(model_name):
                rtol = 1e-3
                atol = 5e-3  # Relaxed for RNNs
            else:
                rtol = 1e-3
                atol = 1e-3  # Slightly relaxed for CNNs (FP32 noise)

            if not validate_onnx_model(
                original_model=model,
                onnx_path=output_path,
                example_input=example_input,
                device=device,
                rtol=rtol,
                atol=atol
            ):
                print("\n✗ Validation failed!")
                return False
                
        print(f"\n✓ {model_name.upper()} converted successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ONNX format')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['lenet', 'resnet18', 'lstm', 'gru', 'all'],
        help='Model to convert'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=13,
        help='ONNX opset version (default: 13)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation'
    )
    parser.add_argument(
        '--no-dynamic',
        action='store_true',
        help='Disable dynamic batch size/shapes'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose ONNX export information'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        models = ['lenet', 'resnet18', 'lstm', 'gru']
        print(f"\nConverting all models to ONNX...\n")
        
        results = {}
        for m in models:
            results[m] = convert_model(
                model_name=m,
                opset_version=args.opset,
                validate=not args.no_validate,
                dynamic_batch=not args.no_dynamic,
                verbose=args.verbose
            )
        
        print("\n" + "="*70)
        print("CONVERSION SUMMARY")
        print("="*70)
        for name, success in results.items():
            print(f"  {name:12s}: {'✓ SUCCESS' if success else '✗ FAILED'}")
        print("="*70 + "\n")
        
        sys.exit(0 if all(results.values()) else 1)
    else:
        success = convert_model(
            model_name=args.model,
            opset_version=args.opset,
            validate=not args.no_validate,
            dynamic_batch=not args.no_dynamic,
            verbose=args.verbose
        )
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
