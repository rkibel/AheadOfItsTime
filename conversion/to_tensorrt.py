"""
Convert ONNX models to TensorRT engines with PyCUDA validation.

Usage:
    python conversion/to_tensorrt.py --model lenet
    python conversion/to_tensorrt.py --model all
    python conversion/to_tensorrt.py --model resnet18 --tolerance 1e-4
"""

import argparse
import sys
import shutil
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import get_model_config, get_model_class

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    import onnx
    from onnx import helper
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    sys.exit(1)


class TensorRTLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)
        self.errors = []
        
    def log(self, severity, msg):
        if severity == trt.ILogger.ERROR:
            self.errors.append(msg)
            print(f"[TRT ERROR] {msg}")
        elif severity == trt.ILogger.WARNING:
            print(f"[TRT WARNING] {msg}")
        elif severity == trt.ILogger.INFO:
            print(f"[TRT INFO] {msg}")


def fix_rnn_onnx(onnx_path: str, output_path: str) -> bool:
    """Add Cast nodes to convert float32 inputs to int64 for RNN embeddings."""
    model = onnx.load(onnx_path)
    graph = model.graph
    
    has_int_input = any(inp.type.tensor_type.elem_type == onnx.TensorProto.INT64 
                        for inp in graph.input)
    if not has_int_input:
        return False
    
    print("  ✓ Adding cast nodes for RNN embeddings...")
    modified_nodes = []
    input_remap = {}
    
    for input_tensor in graph.input:
        if input_tensor.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            orig_name = input_tensor.name
            float_name = orig_name + "_float32"
            cast_name = orig_name + "_int64"
            
            input_tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            input_tensor.name = float_name
            
            modified_nodes.append(helper.make_node('Cast', [float_name], [cast_name], 
                                                   to=onnx.TensorProto.INT64))
            input_remap[orig_name] = cast_name
    
    # Update node inputs
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in input_remap:
                node.input[i] = input_remap[inp]
    
    # Prepend cast nodes
    old_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(modified_nodes + old_nodes)
    onnx.save(model, output_path)
    return True


def build_engine(onnx_path: str, engine_path: str) -> bool:
    """Build TensorRT engine from ONNX with Ampere GPU workarounds."""
    print(f"\nBuilding engine: {Path(onnx_path).name} -> {Path(engine_path).name}")
    print(f"  TensorRT {trt.__version__} | Precision: FP32")
    
    logger = TensorRTLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("  ✗ Parse failed:", logger.errors)
            return False
    
    config = builder.create_builder_config()
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    else:
        config.max_workspace_size = 2 << 30
    
    # Handle dynamic shapes
    if any(-1 in network.get_input(i).shape for i in range(network.num_inputs)):
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = [(1 if d == -1 else d, 1 if d == -1 else d, 512 if d == -1 else d) 
                     for d in inp.shape]
            profile.set_shape(inp.name, *zip(*shape))
        config.add_optimization_profile(profile)
    
    # Disable Cask for RTX 30-series compatibility
    if hasattr(config, 'set_tactic_sources'):
        tactics = sum(1 << int(getattr(trt.TacticSource, t)) 
                     for t in ['CUDNN', 'CUBLAS', 'CUBLAS_LT'] 
                     if hasattr(trt.TacticSource, t))
        config.set_tactic_sources(tactics)
    
    if hasattr(trt.BuilderFlag, 'TF32'):
        config.clear_flag(trt.BuilderFlag.TF32)
    
    print("  Building (may take a minute)...")
    start = time.time()
    
    try:
        if hasattr(builder, 'build_serialized_network'):
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                return False
            Path(engine_path).write_bytes(serialized)
        else:
            engine = builder.build_engine(network, config)
            if engine is None:
                return False
            Path(engine_path).write_bytes(engine.serialize())
    except Exception as e:
        print(f"  ✗ Build failed: {e}")
        return False
    
    size = Path(engine_path).stat().st_size / (1024 * 1024)
    print(f"  ✓ Built in {time.time()-start:.1f}s ({size:.1f} MB)")
    return True


def validate_engine(engine_path: str, example_input: np.ndarray, 
                    model_info: tuple, tolerance: float = 1e-3) -> bool:
    """Validate TensorRT engine against PyTorch model."""
    import torch
    from conversion.utils import load_model_from_checkpoint
    
    print("\nValidating with PyCUDA...")
    
    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if not engine:
        return False
    
    context = engine.create_execution_context()
    
    # Get I/O info
    input_name, output_names, output_shapes = None, [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name, input_shape = name, tuple(shape)
        else:
            output_names.append(name)
            output_shapes.append(tuple(shape))
    
    # Handle dynamic shapes
    if -1 in input_shape:
        input_shape = tuple(example_input.shape)
        context.set_input_shape(input_name, input_shape)
        output_shapes = []
        for name in output_names:
            shape = tuple(context.get_tensor_shape(name))
            if -1 in shape:
                shape = tuple(input_shape[i] if shape[i] == -1 and i < len(input_shape) else 
                            1 if shape[i] == -1 else shape[i] for i in range(len(shape)))
            output_shapes.append(shape)
    
    # Load PyTorch model
    model_class, model_kwargs, ckpt_path = model_info
    pytorch_model, _ = load_model_from_checkpoint(model_class, ckpt_path, 'cpu', **model_kwargs)
    pytorch_model.eval()
    
    # Allocate buffers
    h_input = cuda.pagelocked_empty(int(np.prod(input_shape)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    h_outputs, d_outputs = [], []
    for shape in output_shapes:
        h = cuda.pagelocked_empty(int(np.prod([max(d, 1) for d in shape])), dtype=np.float32)
        d_outputs.append(cuda.mem_alloc(h.nbytes))
        h_outputs.append(h)
    
    stream = cuda.Stream()
    context.set_tensor_address(input_name, int(d_input))
    for name, d_out in zip(output_names, d_outputs):
        context.set_tensor_address(name, int(d_out))
    
    # Run validation
    passed, max_diffs, mean_diffs = 0, [], []
    is_int_input = example_input.dtype in [np.int32, np.int64]
    
    for i in range(5):
        if is_int_input:
            test_int = np.random.randint(0, 1000, input_shape, dtype=np.int64)
            test_input = test_int.astype(np.float32)
        else:
            test_input = np.random.randn(*input_shape).astype(np.float32)
        
        with torch.no_grad():
            pt_in = torch.from_numpy(test_int if is_int_input else test_input)
            pt_out = pytorch_model(pt_in)
            if isinstance(pt_out, tuple):
                pt_out = pt_out[0]
            pt_out = pt_out.numpy()
        
        np.copyto(h_input, test_input.ravel())
        cuda.memcpy_htod_async(d_input, h_input, stream)
        
        if not context.execute_async_v3(stream_handle=stream.handle):
            print(f"    ✗ Test {i+1}: Execution failed")
            continue
        
        cuda.memcpy_dtoh_async(h_outputs[0], d_outputs[0], stream)
        stream.synchronize()
        
        trt_out = h_outputs[0].reshape(output_shapes[0])
        max_diff = np.abs(pt_out - trt_out).max()
        mean_diff = np.abs(pt_out - trt_out).mean()
        
        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)
        
        if max_diff <= tolerance:
            print(f"    ✓ Test {i+1}: max_diff={max_diff:.2e}")
            passed += 1
        else:
            print(f"    ✗ Test {i+1}: max_diff={max_diff:.2e} > {tolerance:.2e}")
    
    if max_diffs:
        print(f"  Accuracy: max={max(max_diffs):.2e}, mean={max(mean_diffs):.2e}")
    
    if passed >= 4:
        print(f"  ✓ Validation passed ({passed}/5)")
        return True
    else:
        print(f"  ✗ Validation failed ({passed}/5)")
        return False


def convert_model(model_name: str, validate: bool = True, tolerance: float = 1e-3) -> bool:
    """Convert model from ONNX to TensorRT with validation."""
    config = get_model_config(model_name)
    onnx_src = Path(config['checkpoints']['onnx'])
    engine_path = Path(config['checkpoints']['tensorrt'])
    
    # Temp files
    temp_work = Path(f'checkpoints/onnx/{model_name}_trt_work.onnx')
    temp_fixed = Path(f'checkpoints/onnx/{model_name}_trt_fixed.onnx')
    
    try:
        print(f"\n{'='*70}")
        print(f"Converting {model_name.upper()} to TensorRT")
        print(f"{'='*70}")
        
        if not onnx_src.exists():
            print(f"✗ ONNX not found: {onnx_src}")
            return False
        
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_src, temp_work)
        
        # Fix for RNN models
        onnx_to_use = temp_fixed if fix_rnn_onnx(str(temp_work), str(temp_fixed)) else temp_work
        
        # Build engine
        if not build_engine(str(onnx_to_use), str(engine_path)):
            return False
        
        # Validate
        if validate:
            example_input = config['example_input']
            if hasattr(example_input, 'numpy'):
                example_input = example_input.cpu().numpy()
            
            model_info = (get_model_class(model_name), 
                         config.get('kwargs', {}), 
                         config['checkpoints']['pytorch'])
            
            if not validate_engine(str(engine_path), example_input, model_info, tolerance):
                return False
        
        print(f"\n{'='*70}")
        print(f"✓ {model_name.upper()} converted successfully")
        print(f"{'='*70}\n")
        return True
        
    finally:
        # Cleanup temp files
        for temp in [temp_work, temp_fixed]:
            if temp.exists():
                temp.unlink()


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT (FP32)')
    parser.add_argument('--model', required=True, 
                       choices=['lenet', 'resnet18', 'lstm', 'gru', 'all'],
                       help='Model to convert')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                       help='Validation tolerance (default: 1e-3)')
    args = parser.parse_args()
    
    if args.model == 'all':
        models = ['lenet', 'resnet18', 'lstm', 'gru']
        print(f"\nConverting all models to TensorRT...\n")
        
        results = {m: convert_model(m, not args.no_validate, args.tolerance) for m in models}
        
        print("\n" + "="*70)
        print("CONVERSION SUMMARY")
        print("="*70)
        for name, success in results.items():
            print(f"  {name:12s}: {'✓ SUCCESS' if success else '✗ FAILED'}")
        print("="*70 + "\n")
        
        sys.exit(0 if all(results.values()) else 1)
    else:
        success = convert_model(args.model, not args.no_validate, args.tolerance)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
