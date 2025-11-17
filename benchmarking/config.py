"""
Configuration loading and validation for benchmarks.

Supports YAML-based configuration files for flexible benchmark definitions.
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import centralized model configurations
from models import MODEL_CONFIGS, get_model_config, get_checkpoint_path


class BenchmarkConfig:
    """Benchmark configuration container."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize from configuration dictionary.
        
        Args:
            config_dict: Configuration loaded from YAML
        """
        self.name = config_dict.get('name', 'benchmark')
        self.description = config_dict.get('description', '')
        
        # Models to benchmark
        self.models = config_dict.get('models', ['lenet'])
        
        # Frameworks to test
        self.frameworks = config_dict.get('frameworks', ['pytorch-eager'])
        
        # Batch sizes to test
        self.batch_sizes = config_dict.get('batch_sizes', [1, 8, 32, 128])
        
        # Benchmark parameters
        self.num_iterations = config_dict.get('num_iterations', 1000)
        self.warmup_iterations = config_dict.get('warmup_iterations', 100)
        
        # Device configuration
        self.device = config_dict.get('device', 'cuda')
        
        # Output configuration
        self.output_dir = config_dict.get('output_dir', 'benchmarking/results')
        
        # Profiling options
        profiling = config_dict.get('profiling', {})
        self.profile_latency = profiling.get('latency', True)
        self.profile_memory = profiling.get('memory', True)
        self.profile_compilation = profiling.get('compilation', True)
        
    def validate(self) -> None:
        """Validate configuration."""
        # Check models
        for model in self.models:
            if model not in MODEL_CONFIGS:
                raise ValueError(f"Unknown model: {model}")
        
        # Check frameworks
        valid_frameworks = ['pytorch-eager', 'torchscript', 'pytorch-compile', 'onnx', 'tensorrt']
        for framework in self.frameworks:
            if framework not in valid_frameworks:
                raise ValueError(f"Unknown framework: {framework}")
        
        # Check device
        if self.device not in ['cuda', 'cpu']:
            raise ValueError(f"Invalid device: {self.device}")
        
        # Check batch sizes
        if not all(bs > 0 for bs in self.batch_sizes):
            raise ValueError("All batch sizes must be positive")
    
    def __repr__(self) -> str:
        return (f"BenchmarkConfig(name={self.name}, "
                f"models={self.models}, "
                f"frameworks={self.frameworks}, "
                f"batch_sizes={self.batch_sizes})")


def load_config(config_path: str) -> BenchmarkConfig:
    """
    Load benchmark configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        BenchmarkConfig object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = BenchmarkConfig(config_dict)
    config.validate()
    
    return config


# get_checkpoint_path is now imported from models.config
# No need to redefine it here

