# AXEN-M (Attention eXtended Efficient Network - Model)

<div align="center">

![AXEN-M](/src/img/R.png)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Overview

AXEN-M is a transformer-based framework designed for large-scale language modeling, fine-tuning, and efficient attention handling. Built by Vinkura, it integrates Infini-Attention with LoRA-based fine-tuning to provide scalability and optimized performance for long-context tasks.

## Features

- **Advanced Attention Mechanisms**
  - Infini-Attention implementation for extended sequence handling
  - Optimized memory usage with sliding window attention
  - Support for sequences up to 32K tokens

- **Model Architecture**
  - LoRA (Low-Rank Adaptation) integration for efficient fine-tuning
  - Compatible with LLaMA model architectures
  - Modular transformer components

- **Training Capabilities**
  - 4-bit quantization support
  - Gradient checkpointing
  - Mixed-precision training
  - Distributed training support

- **Development Tools**
  - Comprehensive data preprocessing pipeline
  - Attention pattern visualization
  - Performance metrics and evaluation
  - Extensive test coverage

![AXEN-M](/src/img/image.png)

# AXEN-M Installation Guide

## Prerequisites

### System Requirements
- Python 3.8 or higher
- CUDA 11.7+ (for GPU support)
- 16GB RAM minimum (32GB recommended)
- Linux, macOS, or Windows (via WSL2)
- Git

### Required Dependencies
- PyTorch 2.0+
- CUDA toolkit (for GPU support)
- cuDNN (for GPU support)

## Project Structure

```
infini-attention/
│
│  
│
├── configs/                     
│   ├── single_node.yaml         # Config for single-node training
│   ├── two_node.yaml            # Config for two-node distributed training
│   └── zero3_offload.json       # DeepSpeed Zero3 offload config
│
├── scripts/
│   └── train.sh                 # Shell script for training
│
├── src/                         # Source code directory
│   ├── __init__.py
│   ├── main.py                  
│   ├── fine_tune.py             
│   ├── llama_model.py           
│   ├── temp_utils.py            
│   └── train.py                 
│
├── tests/                       
│   └── test_llama_model.py
├   └── test_utils.py 
│
├── .gitignore
├── requirements.txt              # List of dependencies
├── README.md                     # Complete project documentation
└── LICENSE
```




## Installation Methods




1. Clone the repository:
```bash
git clone https://github.com/vinkuraai/axen-m.git
cd axen-m
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate  # On Windows
```

3. Install the package:
```bash
# For basic installation
pip install -e .

# For development installation with extra tools
pip install -e ".[dev]"
```

## GPU Support Setup

### CUDA Installation

1. Download CUDA Toolkit 11.7 or higher from NVIDIA website
2. Install CUDA Toolkit following NVIDIA's instructions
3. Verify CUDA installation:
```bash
nvcc --version
```

### Install GPU Dependencies

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

## Installation Verification

Run the verification script:
```bash
python -c "from axen_m import Model; print(Model.version_check())"
```

## Common Installation Issues

### CUDA Not Found
```bash
# Add CUDA to PATH (Linux/macOS)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Add CUDA to PATH (Windows)
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin;%PATH%
```

### Memory Issues During Installation
```bash
# Reduce number of build workers
pip install -e . --no-cache-dir --no-build-isolation
```

## Development Setup

1. Install development dependencies:
```bash
pip install -e ".[dev,test,docs]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

3. Setup documentation build environment:
```bash
cd docs
make html
```

## Configuration

### Environment Variables
```bash
# Required for distributed training
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export WORLD_SIZE="1"
export LOCAL_RANK="0"

# Optional performance tweaks
export CUDA_LAUNCH_BLOCKING="1"
export CUDA_VISIBLE_DEVICES="0,1"
```

### Project Configuration File
Create `config.yaml` in your project root:
```yaml
model:
  type: "llama"
  size: "7b"
  quantization: 4

training:
  batch_size: 32
  gradient_accumulation: 4
  mixed_precision: "fp16"

system:
  num_workers: 4
  pin_memory: true
```

## Troubleshooting

### Error: CUDA capability sm_XX not supported
Solution: Update your GPU drivers or modify CUDA architecture flags:
```bash
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
pip install -e .
```

### Error: Memory allocation failed
Solution: Enable gradient checkpointing in your configuration:
```python
model_config = {
    "gradient_checkpointing": True,
    "max_memory": {0: "12GB"}
}
```


## Performance Benchmarks

| Model Size | Sequence Length | Memory Usage | Training Speed | Inference Speed |
|------------|----------------|--------------|----------------|-----------------|
| 7B         | 2048           | 14GB         | 32 samples/s   | 50 tokens/s     |
| 7B         | 8192           | 20GB         | 24 samples/s   | 45 tokens/s     |
| 7B         | 32768          | 28GB         | 16 samples/s   | 35 tokens/s     |

## Contributing

We welcome contributions! 

- Code style and standards
- Pull request process
- Development setup
- Testing requirements

## Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{axen_m2025,
  author = {VinkuraAI},
  title = {AXEN-M: Attention eXtended Efficient Network},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/vinkuraAI/axen-m}
}
```

## Support

- Documentation: [https://axen-m.readthedocs.io/](https://axen-m.readthedocs.io/)
- Issue Tracker: [GitHub Issues](https://github.com/vinkuraai/axen-m/issues)
- Discussions: [GitHub Discussions](https://github.com/vinkuraai/axen-m/discussions)

## Acknowledgments

- The LLaMA team for their foundational work
- The LoRA authors for their efficient fine-tuning approach
- The open-source AI community

---

<div align="center">
<sub>Built by Vinkura with a focus on efficiency and scalability.</sub>
</div>
