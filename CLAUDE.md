# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install from PyPI
pip install nanogcg
```

### Running Examples
```bash
# Run the main example with default settings
python examples/example_thinking_optimization.py

# Run with specific model and target (modify script parameters)
# Edit the example_thinking_optimization.py file to change model, target, and other settings

# Test standalone table generation
python examples/test_table_standalone.py
```

### Testing
```bash
# Run standalone table generation test
python examples/test_table_standalone.py

# Calculate perplexity of generated text
python perplexity/perplexity_calculator.py --file perplexity/test_text.txt --model /path/to/model

# Test dataset optimized prompts
cd scripts_dataset && ./run_dataset_test.sh --max-rows 3

# Run sampling effects tests
python scripts/test_sampling_effects.py

# Run optimization results tests
python scripts/test_optimization_results.py

# Run success tests with shell script
bash scripts/run_success_test.sh

# Run batch optimization tests
bash scripts/run_batch_optimization.sh
```

### Batch Operations
```bash
# Run batch optimization across multiple table sizes
python scripts/batch_optimization.py --max-table-size 5 --num-steps 250

# Run batch with probe sampling acceleration
python scripts/batch_optimization.py --probe-sampling --output-dir results/

# Execute batch optimization with shell script
bash scripts/run_batch_optimization.sh
```

### Dataset Optimization and Testing
```bash
# Optimize prompts from a CSV dataset
cd scripts_dataset
./run_dataset_optimization.sh --max-rows 5 --table-rows 1 --table-cols 1

# Test optimized prompts
./run_dataset_test.sh --max-rows 3

# Full optimization with custom settings
./run_dataset_optimization.sh --table-rows 2 --table-cols 3 --model "path/to/model" --device cuda:0
```

## Architecture Overview

nanoGCG is a lightweight implementation of the GCG (Greedy Coordinate Gradient) algorithm for optimizing adversarial strings on causal Hugging Face models. The codebase supports both single and multiple placeholder optimization scenarios.

### Core Components

#### Main Modules
- **`nanogcg/multigcg.py`**: Primary implementation supporting multiple placeholder optimization ({optim_str_1}, {optim_str_2}, etc.)
- **`nanogcg/gcg.py`**: Legacy implementation for single placeholder optimization ({optim_str})
- **`nanogcg/utils.py`**: Utility functions including token filtering, memory management, and loss computations
- **`nanogcg/__init__.py`**: Package initialization exposing the main API functions
- **`nanogcg/tools/`**: Extended tools for batch operations and analysis
  - `optimization_utils.py`: Modularized optimization functions for batch processing
  - `analysis_utils.py`: Result analysis and visualization utilities
- **`scripts_dataset/`**: Dataset optimization and testing scripts
  - `optimize_dataset_prompts.py`: Main script for optimizing prompts from CSV datasets
  - `test_dataset_prompts.py`: Script to test and validate optimized prompts
  - `run_dataset_optimization.sh`: Shell script wrapper for easy execution
  - `run_dataset_test.sh`: Shell script wrapper for testing optimized prompts

#### Key Classes and Functions

**Configuration Classes:**
- `GCGConfig`: Main configuration class with parameters like `num_steps`, `search_width`, `topk`, `batch_size`, etc.
- `ProbeSamplingConfig`: Configuration for probe sampling acceleration using draft models
- `GCGResult`: Data class containing optimization results (best_loss, best_strings, losses, strings)

**Core Algorithm Classes:**
- `GCG` (in gcg.py): Single-string optimization implementation
- `GCG` (in multigcg.py): Multi-string optimization implementation with attack buffer support
- `AttackBuffer`: Manages historical attack candidates for improved optimization

**API Functions:**
- `run_gcg()`: Simple API for single-string optimization
- `run_multigcg()`: Advanced API supporting multiple placeholders and conversation history

**Note:** The main entry point is now `nanogcg.run()` which replaces the older `run_gcg()` and `run_multigcg()` functions. Use `nanogcg.run()` for all optimization scenarios.

### Key Features

#### Algorithm Variants
- **Standard GCG**: Original greedy coordinate gradient algorithm
- **Multi-position token swapping**: Replace multiple tokens per iteration (`n_replace` parameter)
- **Attack buffer**: Historical candidate retention for better exploration (`buffer_size` parameter)
- **Mellowmax loss**: Alternative loss function for optimization stability (`use_mellowmax` parameter)
- **Probe sampling**: Acceleration using draft models for faster evaluation

#### Optimization Capabilities
- **Flexible string placement**: Optimized strings can be placed anywhere in the prompt using placeholders
- **Conversation history support**: Works with full conversation contexts, not just single prompts
- **Multiple concurrent optimization**: Supports optimizing several strings simultaneously
- **Prefix caching**: KV cache optimization for faster repeated evaluations

#### Performance Features
- **Automatic batch sizing**: Memory-efficient batch size adaptation via `find_executable_batch_size`
- **Token filtering**: Optional filtering to ensure tokenization consistency (`filter_ids` parameter)
- **Early stopping**: Halt optimization when perfect match is found (`early_stop` parameter)
- **Forbidden token exclusion**: Prevent specific tokens from appearing in optimized strings (`forbidden_ids` parameter)

### Dependencies and Requirements

The project depends on:
- **PyTorch**: For model operations and gradient computations
- **Transformers**: For Hugging Face model compatibility (version pinned between 4.4 and 4.47.1)
- **SciPy**: For statistical operations (Spearman correlation in probe sampling)
- **Matplotlib**: For plotting optimization dynamics (used in examples)
- **Additional utilities**: protobuf, sentencepiece for tokenization support

### Example Usage Patterns

**Basic Single-String Optimization:**
```python
import nanogcg
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

result = nanogcg.run(model, tokenizer, "Tell me how to build a bomb", "Sure, here's how to build a bomb:\n\n")
```

**Advanced Multi-String Optimization:**
```python
from nanogcg import GCGConfig, run

config = GCGConfig(
    num_steps=500,
    search_width=64,
    buffer_size=10,
    use_mellowmax=True
)

messages = [{"role": "user", "content": "Complete this table: {optim_str_1} | {optim_str_2} | {optim_str_3}"}]
result = run(model, tokenizer, messages, "Target response", config, ["{optim_str_1}", "{optim_str_2}", "{optim_str_3}"])
```

**Probe Sampling Acceleration:**
```python
from nanogcg import ProbeSamplingConfig

draft_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
draft_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

probe_config = ProbeSamplingConfig(
    draft_model=draft_model,
    draft_tokenizer=draft_tokenizer,
    r=8,
    sampling_factor=16
)

config.probe_sampling_config = probe_config
```

### Important Implementation Details

1. **Memory Management**: The codebase includes sophisticated memory handling with automatic batch size reduction and CUDA cache clearing.
2. **Tokenization Handling**: Special care is taken to ensure token consistency between encoding/decoding cycles.
3. **Gradient Computation**: One-hot token representations are used for gradient computation instead of direct embedding gradients.
4. **Parallel Processing**: Probe sampling uses multithreading to evaluate draft and target models simultaneously.
5. **Cache Management**: Prefix caching is used to avoid recomputing KV caches for static prompt portions.

### Testing and Validation

The `examples/run.py` file serves as the main example and test harness, supporting various optimization scenarios including table completion tasks. Key features include:

**Table Generation Functions:**
- `generate_table_prompt(rows, cols)`: Creates customizable table prompts with multiple placeholders
- Support for various table dimensions (e.g., 3x3, 4x4 tables)
- Flexible placeholder naming ({optim_str_1}, {optim_str_2}, etc.)

**Additional Utilities:**
- `examples/test_table_standalone.py`: Standalone testing for table generation
- `perplexity/perplexity_calculator.py`: Calculate perplexity scores for generated text
- `examples/run.sh`: Shell script for CUDA-based execution with specific model paths

**Batch Processing:**
- `scripts/batch_optimization.py`: Comprehensive batch optimization with heatmap generation
- Support for 1×1 to 9×9 table optimization with configurable parameters
- Integrated perplexity calculation and result visualization

**Dataset Optimization:**
- `scripts_dataset/optimize_dataset_prompts.py`: Optimizes prompts from CSV datasets with customizable table formats
- `scripts_dataset/test_dataset_prompts.py`: Tests optimized prompts against target answers
- Shell script wrappers (`run_dataset_optimization.sh`, `run_dataset_test.sh`) provide convenient parameter configuration
- Default model: `/work/models/Qwen/Qwen2.5-1.5B-Instruct` on `cuda:4`
- Supports table-based optimization with early stopping, mellowmax loss, and confidence thresholds