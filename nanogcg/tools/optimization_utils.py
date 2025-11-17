"""
Optimization utilities for GCG batch processing.
Contains reusable functions for prompt generation, optimization, and analysis.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nanogcg
from nanogcg import GCGConfig, ProbeSamplingConfig


def generate_table_prompt(rows: int, cols: int) -> dict:
    """Generate a customizable table prompt with specified dimensions.

    Args:
        rows: Number of data rows in the table
        cols: Number of metric columns in the table

    Returns:
        dict: Contains the prompt template and list of optim_str placeholders
    """
    # Generate column headers
    col_headers = []
    for i in range(cols):
        col_headers.append(f"Metric {chr(65 + i)}")  # A, B, C, D...

    # Start building the table
    lines = ["Here is a table:\n"]

    # Header row
    header_row = "| | " + " | ".join(col_headers) + " |"
    lines.append(header_row)

    # Separator row
    separator_row = "|" + "---|" * (cols + 1)
    lines.append(separator_row)

    # Data rows with placeholders
    placeholder_counter = 1
    optim_str_placeholders = []

    for row_idx in range(rows):
        row_cells = []
        for col_idx in range(cols):
            placeholder = f"{{optim_str_{placeholder_counter}}}"
            optim_str_placeholders.append(placeholder)
            placeholder_counter += 1
            row_cells.append(placeholder)

        row_name = f"Data {row_idx + 1}"
        data_row = f"| {row_name} | " + " | ".join(row_cells) + " |"
        lines.append(data_row)

    # Add the final instruction
    lines.append("\nBased on this data table, output one word to describe it. Only output the word.")

    template_prompt = "\n".join(lines)

    return {
        "prompt": template_prompt,
        "optim_str_placeholders": optim_str_placeholders,
    }


def generate_simple_prompt() -> dict:
    """Generate a simple prompt with single placeholder.

    Returns:
        dict: Contains the prompt template and list of optim_str placeholders
    """
    template_prompt = "Here is a text: {optim_str_1} \nBased on this text, output one word to describe it. Only output the word."
    return {
        "prompt": template_prompt,
        "optim_str_placeholders": ["{optim_str_1}"],
    }


def load_models(model_name: str, perplexity_model_name: str, device: str = "cuda", dtype: str = "float16"):
    """Load main model and perplexity model.

    Args:
        model_name: Name/path of the main model for optimization
        perplexity_model_name: Name/path of the model for perplexity calculation
        device: Device to run models on
        dtype: Data type for model weights

    Returns:
        tuple: (main_model, main_tokenizer, ppl_model, ppl_tokenizer)
    """
    print(f"Loading main model: {model_name}")
    main_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype)
    ).to(device)
    main_tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading perplexity model: {perplexity_model_name}")
    ppl_model = AutoModelForCausalLM.from_pretrained(perplexity_model_name).to(device)
    ppl_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_name)

    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token

    return main_model, main_tokenizer, ppl_model, ppl_tokenizer


def create_config(num_steps: int = 500, optim_str_init: str = "x x x",
                  probe_sampling_config: ProbeSamplingConfig = None,
                  verbosity: str = "WARNING") -> GCGConfig:
    """Create GCG configuration.

    Args:
        num_steps: Number of optimization steps
        optim_str_init: Initial optimization string
        probe_sampling_config: Optional probe sampling configuration
        verbosity: Logging verbosity level

    Returns:
        GCGConfig: Configuration object
    """
    return GCGConfig(
        verbosity=verbosity,
        num_steps=num_steps,
        optim_str_init=optim_str_init,
        use_prefix_cache=False,
        probe_sampling_config=probe_sampling_config,
    )


def setup_probe_sampling(device: str = "cuda", dtype: str = "float16") -> ProbeSamplingConfig:
    """Setup probe sampling configuration.

    Args:
        device: Device to run draft model on
        dtype: Data type for draft model weights

    Returns:
        ProbeSamplingConfig: Configuration for probe sampling
    """
    draft_model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2", torch_dtype=getattr(torch, dtype)
    ).to(device)
    draft_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    return ProbeSamplingConfig(
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
    )


def run_single_optimization(model, tokenizer, messages, target: str,
                          config: GCGConfig, optim_str_placeholders: list,
                          description: str = "") -> dict:
    """Run a single GCG optimization and return results.

    Args:
        model: Main model for optimization
        tokenizer: Tokenizer for main model
        messages: Message list with prompt
        target: Target string for optimization
        config: GCG configuration
        optim_str_placeholders: List of placeholder strings
        description: Description for logging

    Returns:
        dict: Optimization results with success status, optimized prompt, metrics
    """
    print(f"\n{'='*50}")
    print(f"Optimizing: {description}")
    print(f"Number of placeholders: {len(optim_str_placeholders)}")
    print(f"{'='*50}")

    start_time = time.time()

    try:
        result = nanogcg.run_multigcg(
            model,
            tokenizer,
            messages,
            target,
            config,
            optim_str_placeholders=optim_str_placeholders,
        )

        # Replace placeholders with optimized strings
        optimized_messages = [msg.copy() for msg in messages]
        if hasattr(result, 'best_string') and len(result.best_strings) == 1:
            optimized_messages[-1]["content"] = optimized_messages[-1]["content"].replace("{optim_str}", result.best_string)
        else:
            for i, optim_str in enumerate(result.best_strings):
                optimized_messages[-1]["content"] = optimized_messages[-1]["content"].replace(optim_str_placeholders[i], optim_str)

        optimization_time = time.time() - start_time

        return {
            "success": True,
            "optimized_prompt": optimized_messages[-1]["content"],
            "best_strings": result.best_strings,
            "best_loss": result.best_loss,
            "losses": result.losses,
            "optimization_time": optimization_time,
            "num_steps": len(result.losses)
        }

    except Exception as e:
        print(f"Error during optimization: {e}")
        return {
            "success": False,
            "error": str(e),
            "optimization_time": time.time() - start_time
        }