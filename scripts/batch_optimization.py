"""
Batch optimization script for running GCG on multiple table sizes and simple prompts.
Calculates perplexity and generates heatmap visualization.

This script uses modularized functions from nanogcg.tools package
to perform batch optimization and analysis.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

# Add parent directory to path to import nanogcg tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogcg.tools import (
    generate_table_prompt, generate_simple_prompt, load_models,
    create_config, setup_probe_sampling, run_single_optimization,
    calculate_perplexity_for_prompt, create_heatmap, save_results,
    create_summary_report, create_loss_plots, calculate_statistics,
    clear_gpu_cache
)
from nanogcg import GCGConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch optimize GCG on various table sizes")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                       help="Main model for optimization")
    parser.add_argument("--target", type=str, default="Copyright",
                       help="Target word for optimization")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run models on")
    parser.add_argument("--dtype", type=str, default="float16",
                       help="Data type for model weights")
    parser.add_argument("--probe-sampling", action="store_true",
                       help="Enable probe sampling for acceleration")
    parser.add_argument("--max-table-size", type=int, default=9,
                       help="Maximum table size (e.g., 9 for 9x9)")
    parser.add_argument("--num-steps", type=int, default=500,
                       help="Number of optimization steps")
    parser.add_argument("--output-dir", type=str, default="batch_results",
                       help="Output directory for results")
    parser.add_argument("--perplexity-model", type=str, default="/work/models/openai-community/gpt2",
                       help="Model for perplexity calculation")
    parser.add_argument("--optim-init", type=str, default="x x x",
                       help="Initial optimization string")
    parser.add_argument("--reverse", action="store_true",
                       help="Start from largest table size (recommended to avoid wasted time)")
    return parser.parse_args()


def run_simple_optimization(main_model, main_tokenizer, ppl_model, ppl_tokenizer,
                           target: str, config: GCGConfig, device: str) -> dict:
    """Run optimization for simple prompt."""
    print(f"\n{'='*60}")
    print("OPTIMIZING SIMPLE PROMPT")
    print(f"{'='*60}")

    simple_prompt_data = generate_simple_prompt()
    simple_messages = [{"role": "user", "content": simple_prompt_data["prompt"]}]

    result = run_single_optimization(
        main_model, main_tokenizer, simple_messages, target, config,
        simple_prompt_data["optim_str_placeholders"], "Simple Prompt"
    )

    if result["success"]:
        perplexity = calculate_perplexity_for_prompt(
            result["optimized_prompt"], ppl_model, ppl_tokenizer, device
        )
        result["perplexity"] = perplexity
        print(f"Simple prompt perplexity: {perplexity:.2f}")

    return result


def run_table_optimizations(main_model, main_tokenizer, ppl_model, ppl_tokenizer,
                           target: str, config: GCGConfig, max_size: int, device: str,
                           reverse: bool = False) -> tuple:
    """Run optimizations for all table sizes."""
    results = {}
    perplexity_matrix = np.full((max_size, max_size), np.nan)

    print(f"\n{'='*60}")
    if reverse:
        print(f"OPTIMIZING TABLE PROMPTS (REVERSE MODE: {max_size}x{max_size} → 1x1)")
        print("Starting with largest table size to detect memory issues early")
    else:
        print("OPTIMIZING TABLE PROMPTS (NORMAL MODE: 1x1 → {max_size}x{max_size})")
    print(f"{'='*60}")

    # Create list of (rows, cols) pairs in desired order
    table_sizes = []
    if reverse:
        # Start from largest: max_size x max_size, max_size x (max_size-1), ..., 1x1
        for rows in range(max_size, 0, -1):
            for cols in range(max_size, 0, -1):
                table_sizes.append((rows, cols))
    else:
        # Normal order: 1x1, 1x2, ..., max_size x max_size
        for rows in range(1, max_size + 1):
            for cols in range(1, max_size + 1):
                table_sizes.append((rows, cols))

    for rows, cols in table_sizes:
        description = f"Table {rows}x{cols}"

        # Show progress
        total_tables = len(table_sizes)
        current_index = table_sizes.index((rows, cols))
        progress = (current_index + 1) / total_tables * 100
        print(f"[{current_index + 1}/{total_tables}] ({progress:.1f}%) Optimizing {description}")

        try:
            table_prompt_data = generate_table_prompt(rows, cols)
            table_messages = [{"role": "user", "content": table_prompt_data["prompt"]}]

            result = run_single_optimization(
                main_model, main_tokenizer, table_messages, target, config,
                table_prompt_data["optim_str_placeholders"], description
            )

            if result["success"]:
                perplexity = calculate_perplexity_for_prompt(
                    result["optimized_prompt"], ppl_model, ppl_tokenizer, device
                )
                result["perplexity"] = perplexity
                perplexity_matrix[rows-1, cols-1] = perplexity
                print(f"  ✓ {description} perplexity: {perplexity:.2f}")
            else:
                print(f"  ❌ {description} optimization failed")

            # Store result
            key = f"table_{rows}x{cols}"
            results[key] = result

            # Clear GPU cache periodically
            if (current_index + 1) % 5 == 0:
                clear_gpu_cache()

        except Exception as e:
            print(f"  ❌ {description} failed with error: {e}")
            if reverse:
                print("  💡 In reverse mode, you can stop early to avoid wasting time")

            # Store failed result
            key = f"table_{rows}x{cols}"
            results[key] = {"success": False, "error": str(e)}

    return results, perplexity_matrix


def main():
    """Main function to run batch optimization."""
    args = parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"batch_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Print configuration
    print(f"Batch optimization started at: {datetime.now()}")
    print(f"Output directory: {output_dir}")
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Target: {args.target}")
    print(f"  Max table size: {args.max_table_size}x{args.max_table_size}")
    print(f"  Optimization steps: {args.num_steps}")
    print(f"  Probe sampling: {args.probe_sampling}")
    print(f"  Device: {args.device}")
    print(f"  Data type: {args.dtype}")

    # Load models
    main_model, main_tokenizer, ppl_model, ppl_tokenizer = load_models(
        args.model, args.perplexity_model, args.device, args.dtype
    )

    # Setup probe sampling if requested
    probe_sampling_config = None
    if args.probe_sampling:
        probe_sampling_config = setup_probe_sampling(args.device, args.dtype)

    # Create configuration
    config = create_config(
        num_steps=args.num_steps,
        optim_str_init=args.optim_init,
        probe_sampling_config=probe_sampling_config,
        verbosity="WARNING"  # Reduce verbosity for batch processing
    )

    # Run optimizations
    all_results = {}

    # Simple prompt optimization
    simple_result = run_simple_optimization(
        main_model, main_tokenizer, ppl_model, ppl_tokenizer,
        args.target, config, args.device
    )
    all_results["simple"] = simple_result

    # Table optimizations
    table_results, perplexity_matrix = run_table_optimizations(
        main_model, main_tokenizer, ppl_model, ppl_tokenizer,
        args.target, config, args.max_table_size, args.device, args.reverse
    )
    all_results.update(table_results)

    # Save results and create visualizations
    print(f"\n{'='*60}")
    print("SAVING RESULTS AND CREATING VISUALIZATIONS")
    print(f"{'='*60}")

    # Configuration dictionary for report
    config_dict = {
        "model": args.model,
        "target": args.target,
        "max_table_size": args.max_table_size,
        "num_steps": args.num_steps,
        "probe_sampling": args.probe_sampling,
        "device": args.device,
        "dtype": args.dtype,
        "perplexity_model": args.perplexity_model
    }

    # Save results
    results_file = os.path.join(output_dir, "optimization_results.json")
    save_results(all_results, results_file)

    # Create heatmap
    heatmap_path = os.path.join(output_dir, "perplexity_heatmap.png")
    create_heatmap(perplexity_matrix, heatmap_path,
                  f"Perplexity Heatmap for {args.target} Optimization")

    # Create summary report
    summary_path = os.path.join(output_dir, "summary_report.txt")
    create_summary_report(all_results, config_dict, summary_path, perplexity_matrix)

    # Create loss plots
    plots_dir = os.path.join(output_dir, "loss_plots")
    create_loss_plots(all_results, plots_dir)

    # Calculate and print statistics
    stats = calculate_statistics(all_results)

    print(f"\n{'='*60}")
    print("BATCH OPTIMIZATION COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Files created:")
    print(f"  - Summary report: {summary_path}")
    print(f"  - Heatmap: {heatmap_path}")
    print(f"  - Full results: {results_file}")
    print(f"  - Loss plots: {plots_dir}")

    # Print key results
    if "simple_perplexity" in stats:
        print(f"\nSimple prompt perplexity: {stats['simple_perplexity']:.2f}")

    if "successful_table_optimizations" in stats:
        total_possible = args.max_table_size * args.max_table_size
        success_rate = stats["successful_table_optimizations"] / total_possible * 100
        print(f"Table optimization success rate: {success_rate:.1f}% "
              f"({stats['successful_table_optimizations']}/{total_possible})")

    if "table_perplexity_mean" in stats:
        print(f"Table perplexity statistics:")
        print(f"  Mean: {stats['table_perplexity_mean']:.2f}")
        print(f"  Range: {stats['table_perplexity_min']:.2f} - {stats['table_perplexity_max']:.2f}")

    # Find best and worst performing table sizes
    table_results = [(key, result) for key, result in all_results.items()
                    if key.startswith("table_") and result["success"]]
    if table_results:
        table_results.sort(key=lambda x: x[1]["perplexity"])
        best_table = table_results[0]
        worst_table = table_results[-1]
        print(f"\nBest table performance: {best_table[0]} (PPL: {best_table[1]['perplexity']:.2f})")
        print(f"Worst table performance: {worst_table[0]} (PPL: {worst_table[1]['perplexity']:.2f})")


if __name__ == "__main__":
    main()