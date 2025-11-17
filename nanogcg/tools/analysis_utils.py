"""
Analysis utilities for GCG results processing and visualization.
Contains functions for perplexity calculation, heatmap generation, and report creation.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any
import torch

from perplexity.perplexity_calculator import calculate_perplexity


def calculate_perplexity_for_prompt(prompt_text: str, model, tokenizer, device: str) -> float:
    """Calculate perplexity for a given prompt.

    Args:
        prompt_text: The optimized prompt text
        model: Model for perplexity calculation
        tokenizer: Tokenizer for perplexity model
        device: Device to run calculation on

    Returns:
        float: Perplexity score (inf if calculation fails)
    """
    try:
        perplexity = calculate_perplexity(prompt_text, model, tokenizer, device)
        return perplexity
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return float('inf')


def create_heatmap(perplexity_matrix: np.ndarray, output_path: str, title: str = "Perplexity Heatmap"):
    """Create and save heatmap visualization.

    Args:
        perplexity_matrix: 2D numpy array with perplexity values
        output_path: Path to save the heatmap image
        title: Title for the heatmap
    """
    plt.figure(figsize=(12, 10))

    # Create heatmap with blue-green color scheme (better contrast)
    mask = np.isnan(perplexity_matrix)  # Mask NaN values
    sns.heatmap(perplexity_matrix,
                annot=True,
                fmt='.2f',
                cmap='YlGnBu',
                square=True,
                mask=mask,
                cbar_kws={'label': 'Perplexity'},
                annot_kws={'size': 10})

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Number of Columns', fontsize=12)
    plt.ylabel('Number of Rows', fontsize=12)

    # Set tick labels
    plt.xticks(np.arange(perplexity_matrix.shape[1]) + 0.5,
               range(1, perplexity_matrix.shape[1] + 1))
    plt.yticks(np.arange(perplexity_matrix.shape[0]) + 0.5,
               range(1, perplexity_matrix.shape[0] + 1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to: {output_path}")


def save_results(results: Dict[str, Any], output_path: str):
    """Save optimization results to JSON file.

    Args:
        results: Dictionary containing all optimization results
        output_path: Path to save the JSON file
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    converted_results = convert_numpy_types(results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False)


def create_summary_report(results: Dict[str, Any], config: Dict[str, Any],
                         output_path: str, perplexity_matrix: np.ndarray):
    """Create a comprehensive summary report.

    Args:
        results: Dictionary containing all optimization results
        config: Configuration parameters used
        output_path: Path to save the report
        perplexity_matrix: 2D array of perplexity values for tables
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("BATCH OPTIMIZATION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Model: {config.get('model', 'N/A')}\n")
        f.write(f"Target: {config.get('target', 'N/A')}\n")
        f.write(f"Max table size: {config.get('max_table_size', 'N/A')}x{config.get('max_table_size', 'N/A')}\n")
        f.write(f"Optimization steps: {config.get('num_steps', 'N/A')}\n")
        f.write(f"Probe sampling: {config.get('probe_sampling', 'N/A')}\n\n")

        # Simple prompt results
        f.write("SIMPLE PROMPT RESULTS\n")
        f.write("-" * 30 + "\n")
        if "simple" in results and results["simple"]["success"]:
            sr = results["simple"]
            f.write(f"Best loss: {sr['best_loss']:.4f}\n")
            f.write(f"Perplexity: {sr['perplexity']:.2f}\n")
            f.write(f"Optimization time: {sr['optimization_time']:.2f}s\n")
            f.write(f"Number of steps: {sr['num_steps']}\n")
            f.write(f"Optimized prompt: {sr['optimized_prompt'][:200]}...\n")
        else:
            f.write("Simple prompt optimization failed\n")

        # Table optimization summary
        f.write(f"\nTABLE OPTIMIZATION RESULTS\n")
        f.write("-" * 30 + "\n")
        successful_count = 0
        total_count = 0
        table_perplexities = []

        for key, result in results.items():
            if key.startswith("table_"):
                total_count += 1
                if result["success"]:
                    successful_count += 1
                    table_perplexities.append(result["perplexity"])
                    # Extract dimensions from key like "table_3x4"
                    dims = key.replace("table_", "").split("x")
                    rows, cols = int(dims[0]), int(dims[1])
                    f.write(f"{rows}x{cols}: Loss={result['best_loss']:.4f}, "
                           f"PPL={result['perplexity']:.2f}, "
                           f"Time={result['optimization_time']:.2f}s\n")

        # Statistics
        f.write(f"\nSUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Successful optimizations: {successful_count}/{total_count}\n")

        if table_perplexities:
            f.write(f"Table perplexity stats:\n")
            f.write(f"  Mean: {np.mean(table_perplexities):.2f}\n")
            f.write(f"  Std: {np.std(table_perplexities):.2f}\n")
            f.write(f"  Min: {np.min(table_perplexities):.2f}\n")
            f.write(f"  Max: {np.max(table_perplexities):.2f}\n")
            f.write(f"  Median: {np.median(table_perplexities):.2f}\n")

        # Best and worst performing tables
        table_results = [(key, result) for key, result in results.items()
                        if key.startswith("table_") and result["success"]]
        if table_results:
            table_results.sort(key=lambda x: x[1]["perplexity"])
            best_table = table_results[0]
            worst_table = table_results[-1]
            f.write(f"\nBest table performance: {best_table[0]} (PPL: {best_table[1]['perplexity']:.2f})\n")
            f.write(f"Worst table performance: {worst_table[0]} (PPL: {worst_table[1]['perplexity']:.2f})\n")

        # Failed optimizations
        failed_results = [(key, result) for key, result in results.items()
                         if not result["success"]]
        if failed_results:
            f.write(f"\nFAILED OPTIMIZATIONS\n")
            f.write("-" * 25 + "\n")
            for key, result in failed_results:
                f.write(f"{key}: {result.get('error', 'Unknown error')}\n")


def create_loss_plots(results: Dict[str, Any], output_dir: str):
    """Create loss optimization plots for all successful optimizations.

    Args:
        results: Dictionary containing all optimization results
        output_dir: Directory to save plot images
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, result in results.items():
        if result["success"] and "losses" in result and result["losses"]:
            plt.figure(figsize=(10, 6))
            plt.plot(result["losses"])
            plt.title(f"Loss Optimization Dynamics - {key}")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Loss")
            plt.grid(True, alpha=0.3)

            # Add best loss annotation
            best_loss_idx = np.argmin(result["losses"])
            plt.axvline(x=best_loss_idx, color='red', linestyle='--', alpha=0.7)
            plt.text(best_loss_idx, result["losses"][best_loss_idx],
                    f'Best: {result["losses"][best_loss_idx]:.4f}',
                    ha='center', va='bottom')

            plot_path = os.path.join(output_dir, f"loss_plot_{key}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

    print(f"Loss plots saved to: {output_dir}")


def calculate_statistics(results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate various statistics from optimization results.

    Args:
        results: Dictionary containing optimization results

    Returns:
        Dictionary with calculated statistics
    """
    stats = {}

    # Simple prompt stats
    if "simple" in results and results["simple"]["success"]:
        stats["simple_perplexity"] = results["simple"]["perplexity"]
        stats["simple_loss"] = results["simple"]["best_loss"]
        stats["simple_time"] = results["simple"]["optimization_time"]

    # Table stats
    table_perplexities = []
    table_losses = []
    table_times = []

    for key, result in results.items():
        if key.startswith("table_") and result["success"]:
            table_perplexities.append(result["perplexity"])
            table_losses.append(result["best_loss"])
            table_times.append(result["optimization_time"])

    if table_perplexities:
        stats["table_perplexity_mean"] = np.mean(table_perplexities)
        stats["table_perplexity_std"] = np.std(table_perplexities)
        stats["table_perplexity_min"] = np.min(table_perplexities)
        stats["table_perplexity_max"] = np.max(table_perplexities)
        stats["table_loss_mean"] = np.mean(table_losses)
        stats["table_time_mean"] = np.mean(table_times)
        stats["successful_table_optimizations"] = len(table_perplexities)

    return stats


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")