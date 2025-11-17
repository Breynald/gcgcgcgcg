"""
Test script to evaluate optimization results under different sampling parameters.
Tests the impact of temperature, top_p, and top_k on success rates.
"""

import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path to import nanogcg tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_optimization_results(results_file: str):
    """Load optimization results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer for testing."""
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response_with_sampling(model, tokenizer, prompt: str,
                                  temperature: float = 1.0,
                                  top_p: float = 0.9,
                                  top_k: int = 50,
                                  max_new_tokens: int = 1,
                                  device: str = "cuda"):
    """Generate model response with specified sampling parameters."""
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]

    try:
        # Apply chat template
        if tokenizer.chat_template is not None:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
        else:
            # Fallback for models without chat template
            formatted_prompt = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Generate response with sampling
        with torch.no_grad():
            outputs = model.generate(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # Extract generated part
        if hasattr(formatted_prompt, 'shape') and len(formatted_prompt.shape) == 2:
            prompt_length = formatted_prompt.shape[1]
        else:
            prompt_length = len(formatted_prompt)

        generated_ids = outputs[0][prompt_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    except Exception as e:
        return f"ERROR: {str(e)}"


def extract_target_word(response: str, target: str):
    """Extract and check for target word using prefix exact match."""
    # Clean up response
    response = response.strip()

    # If response is empty or just whitespace
    if not response:
        return False, ""

    # Get the entire response
    cleaned_response = response.strip()

    # Check for prefix exact match (case-sensitive only)
    is_match = cleaned_response.startswith(target)

    return is_match, cleaned_response


def test_prompt_with_sampling(model, tokenizer, prompt: str, target: str,
                            temperature: float, top_p: float, top_k: int,
                            num_samples: int = 8, device: str = "cuda"):
    """Test success rate of a prompt with specific sampling parameters."""
    # Calculate target token count once at the beginning
    target_token_ids = tokenizer.encode(target, add_special_tokens=False)
    max_new_tokens = max(len(target_token_ids), 1)

    successes = 0
    responses = []

    for i in range(num_samples):
        response = generate_response_with_sampling(
            model, tokenizer, prompt, temperature, top_p, top_k, max_new_tokens, device=device
        )
        is_match, extracted_word = extract_target_word(response, target)

        successes += 1 if is_match else 0
        responses.append({
            "sample": i + 1,
            "response": response,
            "extracted_word": extracted_word,
            "success": is_match,
            "expected": target,
            "exact_match": extracted_word == target
        })

    success_rate = successes / num_samples
    return success_rate, responses


def test_temperature_effects(model, tokenizer, results: dict, target: str,
                           temperatures: list, num_samples: int = 8, device: str = "cuda"):
    """Test effect of different temperatures on success rates."""
    results_data = []

    print(f"\n{'='*60}")
    print("TESTING TEMPERATURE EFFECTS")
    print(f"{'='*60}")

    for temp in temperatures:
        print(f"\nTesting Temperature = {temp}")

        # Test simple prompt if available
        if "simple" in results and results["simple"]["success"]:
            prompt = results["simple"]["optimized_prompt"]
            success_rate, _ = test_prompt_with_sampling(
                model, tokenizer, prompt, target, temp, 0.9, 50, num_samples, device
            )

            results_data.append({
                "prompt_type": "simple",
                "temperature": temp,
                "top_p": 0.9,
                "top_k": 50,
                "success_rate": success_rate
            })

            print(f"  Simple prompt: {success_rate:.2%}")

        # Test table prompts
        for key, result in results.items():
            if key.startswith("table_") and result["success"]:
                table_name = key.replace("table_", "")
                prompt = result["optimized_prompt"]

                success_rate, _ = test_prompt_with_sampling(
                    model, tokenizer, prompt, target, temp, 0.9, 50, num_samples, device
                )

                results_data.append({
                    "prompt_type": f"table_{table_name}",
                    "temperature": temp,
                    "top_p": 0.9,
                    "top_k": 50,
                    "success_rate": success_rate
                })

                print(f"  Table {table_name}: {success_rate:.2%}")

    return results_data


def test_top_p_effects(model, tokenizer, results: dict, target: str,
                      top_p_values: list, num_samples: int = 8, device: str = "cuda"):
    """Test effect of different top_p values on success rates."""
    results_data = []

    print(f"\n{'='*60}")
    print("TESTING TOP_P EFFECTS")
    print(f"{'='*60}")

    for top_p in top_p_values:
        print(f"\nTesting Top-p = {top_p}")

        # Test simple prompt if available
        if "simple" in results and results["simple"]["success"]:
            prompt = results["simple"]["optimized_prompt"]
            success_rate, _ = test_prompt_with_sampling(
                model, tokenizer, prompt, target, 1.0, top_p, 50, num_samples, device
            )

            results_data.append({
                "prompt_type": "simple",
                "temperature": 1.0,
                "top_p": top_p,
                "top_k": 50,
                "success_rate": success_rate
            })

            print(f"  Simple prompt: {success_rate:.2%}")

        # Test table prompts
        for key, result in results.items():
            if key.startswith("table_") and result["success"]:
                table_name = key.replace("table_", "")
                prompt = result["optimized_prompt"]

                success_rate, _ = test_prompt_with_sampling(
                    model, tokenizer, prompt, target, 1.0, top_p, 50, num_samples, device
                )

                results_data.append({
                    "prompt_type": f"table_{table_name}",
                    "temperature": 1.0,
                    "top_p": top_p,
                    "top_k": 50,
                    "success_rate": success_rate
                })

                print(f"  Table {table_name}: {success_rate:.2%}")

    return results_data


def test_top_k_effects(model, tokenizer, results: dict, target: str,
                      top_k_values: list, num_samples: int = 8, device: str = "cuda"):
    """Test effect of different top_k values on success rates."""
    results_data = []

    print(f"\n{'='*60}")
    print("TESTING TOP_K EFFECTS")
    print(f"{'='*60}")

    for top_k in top_k_values:
        print(f"\nTesting Top-k = {top_k}")

        # Test simple prompt if available
        if "simple" in results and results["simple"]["success"]:
            prompt = results["simple"]["optimized_prompt"]
            success_rate, _ = test_prompt_with_sampling(
                model, tokenizer, prompt, target, 1.0, 0.9, top_k, num_samples, device
            )

            results_data.append({
                "prompt_type": "simple",
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": top_k,
                "success_rate": success_rate
            })

            print(f"  Simple prompt: {success_rate:.2%}")

        # Test table prompts
        for key, result in results.items():
            if key.startswith("table_") and result["success"]:
                table_name = key.replace("table_", "")
                prompt = result["optimized_prompt"]

                success_rate, _ = test_prompt_with_sampling(
                    model, tokenizer, prompt, target, 1.0, 0.9, top_k, num_samples, device
                )

                results_data.append({
                    "prompt_type": f"table_{table_name}",
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "top_k": top_k,
                    "success_rate": success_rate
                })

                print(f"  Table {table_name}: {success_rate:.2%}")

    return results_data


def create_visualizations(all_results: list, output_dir: str):
    """Create visualizations of sampling parameter effects."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(all_results)

    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Temperature effects plot
    temp_data = df[df['temperature'] != 1.0].copy()
    if not temp_data.empty:
        plt.figure(figsize=(12, 8))

        # Group by prompt type and temperature
        temp_grouped = temp_data.groupby(['prompt_type', 'temperature'])['success_rate'].mean().reset_index()

        # Separate simple and table prompts
        simple_data = temp_grouped[temp_grouped['prompt_type'] == 'simple']
        table_data = temp_grouped[temp_grouped['prompt_type'] != 'simple']

        # Plot simple prompt
        if not simple_data.empty:
            plt.subplot(2, 2, 1)
            plt.plot(simple_data['temperature'], simple_data['success_rate'],
                    marker='o', linewidth=2, markersize=8, label='Simple Prompt')
            plt.xlabel('Temperature')
            plt.ylabel('Success Rate')
            plt.title('Temperature Effect - Simple Prompt')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

        # Plot table prompts
        if not table_data.empty:
            plt.subplot(2, 2, 2)
            for table_type in table_data['prompt_type'].unique():
                table_subset = table_data[table_data['prompt_type'] == table_type]
                plt.plot(table_subset['temperature'], table_subset['success_rate'],
                        marker='o', linewidth=2, markersize=6, label=table_type)
            plt.xlabel('Temperature')
            plt.ylabel('Success Rate')
            plt.title('Temperature Effect - Table Prompts')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

        plt.tight_layout()
        temp_plot_file = os.path.join(output_dir, "temperature_effects.png")
        plt.savefig(temp_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Temperature plot saved: {temp_plot_file}")

    # Top-p effects plot
    topp_data = df[df['top_p'] != 0.9].copy()
    if not topp_data.empty:
        plt.figure(figsize=(12, 8))

        topp_grouped = topp_data.groupby(['prompt_type', 'top_p'])['success_rate'].mean().reset_index()

        simple_data = topp_grouped[topp_grouped['prompt_type'] == 'simple']
        table_data = topp_grouped[topp_grouped['prompt_type'] != 'simple']

        # Plot simple prompt
        if not simple_data.empty:
            plt.subplot(2, 2, 1)
            plt.plot(simple_data['top_p'], simple_data['success_rate'],
                    marker='s', linewidth=2, markersize=8, label='Simple Prompt')
            plt.xlabel('Top-p')
            plt.ylabel('Success Rate')
            plt.title('Top-p Effect - Simple Prompt')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

        # Plot table prompts
        if not table_data.empty:
            plt.subplot(2, 2, 2)
            for table_type in table_data['prompt_type'].unique():
                table_subset = table_data[table_data['prompt_type'] == table_type]
                plt.plot(table_subset['top_p'], table_subset['success_rate'],
                        marker='s', linewidth=2, markersize=6, label=table_type)
            plt.xlabel('Top-p')
            plt.ylabel('Success Rate')
            plt.title('Top-p Effect - Table Prompts')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

        plt.tight_layout()
        topp_plot_file = os.path.join(output_dir, "topp_effects.png")
        plt.savefig(topp_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top-p plot saved: {topp_plot_file}")

    # Top-k effects plot
    topk_data = df[df['top_k'] != 50].copy()
    if not topk_data.empty:
        plt.figure(figsize=(12, 8))

        topk_grouped = topk_data.groupby(['prompt_type', 'top_k'])['success_rate'].mean().reset_index()

        simple_data = topk_grouped[topk_grouped['prompt_type'] == 'simple']
        table_data = topk_grouped[topk_grouped['prompt_type'] != 'simple']

        # Plot simple prompt
        if not simple_data.empty:
            plt.subplot(2, 2, 1)
            plt.plot(simple_data['top_k'], simple_data['success_rate'],
                    marker='^', linewidth=2, markersize=8, label='Simple Prompt')
            plt.xlabel('Top-k')
            plt.ylabel('Success Rate')
            plt.title('Top-k Effect - Simple Prompt')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

        # Plot table prompts
        if not table_data.empty:
            plt.subplot(2, 2, 2)
            for table_type in table_data['prompt_type'].unique():
                table_subset = topk_data[topk_data['prompt_type'] == table_type]
                topk_subset = topk_grouped[topk_grouped['prompt_type'] == table_type]
                plt.plot(topk_subset['top_k'], topk_subset['success_rate'],
                        marker='^', linewidth=2, markersize=6, label=table_type)
            plt.xlabel('Top-k')
            plt.ylabel('Success Rate')
            plt.title('Top-k Effect - Table Prompts')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

        plt.tight_layout()
        topk_plot_file = os.path.join(output_dir, "topk_effects.png")
        plt.savefig(topk_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top-k plot saved: {topk_plot_file}")

    # Create better heatmaps
    if len(df) > 0:
        create_meaningful_heatmaps(df, output_dir)


def create_meaningful_heatmaps(df, output_dir):
    """Create meaningful heatmaps for sampling parameters."""

    # 1. Temperature vs Prompt Type heatmap
    temp_data = df[df['temperature'] != 1.0].copy()
    if not temp_data.empty and len(temp_data['temperature'].unique()) > 1:
        plt.figure(figsize=(14, 8))

        # Create pivot table: temperature vs prompt types
        temp_pivot = temp_data.pivot_table(
            index='temperature',
            columns='prompt_type',
            values='success_rate',
            aggfunc='mean'
        )

        # Sort by temperature
        temp_pivot = temp_pivot.sort_index()

        sns.heatmap(temp_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                   vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'})
        plt.title('Success Rate by Temperature and Prompt Type')
        plt.xlabel('Prompt Type')
        plt.ylabel('Temperature')

        temp_heatmap_file = os.path.join(output_dir, "temperature_prompt_heatmap.png")
        plt.savefig(temp_heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Temperature-Prompt heatmap saved: {temp_heatmap_file}")

    # 2. Top-p vs Prompt Type heatmap
    topp_data = df[df['top_p'] != 0.9].copy()
    if not topp_data.empty and len(topp_data['top_p'].unique()) > 1:
        plt.figure(figsize=(14, 8))

        # Create pivot table: top_p vs prompt types
        topp_pivot = topp_data.pivot_table(
            index='top_p',
            columns='prompt_type',
            values='success_rate',
            aggfunc='mean'
        )

        # Sort by top_p
        topp_pivot = topp_pivot.sort_index()

        sns.heatmap(topp_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                   vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'})
        plt.title('Success Rate by Top-p and Prompt Type')
        plt.xlabel('Prompt Type')
        plt.ylabel('Top-p')

        topp_heatmap_file = os.path.join(output_dir, "topp_prompt_heatmap.png")
        plt.savefig(topp_heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top-p-Prompt heatmap saved: {topp_heatmap_file}")

    # 3. Top-k vs Prompt Type heatmap
    topk_data = df[df['top_k'] != 50].copy()
    if not topk_data.empty and len(topk_data['top_k'].unique()) > 1:
        plt.figure(figsize=(14, 8))

        # Create pivot table: top_k vs prompt types
        topk_pivot = topk_data.pivot_table(
            index='top_k',
            columns='prompt_type',
            values='success_rate',
            aggfunc='mean'
        )

        # Sort by top_k
        topk_pivot = topk_pivot.sort_index()

        sns.heatmap(topk_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                   vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'})
        plt.title('Success Rate by Top-k and Prompt Type')
        plt.xlabel('Prompt Type')
        plt.ylabel('Top-k')

        topk_heatmap_file = os.path.join(output_dir, "topk_prompt_heatmap.png")
        plt.savefig(topk_heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top-k-Prompt heatmap saved: {topk_heatmap_file}")

    

def save_sampling_results(all_results: list, target: str, model_path: str, output_dir: str):
    """Save sampling test results to files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed JSON results
    json_file = os.path.join(output_dir, f"sampling_results_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "target": target,
            "model_path": model_path,
            "sampling_results": all_results
        }, f, indent=2, ensure_ascii=False)

    # Create summary CSV
    csv_file = os.path.join(output_dir, f"sampling_summary_{timestamp}.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False)

    print(f"Sampling results saved:")
    print(f"  Detailed: {json_file}")
    print(f"  Summary: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Test optimization success rates with sampling")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to optimization results JSON file")
    parser.add_argument("--model", type=str,
                       default="/work/models/Qwen/Qwen2.5-1.5B-Instruct",
                       help="Model path for testing")
    parser.add_argument("--target", type=str, default="Copyright",
                       help="Target word to check for")
    parser.add_argument("--samples", type=int, default=8,
                       help="Number of samples per configuration")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--output", type=str, default="sampling_results",
                       help="Output directory for results")

    # Sampling parameter ranges
    parser.add_argument("--temperatures", type=float, nargs='+',
                       default=[0.1, 0.5, 0.8, 1.0, 1.2, 1.5],
                       help="Temperature values to test")
    parser.add_argument("--topp-values", type=float, nargs='+',
                       default=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
                       help="Top-p values to test")
    parser.add_argument("--topk-values", type=int, nargs='+',
                       default=[1, 10, 50, 100, 200],
                       help="Top-k values to test")

    args = parser.parse_args()

    print("=" * 80)
    print("SAMPLING PARAMETER EFFECT TESTING")
    print("=" * 80)
    print(f"Results file: {args.results}")
    print(f"Model: {args.model}")
    print(f"Target: {args.target}")
    print(f"Samples per configuration: {args.samples}")
    print(f"Device: {args.device}")
    print(f"Temperature range: {args.temperatures}")
    print(f"Top-p range: {args.topp_values}")
    print(f"Top-k range: {args.topk_values}")

    # Load optimization results
    if not os.path.exists(args.results):
        print(f"Error: Results file {args.results} not found")
        return

    results = load_optimization_results(args.results)
    print(f"Loaded {len(results)} optimization results")

    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model, args.device)
        print(f"Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run sampling tests
    all_results = []

    try:
        # Test temperature effects
        temp_results = test_temperature_effects(
            model, tokenizer, results, args.target, args.temperatures,
            args.samples, args.device
        )
        all_results.extend(temp_results)

        # Test top-p effects
        topp_results = test_top_p_effects(
            model, tokenizer, results, args.target, args.topp_values,
            args.samples, args.device
        )
        all_results.extend(topp_results)

        # Test top-k effects
        topk_results = test_top_k_effects(
            model, tokenizer, results, args.target, args.topk_values,
            args.samples, args.device
        )
        all_results.extend(topk_results)

        # Save results
        save_sampling_results(all_results, args.target, args.model, args.output)

        # Create visualizations
        create_visualizations(all_results, args.output)

        print(f"\n{'='*80}")
        print("SAMPLING TESTING COMPLETED")
        print(f"{'='*80}")

        # Print summary statistics
        df = pd.DataFrame(all_results)
        if len(df) > 0:
            print(f"\nSummary Statistics:")
            print(f"Total configurations tested: {len(df)}")
            print(f"Average success rate: {df['success_rate'].mean():.2%}")
            print(f"Best success rate: {df['success_rate'].max():.2%}")
            print(f"Worst success rate: {df['success_rate'].min():.2%}")

            # Find best configuration
            best_config = df.loc[df['success_rate'].idxmax()]
            print(f"\nBest configuration:")
            print(f"  Prompt type: {best_config['prompt_type']}")
            print(f"  Temperature: {best_config['temperature']}")
            print(f"  Top-p: {best_config['top_p']}")
            print(f"  Top-k: {best_config['top_k']}")
            print(f"  Success rate: {best_config['success_rate']:.2%}")

    except Exception as e:
        print(f"Error during sampling testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()