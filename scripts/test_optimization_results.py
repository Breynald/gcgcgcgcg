"""
Test script to evaluate optimization results by checking if model outputs match target.
Tests each optimized prompt multiple times to calculate success rate.
Extended to test false positive rates on multiple additional models.
"""

import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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


def parse_table_size(table_key: str):
    """Extract rows and cols from table key like 'table_5x5'."""
    if table_key.startswith('table_'):
        # Extract numbers from table_XxX format (e.g., table_5x5)
        match = re.match(r'table_(\d+)x(\d+)', table_key)
        if match:
            rows = int(match.group(1))
            cols = int(match.group(2))
            return rows, cols
    return None, None


def get_table_category(rows: int, cols: int):
    """Categorize tables by size for better visualization."""
    total_cells = rows * cols
    if total_cells <= 4:
        return "Small (≤4)"
    elif total_cells <= 9:
        return "Medium (5-9)"
    elif total_cells <= 16:
        return "Large (10-16)"
    else:
        return "X-Large (>16)"


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer for testing."""
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # Set pad token properly to avoid warnings
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token ({tokenizer.eos_token}) for {model_path}")
        else:
            # Fallback: add a new pad token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            print(f"Added new pad token [PAD] for {model_path}")

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 1, device: str = "cuda"):
    """Generate model response for given prompt."""
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]

    try:
        # Apply chat template
        if tokenizer.chat_template is not None:
            formatted_inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            input_ids = formatted_inputs.to(device)
        else:
            # Fallback for models without chat template
            formatted_inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = formatted_inputs.input_ids.to(device)

        # Get attention mask
        attention_mask = formatted_inputs.attention_mask.to(device) if hasattr(formatted_inputs, 'attention_mask') else None

        # Generate response with proper parameters (deterministic for consistency)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False  # Use deterministic generation for success rate testing
            )

        # Extract generated part
        prompt_length = input_ids.shape[-1]
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


def test_prompt_success_rate(model, tokenizer, optimized_prompt: str,
                           target: str, num_samples: int = 8, device: str = "cuda"):
    """Test success rate of a single optimized prompt."""
    # Calculate target token count once at the beginning
    target_token_ids = tokenizer.encode(target, add_special_tokens=False)
    max_new_tokens = max(len(target_token_ids), 1)

    successes = 0
    responses = []

    for i in range(num_samples):
        response = generate_response(model, tokenizer, optimized_prompt, max_new_tokens, device=device)
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


def test_all_optimizations(results: dict, model, tokenizer, target: str,
                         num_samples: int = 8, device: str = "cuda", model_name: str = "target"):
    """Test all optimized prompts from results."""
    test_results = {}

    # Test simple prompt
    if "simple" in results and results["simple"]["success"]:
        print(f"\n{'='*50}")
        print(f"TESTING SIMPLE PROMPT - {model_name.upper()} MODEL")
        print(f"{'='*50}")

        prompt = results["simple"]["optimized_prompt"]
        print(f"Prompt: {prompt[:100]}...")

        success_rate, responses = test_prompt_success_rate(
            model, tokenizer, prompt, target, num_samples, device
        )

        test_results["simple"] = {
            "success_rate": success_rate,
            "responses": responses,
            "prompt": prompt
        }

        print(f"Success rate: {success_rate:.2%} ({sum(r['success'] for r in responses)}/{len(responses)})")

    # Test table prompts
    table_results = {}
    successful_table_tests = 0
    total_table_tests = 0

    print(f"\n{'='*50}")
    print(f"TESTING TABLE PROMPTS - {model_name.upper()} MODEL")
    print(f"{'='*50}")

    for key, result in results.items():
        if key.startswith("table_") and result["success"]:
            total_table_tests += 1
            table_name = key.replace("table_", "")

            print(f"\nTesting {table_name}...")
            prompt = result["optimized_prompt"]

            # Show truncated prompt
            prompt_preview = prompt[:150] + "..." if len(prompt) > 150 else prompt
            print(f"Prompt preview: {prompt_preview}")

            success_rate, responses = test_prompt_success_rate(
                model, tokenizer, prompt, target, num_samples, device
            )

            if success_rate > 0:
                successful_table_tests += 1

            table_results[key] = {
                "success_rate": success_rate,
                "responses": responses,
                "prompt": prompt,
                "table_size": table_name
            }

            print(f"Success rate: {success_rate:.2%} ({sum(r['success'] for r in responses)}/{len(responses)})")

            # Show response details
            exact_matches = [r for r in responses if r['exact_match']]
            case_insensitive_matches = [r for r in responses if r['success'] and not r['exact_match']]

            print(f"  Exact matches: {len(exact_matches)}/{len(responses)}")
            if case_insensitive_matches:
                print(f"  Case-insensitive matches: {len(case_insensitive_matches)}/{len(responses)}")

            # Show example responses
            if exact_matches:
                example = exact_matches[0]
                print(f"  Example exact match: '{example['extracted_word']}' == '{example['expected']}' ✓")
            elif responses:
                example = responses[0]
                print(f"  Example response: '{example['extracted_word']}' (expected: '{example['expected']}')")

    test_results["tables"] = table_results

    # Summary statistics
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY - {model_name.upper()} MODEL")
    print(f"{'='*50}")

    if "simple" in test_results:
        print(f"Simple prompt success rate: {test_results['simple']['success_rate']:.2%}")

    if table_results:
        success_rates = [r['success_rate'] for r in table_results.values()]
        print(f"Table prompts - Average success rate: {np.mean(success_rates):.2%}")
        print(f"Table prompts - Best success rate: {np.max(success_rates):.2%}")
        print(f"Table prompts - Worst success rate: {np.min(success_rates):.2%}")
        print(f"Successful table tests (>0%): {successful_table_tests}/{total_table_tests}")

        # Find best performing table
        best_table = max(table_results.items(), key=lambda x: x[1]['success_rate'])
        worst_table = min(table_results.items(), key=lambda x: x[1]['success_rate'])
        print(f"Best table: {best_table[0]} ({best_table[1]['success_rate']:.2%})")
        print(f"Worst table: {worst_table[0]} ({worst_table[1]['success_rate']:.2%})")

    return test_results


def test_multiple_models(results: dict, model_configs: list, target: str,
                        num_samples: int = 8, device: str = "cuda"):
    """Test optimized prompts on multiple models."""
    all_results = {}

    for config in model_configs:
        model_path = config["path"]
        model_name = config["name"]
        is_target = config.get("is_target", False)

        print(f"\n{'#'*80}")
        if is_target:
            print(f"# TESTING TARGET MODEL: {model_name} ({model_path})")
        else:
            print(f"# TESTING FALSE POSITIVE MODEL: {model_name} ({model_path})")
        print(f"{'#'*80}")

        # Load model
        try:
            model, tokenizer = load_model_and_tokenizer(model_path, device)
            if model is None or tokenizer is None:
                print(f"Skipping {model_name} due to loading error")
                continue

            # Run tests
            test_results = test_all_optimizations(
                results, model, tokenizer, target, num_samples, device, model_name
            )

            # Store results with model info
            all_results[model_name] = {
                "model_path": model_path,
                "is_target": is_target,
                "test_results": test_results
            }

            # Clean up GPU memory
            del model, tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            all_results[model_name] = {
                "model_path": model_path,
                "is_target": is_target,
                "error": str(e)
            }

    return all_results


def create_table_size_heatmaps(all_results: dict, output_dir: str):
    """Create heatmaps showing success rates and false positive rates by table size."""
    print("  Creating table size heatmaps...")
    os.makedirs(output_dir, exist_ok=True)

    # Collect data for heatmaps
    print("  Collecting table data...")
    table_data = []

    for model_name, results in all_results.items():
        if "test_results" not in results:
            continue

        model_type = "Success Rate" if results.get("is_target", False) else "False Positive Rate"
        test_results = results["test_results"]

        if "tables" in test_results:
            for table_key, table_result in test_results["tables"].items():
                rows, cols = parse_table_size(table_key)
                if rows is not None and cols is not None:
                    table_data.append({
                        "Model": model_name,
                        "Model_Type": model_type,
                        "Rows": rows,
                        "Cols": cols,
                        "Total_Cells": rows * cols,
                        "Table_Size": f"{rows}×{cols}",
                        "Category": get_table_category(rows, cols),
                        "Rate": table_result['success_rate'],
                        "Successes": sum(1 for r in table_result['responses'] if r['success']),
                        "Total_Samples": len(table_result['responses'])
                    })

    if not table_data:
        print("  No table data found for heatmap generation")
        return

    print(f"  Found {len(table_data)} table data entries")
    df = pd.DataFrame(table_data)
    print("  Dataframe created successfully")

    # Set up the plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
    sns.set_palette("RdYlGn")
    print("  Plotting style configured")

    # 1. Combined Success Rate and False Positive Rate Heatmap
    print("  Creating combined success/FP rate heatmap...")
    create_combined_rate_heatmap(df, output_dir)

    # 2. Simple Prompt Comparison Chart
    print("  Creating simple prompt comparison chart...")
    create_simple_prompt_chart(all_results, output_dir)

    print(f"  All visualizations created")


def create_combined_rate_heatmap(df: pd.DataFrame, output_dir: str):
    """Create combined heatmap showing success rates and false positive rates."""
    try:
        # Get unique models
        models = df['Model'].unique()
        n_models = len(models)

        if n_models == 0:
            print("  No model data found")
            return

        # Create subplots - one column per model
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]

        for i, model in enumerate(models):
            model_data = df[df['Model'] == model].copy()

            if model_data.empty:
                continue

            # Determine model type
            model_type = model_data['Model_Type'].iloc[0]

            # Create pivot table
            pivot_data = model_data.pivot_table(
                index='Rows',
                columns='Cols',
                values='Rate',
                aggfunc='mean'
            )

            # Sort for better visualization
            pivot_data = pivot_data.sort_index().sort_index(axis=1)

            # Choose colormap based on model type
            if model_type == 'Success Rate':
                cmap = 'RdYlGn'  # Green = good (high success rate)
                label = 'Success Rate'
            else:
                cmap = 'RdYlGn_r'  # Green = good (low FP rate)
                label = 'False Positive Rate'

            # Create heatmap
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.1%',
                cmap=cmap,
                vmin=0,
                vmax=1,
                cbar_kws={'label': label},
                ax=axes[i],
                square=True
            )

            # Set title
            axes[i].set_title(f'{model}\n({label})', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Columns', fontsize=10)
            if i == 0:
                axes[i].set_ylabel('Rows', fontsize=10)
            else:
                axes[i].set_ylabel('')

        plt.suptitle('Success Rate and False Positive Rate by Table Size', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save
        heatmap_file = os.path.join(output_dir, "combined_rates_heatmap.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Combined rates heatmap saved: {heatmap_file}")

    except Exception as e:
        print(f"  Error creating combined rates heatmap: {e}")
        import traceback
        traceback.print_exc()


def create_simple_prompt_chart(all_results: dict, output_dir: str):
    """Create bar chart comparing simple prompt performance across models."""
    try:
        # Collect simple prompt data
        simple_data = []

        for model_name, results in all_results.items():
            if "test_results" not in results:
                continue

            model_type = "Success Rate" if results.get("is_target", False) else "False Positive Rate"
            test_results = results["test_results"]

            if "simple" in test_results:
                simple_data.append({
                    "Model": model_name,
                    "Model_Type": model_type,
                    "Rate": test_results["simple"]["success_rate"]
                })

        if not simple_data:
            print("  No simple prompt data found")
            return

        df = pd.DataFrame(simple_data)

        # Create bar chart
        plt.figure(figsize=(10, 6))

        # Separate by model type for better visualization
        target_models = df[df['Model_Type'] == 'Success Rate']
        fp_models = df[df['Model_Type'] == 'False Positive Rate']

        x = range(len(df))
        colors = ['#2ecc71' if mt == 'Success Rate' else '#e74c3c' for mt in df['Model_Type']]

        bars = plt.bar(x, df['Rate'], color=colors, alpha=0.7, width=0.6)

        # Customize the chart
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.title('Simple Prompt Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, df['Model'], rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        # Add percentage labels on bars
        for bar, rate in zip(bars, df['Rate']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', alpha=0.7, label='Success Rate'),
            Patch(facecolor='#e74c3c', alpha=0.7, label='False Positive Rate')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        # Save
        chart_file = os.path.join(output_dir, "simple_prompt_comparison.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Simple prompt comparison chart saved: {chart_file}")

    except Exception as e:
        print(f"  Error creating simple prompt chart: {e}")
        import traceback
        traceback.print_exc()


def save_multi_model_test_results(all_results: dict, target: str, output_dir: str):
    """Save multi-model test results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed JSON results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(output_dir, f"multi_model_test_results_{timestamp}.json")

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "target": target,
            "multi_model_results": all_results
        }, f, indent=2, ensure_ascii=False)

    # Create comprehensive summary report
    summary_file = os.path.join(output_dir, f"multi_model_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("MULTI-MODEL OPTIMIZATION SUCCESS TEST REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Target: {target}\n\n")

        # Target model results
        f.write("TARGET MODEL RESULTS:\n")
        f.write("-" * 40 + "\n")
        for model_name, results in all_results.items():
            if results.get("is_target", False) and "test_results" in results:
                f.write(f"\n{model_name} ({results['model_path']}):\n")
                test_results = results["test_results"]

                if "simple" in test_results:
                    f.write(f"  Simple prompt: {test_results['simple']['success_rate']:.2%}\n")

                if "tables" in test_results:
                    f.write(f"\n  Table Prompts - Individual Results:\n")
                    f.write("  " + "-" * 35 + "\n")
                    f.write(f"  {'Table Size':<12} {'Success Rate':<12} {'Samples':<8} {'Successes':<10}\n")
                    f.write("  " + "-" * 35 + "\n")

                    # Sort tables by success rate (descending)
                    sorted_tables = sorted(test_results["tables"].items(),
                                         key=lambda x: x[1]['success_rate'], reverse=True)

                    for table_key, table_result in sorted_tables:
                        table_size = table_result['table_size']
                        success_rate = table_result['success_rate']
                        num_responses = len(table_result['responses'])
                        successes = sum(1 for r in table_result['responses'] if r['success'])

                        f.write(f"  {table_size:<12} {success_rate:<12.2%} {num_responses:<8} {successes:<10}\n")

                    # Summary statistics
                    table_rates = [r['success_rate'] for r in test_results["tables"].values()]
                    if table_rates:
                        f.write("  " + "-" * 35 + "\n")
                        f.write(f"  Average Success Rate: {np.mean(table_rates):.2%}\n")
                        f.write(f"  Best Success Rate: {np.max(table_rates):.2%}\n")
                        f.write(f"  Worst Success Rate: {np.min(table_rates):.2%}\n")

                        # Count successful tables
                        successful_tables = sum(1 for rate in table_rates if rate > 0)
                        total_tables = len(table_rates)
                        f.write(f"  Successful Tables (>0%): {successful_tables}/{total_tables}\n")

        # False positive model results
        f.write(f"\n\nFALSE POSITIVE MODEL RESULTS:\n")
        f.write("-" * 40 + "\n")
        for model_name, results in all_results.items():
            if not results.get("is_target", False) and "test_results" in results:
                f.write(f"\n{model_name} ({results['model_path']}):\n")
                test_results = results["test_results"]

                if "simple" in test_results:
                    f.write(f"  Simple prompt: {test_results['simple']['success_rate']:.2%} (False Positive Rate)\n")

                if "tables" in test_results:
                    f.write(f"\n  Table Prompts - Individual Results (False Positive Rates):\n")
                    f.write("  " + "-" * 35 + "\n")
                    f.write(f"  {'Table Size':<12} {'FP Rate':<12} {'Samples':<8} {'False Pos':<10}\n")
                    f.write("  " + "-" * 35 + "\n")

                    # Sort tables by false positive rate (ascending for FP models)
                    sorted_tables = sorted(test_results["tables"].items(),
                                         key=lambda x: x[1]['success_rate'])

                    for table_key, table_result in sorted_tables:
                        table_size = table_result['table_size']
                        fp_rate = table_result['success_rate']
                        num_responses = len(table_result['responses'])
                        false_positives = sum(1 for r in table_result['responses'] if r['success'])

                        f.write(f"  {table_size:<12} {fp_rate:<12.2%} {num_responses:<8} {false_positives:<10}\n")

                    # Summary statistics
                    table_rates = [r['success_rate'] for r in test_results["tables"].values()]
                    if table_rates:
                        f.write("  " + "-" * 35 + "\n")
                        f.write(f"  Average FP Rate: {np.mean(table_rates):.2%}\n")
                        f.write(f"  Best FP Rate: {np.max(table_rates):.2%}\n")
                        f.write(f"  Worst FP Rate: {np.min(table_rates):.2%}\n")

                        # Count tables with false positives
                        fp_tables = sum(1 for rate in table_rates if rate > 0)
                        total_tables = len(table_rates)
                        f.write(f"  Tables with FPs: {fp_tables}/{total_tables}\n")

        # Summary comparison table
        f.write(f"\n\nDETAILED COMPARISON TABLE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<15} {'Type':<12} {'Simple':<10} {'Table Avg':<12} {'Best Table':<12} {'Worst Table':<12}\n")
        f.write("-" * 80 + "\n")

        for model_name, results in all_results.items():
            if "test_results" in results:
                test_results = results["test_results"]
                model_type = "Target" if results.get("is_target", False) else "False Positive"

                simple_rate = "N/A"
                if "simple" in test_results:
                    simple_rate = f"{test_results['simple']['success_rate']:.2%}"

                table_avg = "N/A"
                best_table = "N/A"
                worst_table = "N/A"
                if "tables" in test_results:
                    table_rates = [r['success_rate'] for r in test_results["tables"].values()]
                    if table_rates:
                        table_avg = f"{np.mean(table_rates):.2%}"
                        best_table = f"{np.max(table_rates):.2%}"
                        worst_table = f"{np.min(table_rates):.2%}"

                f.write(f"{model_name:<15} {model_type:<12} {simple_rate:<10} {table_avg:<12} {best_table:<12} {worst_table:<12}\n")

        # Find best performing prompt across all models
        f.write(f"\n\nBEST PERFORMING PROMPTS:\n")
        f.write("-" * 40 + "\n")

        # Collect all table results
        all_table_results = []
        for model_name, results in all_results.items():
            if "test_results" in results and "tables" in results["test_results"]:
                model_type = "Target" if results.get("is_target", False) else "False Positive"
                for table_key, table_result in results["test_results"]["tables"].items():
                    all_table_results.append({
                        "model": model_name,
                        "model_type": model_type,
                        "table_size": table_result['table_size'],
                        "success_rate": table_result['success_rate']
                    })

        if all_table_results:
            # Sort by success rate
            if any(r["model_type"] == "Target" for r in all_table_results):
                # For target models, highest success rate is best
                sorted_results = sorted(all_table_results, key=lambda x: x["success_rate"], reverse=True)
                f.write("Top 5 Overall Success Rates:\n")
            else:
                # For false positive models, lowest success rate (FP rate) is best
                sorted_results = sorted(all_table_results, key=lambda x: x["success_rate"])
                f.write("Top 5 Lowest False Positive Rates:\n")

            for i, result in enumerate(sorted_results[:5]):
                rate_type = "Success Rate" if result["model_type"] == "Target" else "FP Rate"
                f.write(f"  {i+1}. {result['model']} - {result['table_size']}: {result['success_rate']:.2%} ({rate_type})\n")

    # Generate table size heatmaps
    try:
        print(f"\n{'='*60}")
        print("GENERATING TABLE SIZE HEATMAPS")
        print(f"{'='*60}")
        print("Starting heatmap generation...")
        create_table_size_heatmaps(all_results, output_dir)
        print("Heatmap generation completed successfully!")
    except Exception as e:
        print(f"Warning: Could not generate heatmaps: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure all matplotlib resources are cleaned up
        try:
            plt.close('all')
            print("All matplotlib figures closed.")
        except:
            pass

    print(f"Multi-model test results saved:")
    print(f"  Detailed: {json_file}")
    print(f"  Summary: {summary_file}")
    if os.path.exists(output_dir):
        heatmap_files = [f for f in os.listdir(output_dir) if f.endswith('_heatmap.png')]
        if heatmap_files:
            print(f"  Heatmaps: {len(heatmap_files)} files generated in {output_dir}/")
        else:
            print(f"  Heatmaps: No heatmap files generated")

    return json_file, summary_file


def main():
    parser = argparse.ArgumentParser(description="Test optimization success rates on multiple models")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to optimization results JSON file")

    # Target model (for success rate testing)
    parser.add_argument("--model", type=str,
                       default="/work/models/Qwen/Qwen2.5-1.5B-Instruct",
                       help="Target model path for success rate testing")

    # False positive models (for false positive rate testing)
    parser.add_argument("--fp-models", type=str, nargs='*',
                       default=[],
                       help="Additional model paths for false positive rate testing")

    # Model aliases/names for better reporting
    parser.add_argument("--model-names", type=str, nargs='*',
                       default=[],
                       help="Names for models (must match number of models)")

    parser.add_argument("--target", type=str, default="Copyright",
                       help="Target word to check for")
    parser.add_argument("--samples", type=int, default=8,
                       help="Number of samples per prompt")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--output", type=str, default="test_results",
                       help="Output directory for test results")

    # Legacy mode: single model testing
    parser.add_argument("--single-model", action="store_true",
                       help="Use legacy single model mode")

    args = parser.parse_args()

    print("=" * 80)
    if args.single_model:
        print("SINGLE MODEL OPTIMIZATION SUCCESS TESTING")
    else:
        print("MULTI-MODEL OPTIMIZATION SUCCESS TESTING")
    print("=" * 80)
    print(f"Results file: {args.results}")
    print(f"Target model: {args.model}")
    if not args.single_model and args.fp_models:
        print(f"False positive models: {args.fp_models}")
    print(f"Target: {args.target}")
    print(f"Samples per prompt: {args.samples}")
    print(f"Device: {args.device}")

    # Load optimization results
    if not os.path.exists(args.results):
        print(f"Error: Results file {args.results} not found")
        return

    results = load_optimization_results(args.results)
    print(f"Loaded {len(results)} optimization results")

    if args.single_model:
        # Legacy single model mode
        print(f"\nRunning in single model mode...")

        # Load model
        try:
            model, tokenizer = load_model_and_tokenizer(args.model, args.device)
            if model is None or tokenizer is None:
                print(f"Error loading model: {args.model}")
                return
            print(f"Model loaded successfully on {args.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # Run tests
        try:
            test_results = test_all_optimizations(
                results, model, tokenizer, args.target, args.samples, args.device, "target"
            )

            # Save results using legacy function
            save_test_results(test_results, args.target, args.model, args.output)

            print(f"\n{'='*60}")
            print("SINGLE MODEL TESTING COMPLETED")
            print(f"{'='*60}")

        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Multi model mode
        print(f"\nRunning in multi-model mode...")

        # Build model configurations
        model_configs = []

        # Add target model
        target_name = args.model_names[0] if args.model_names and len(args.model_names) > 0 else "Target"
        model_configs.append({
            "path": args.model,
            "name": target_name,
            "is_target": True
        })

        # Add false positive models
        for i, fp_model in enumerate(args.fp_models):
            fp_name = None
            if args.model_names and len(args.model_names) > i + 1:
                fp_name = args.model_names[i + 1]
            else:
                fp_name = f"FP_Model_{i+1}"

            model_configs.append({
                "path": fp_model,
                "name": fp_name,
                "is_target": False
            })

        print(f"\nTesting {len(model_configs)} models:")
        for config in model_configs:
            model_type = "Target" if config["is_target"] else "False Positive"
            print(f"  {config['name']}: {config['path']} ({model_type})")

        # Run multi-model tests
        try:
            all_results = test_multiple_models(
                results, model_configs, args.target, args.samples, args.device
            )

            # Save results
            json_file, summary_file = save_multi_model_test_results(
                all_results, args.target, args.output
            )

            print(f"\n{'#'*80}")
            print("MULTI-MODEL TESTING COMPLETED")
            print(f"{'#'*80}")

            # Print final summary
            print(f"\nFINAL SUMMARY:")
            print(f"Target: {args.target}")
            print(f"Models tested: {len(model_configs)}")

            target_results = [r for r in all_results.values() if r.get("is_target", False) and "test_results" in r]
            fp_results = [r for r in all_results.values() if not r.get("is_target", False) and "test_results" in r]

            print(f"Target models tested successfully: {len(target_results)}")
            print(f"False positive models tested successfully: {len(fp_results)}")

        except Exception as e:
            print(f"Error during multi-model testing: {e}")
            import traceback
            traceback.print_exc()


def create_single_model_visualizations(test_results: dict, model_path: str, output_dir: str):
    """Create visualizations for single model results."""
    try:
        # Extract model name from path
        model_name = model_path.split('/')[-1] if '/' in model_path else model_path

        # 1. Simple prompt performance if available
        if "simple" in test_results:
            plt.figure(figsize=(8, 6))

            simple_rate = test_results["simple"]["success_rate"]
            color = '#2ecc71' if simple_rate > 0.5 else '#e74c3c'

            plt.bar(['Simple Prompt'], [simple_rate], color=color, alpha=0.7, width=0.4)
            plt.ylabel('Success Rate', fontsize=12)
            plt.title(f'Simple Prompt Performance - {model_name}', fontsize=14, fontweight='bold')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)

            # Add percentage label
            plt.text(0, simple_rate + 0.02, f'{simple_rate:.1%}', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

            plt.tight_layout()

            simple_file = os.path.join(output_dir, "single_model_simple.png")
            plt.savefig(simple_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Single model simple prompt chart saved: {simple_file}")

        # 2. Table heatmap if available
        if "tables" not in test_results:
            print("No table results found for heatmap generation")
            return

        # Collect table data
        table_data = []
        for table_key, table_result in test_results["tables"].items():
            rows, cols = parse_table_size(table_key)
            if rows is not None and cols is not None:
                table_data.append({
                    "Rows": rows,
                    "Cols": cols,
                    "Table_Size": f"{rows}×{cols}",
                    "Success_Rate": table_result['success_rate']
                })

        if not table_data:
            print("No valid table data found for heatmap generation")
            return

        df = pd.DataFrame(table_data)

        # Create pivot table
        pivot_data = df.pivot_table(
            index='Rows',
            columns='Cols',
            values='Success_Rate',
            aggfunc='mean'
        )

        # Sort for better visualization
        pivot_data = pivot_data.sort_index().sort_index(axis=1)

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Success Rate'},
            square=True
        )

        plt.title(f'Success Rate by Table Size - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Columns', fontsize=12)
        plt.ylabel('Rows', fontsize=12)
        plt.tight_layout()

        # Save
        heatmap_file = os.path.join(output_dir, "single_model_tables.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Single model tables heatmap saved: {heatmap_file}")

    except Exception as e:
        print(f"Error creating single model visualizations: {e}")
        import traceback
        traceback.print_exc()


def save_test_results(test_results: dict, target: str, model_path: str, output_dir: str):
    """Save test results to files (legacy single model mode)."""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed JSON results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(output_dir, f"test_results_{timestamp}.json")

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "target": target,
            "model_path": model_path,
            "test_results": test_results
        }, f, indent=2, ensure_ascii=False)

    # Create detailed summary report
    summary_file = os.path.join(output_dir, f"test_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("OPTIMIZATION SUCCESS TEST REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Target: {target}\n")
        f.write(f"Model: {model_path}\n\n")

        if "simple" in test_results:
            f.write(f"SIMPLE PROMPT: {test_results['simple']['success_rate']:.2%} success rate\n\n")

        if "tables" in test_results:
            f.write("TABLE PROMPTS - DETAILED RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Table Size':<12} {'Success Rate':<12} {'Samples':<8} {'Successes':<10}\n")
            f.write("-" * 40 + "\n")

            # Sort tables by success rate (descending)
            sorted_tables = sorted(test_results["tables"].items(),
                                 key=lambda x: x[1]['success_rate'], reverse=True)

            for table_key, table_result in sorted_tables:
                table_size = table_result['table_size']
                success_rate = table_result['success_rate']
                num_responses = len(table_result['responses'])
                successes = sum(1 for r in table_result['responses'] if r['success'])

                f.write(f"{table_size:<12} {success_rate:<12.2%} {num_responses:<8} {successes:<10}\n")

            # Summary statistics
            success_rates = [r['success_rate'] for r in test_results["tables"].values()]
            if success_rates:
                f.write("-" * 40 + "\n")
                f.write(f"Average Success Rate: {np.mean(success_rates):.2%}\n")
                f.write(f"Best Success Rate: {np.max(success_rates):.2%}\n")
                f.write(f"Worst Success Rate: {np.min(success_rates):.2%}\n")

                # Count successful tables
                successful_tables = sum(1 for rate in success_rates if rate > 0)
                total_tables = len(success_rates)
                f.write(f"Successful Tables (>0%): {successful_tables}/{total_tables}\n")

                # Show best performing tables
                f.write(f"\nTop 5 Best Performing Tables:\n")
                for i, (table_key, table_result) in enumerate(sorted_tables[:5]):
                    table_size = table_result['table_size']
                    success_rate = table_result['success_rate']
                    f.write(f"  {i+1}. {table_size}: {success_rate:.2%}\n")

    # Generate visualizations for single model
    try:
        print(f"\n{'='*60}")
        print("GENERATING SINGLE MODEL VISUALIZATIONS")
        print(f"{'='*60}")
        create_single_model_visualizations(test_results, model_path, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure all matplotlib resources are cleaned up
        try:
            plt.close('all')
        except:
            pass

    print(f"Test results saved:")
    print(f"  Detailed: {json_file}")
    print(f"  Summary: {summary_file}")
    if os.path.exists(output_dir):
        viz_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        if viz_files:
            print(f"  Visualizations: {len(viz_files)} files generated in {output_dir}/")
        else:
            print(f"  Visualizations: No files generated")


if __name__ == "__main__":
    main()