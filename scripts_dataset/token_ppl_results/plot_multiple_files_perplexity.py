#!/usr/bin/env python3
"""Plot perplexity vs token count for multiple optimized prompt files with different colors."""

import os
import sys
import pandas as pd
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.legend import Legend

def calculate_perplexity(text, model, tokenizer, device="cuda:5"):
    """Calculate perplexity of a given text."""
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return torch.exp(loss).item()
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return float('inf')

def count_tokens(text, tokenizer):
    """Count the number of tokens in a text."""
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

def process_file(file_path, model, tokenizer, device, max_samples=1000):
    """Process a single CSV file and return token counts and perplexities."""
    print(f"\nProcessing {os.path.basename(file_path)}...")

    # Load the CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"  Loaded {len(df)} prompts")
    except Exception as e:
        print(f"  Error loading CSV: {e}")
        return None, None, None

    # Check if the prompt column exists
    if 'full_optimized_prompt' not in df.columns:
        print(f"  Error: Column 'full_optimized_prompt' not found in CSV!")
        print(f"  Available columns: {list(df.columns)}")
        return None, None, None

    # Limit the number of samples if specified
    if max_samples > 0 and len(df) > max_samples:
        df = df.head(max_samples)
        print(f"  Processing first {max_samples} samples")

    results = []

    for idx, row in df.iterrows():
        prompt = row['full_optimized_prompt']

        # Skip empty or invalid prompts
        if pd.isna(prompt) or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            continue

        # Count tokens
        token_count = count_tokens(prompt, tokenizer)

        # Skip prompts that are too short or too long
        if token_count == 0 or token_count > 2048:
            continue

        # Calculate perplexity
        ppl = calculate_perplexity(prompt, model, tokenizer, device)

        # Calculate log perplexity (natural log)
        log_ppl = math.log(ppl) if ppl > 0 and ppl != float('inf') else float('nan')

        results.append({
            'token_count': token_count,
            'perplexity': ppl,
            'log_perplexity': log_ppl
        })

        # Print progress
        if (len(results) % 10) == 0 or len(results) <= 10:
            print(f"    Processed {len(results)} prompts... Last: tokens={token_count}, log(PPL)={log_ppl:.2f}")

    print(f"  Completed processing {len(results)} prompts")

    if results:
        df_results = pd.DataFrame(results)
        token_counts = df_results['token_count'].values
        perplexities = df_results['perplexity'].values
        log_perplexities = df_results['log_perplexity'].values

        # Filter out infinite or NaN values
        valid_mask = np.isfinite(perplexities) & np.isfinite(log_perplexities)
        token_counts_valid = token_counts[valid_mask]
        perplexities_valid = perplexities[valid_mask]
        log_perplexities_valid = log_perplexities[valid_mask]

        return token_counts_valid, perplexities_valid, log_perplexities_valid

    return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Plot perplexity vs token count for multiple files")
    parser.add_argument("--files", nargs='+',
                        default=[
                            "/work/table-fp/nanoGCG-main/assets/optimized_prompts_1.5b_10init.csv",
                            "/work/table-fp/nanoGCG-main/assets/optimized_prompts_1.5b_20init.csv",
                            "/work/table-fp/nanoGCG-main/assets/optimized_prompts_1.5b.csv"
                        ],
                        help="List of CSV files to process")
    parser.add_argument("--model", default="/work/models/openai-community/gpt2",
                        help="Model path for perplexity calculation")
    parser.add_argument("--device", default="cuda:5",
                        help="Device for calculation")
    parser.add_argument("--output-plot", default="multiple_files_perplexity_plot.png",
                        help="Output plot file")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "jpg"],
                        help="Plot file format")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum number of samples to process per file")
    parser.add_argument("--labels", nargs='+',
                        default=["10 init", "20 init", "default"],
                        help="Labels for legend (must match number of files)")
    parser.add_argument("--no-trend-line", nargs='+',
                        default=[],
                        help="List of labels for which NOT to draw trend lines")

    args = parser.parse_args()

    # Check that labels match number of files
    if len(args.labels) != len(args.files):
        print(f"Error: Number of labels ({len(args.labels)}) must match number of files ({len(args.files)})")
        return

    # Check if input files exist
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Error: Input file '{file_path}' not found!")
            return

    print("="*60)
    print("Multiple Files Perplexity Analysis")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output plot: {args.output_plot}")
    print(f"Max samples per file: {args.max_samples}")
    print("="*60)

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map={"": args.device}
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Define colors for each file
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    all_data = []

    # Process each file
    for i, (file_path, label) in enumerate(zip(args.files, args.labels)):
        color = colors[i % len(colors)]

        token_counts, perplexities, log_perplexities = process_file(
            file_path, model, tokenizer, args.device, args.max_samples
        )

        if token_counts is not None and len(token_counts) > 0:
            # Plot scatter points
            ax.scatter(token_counts, log_perplexities, alpha=0.6, s=20,
                      color=color, label=label)

            # Store basic statistics
            stats = {
                'label': label,
                'color': color,
                'n_samples': len(token_counts),
                'mean_tokens': token_counts.mean(),
                'mean_log_ppl': log_perplexities.mean(),
                'std_tokens': token_counts.std(),
                'std_log_ppl': log_perplexities.std(),
            }

            # Add trend line (unless this label is in no-trend-line list)
            if len(token_counts) > 1 and label not in args.no_trend_line:
                z = np.polyfit(token_counts, log_perplexities, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(token_counts.min(), token_counts.max(), 100)
                ax.plot(x_trend, p(x_trend), color=color, linestyle='--',
                       alpha=0.8, linewidth=2)

                # Add trend line stats
                stats['slope'] = z[0]
                stats['intercept'] = z[1]
                stats['correlation'] = np.corrcoef(token_counts, log_perplexities)[0, 1]
            else:
                # No trend line stats
                stats['slope'] = 'N/A'
                stats['intercept'] = 'N/A'
                stats['correlation'] = 'N/A'

            all_data.append(stats)

    # Customize plot
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Log Perplexity')
    ax.set_title('Log Perplexity vs Token Count for Different Initialization Methods')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {args.output_plot}")

    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS BY FILE")
    print("="*60)
    for data in all_data:
        print(f"\n{data['label']} ({data['color']}):")
        print(f"  Samples: {data['n_samples']}")
        print(f"  Token Count: mean={data['mean_tokens']:.2f} ± {data['std_tokens']:.2f}")
        print(f"  Log Perplexity: mean={data['mean_log_ppl']:.4f} ± {data['std_log_ppl']:.4f}")
        print(f"  Trend line: log(PPL) = {data['slope']:.4f} * tokens + {data['intercept']:.4f}")
        print(f"  Correlation: r = {data['correlation']:.4f}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    print("\nDone!")

if __name__ == "__main__":
    main()