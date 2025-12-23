#!/usr/bin/env python3
"""Analyze relationship between perplexity and token count for optimized prompts."""

import os
import sys
import pandas as pd
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import math

def calculate_perplexity(text, model, tokenizer, device="cuda:4"):
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

def main():
    parser = argparse.ArgumentParser(description="Analyze perplexity vs token count for optimized prompts")
    parser.add_argument("--input-csv", default="/work/table-fp/nanoGCG-main/assets/optimized_prompts_1.5b.csv",
                        help="CSV file with optimized prompts")
    parser.add_argument("--model", default="/work/models/Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model path for perplexity calculation")
    parser.add_argument("--device", default="cuda:4",
                        help="Device for calculation")
    parser.add_argument("--output-csv", default="prompt_perplexity_token_analysis.csv",
                        help="Output CSV file with results")
    parser.add_argument("--output-plot", default="prompt_perplexity_token_plot.png",
                        help="Output plot file")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "jpg"],
                        help="Plot file format")
    parser.add_argument("--prompt-column", default="prompt",
                        help="Column name containing the prompts")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum number of samples to process")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file '{args.input_csv}' not found!")
        return

    print(f"Loading prompts from: {args.input_csv}")
    print(f"Loading model: {args.model}")

    # Load the CSV file
    try:
        df = pd.read_csv(args.input_csv)
        print(f"Loaded {len(df)} prompts")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Check if the prompt column exists
    if args.prompt_column not in df.columns:
        print(f"Error: Column '{args.prompt_column}' not found in CSV!")
        print(f"Available columns: {list(df.columns)}")
        return

    # Load model and tokenizer
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

    # Limit the number of samples if specified
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.head(args.max_samples)
        print(f"Processing first {args.max_samples} samples")

    results = []

    print("\nProcessing prompts...")
    for idx, row in df.iterrows():
        prompt = row[args.prompt_column]

        # Skip empty or invalid prompts
        if pd.isna(prompt) or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            continue

        # Count tokens
        token_count = count_tokens(prompt, tokenizer)

        # Skip prompts that are too short or too long
        if token_count == 0 or token_count > 2048:
            continue

        # Calculate perplexity
        ppl = calculate_perplexity(prompt, model, tokenizer, args.device)

        # Calculate log perplexity (natural log, base e)
        log_ppl = math.log(ppl) if ppl > 0 and ppl != float('inf') else float('nan')

        result = {
            'index': idx,
            'token_count': token_count,
            'perplexity': ppl,
            'log_perplexity': log_ppl,
            'prompt_preview': prompt[:100] + "..." if len(prompt) > 100 else prompt,
            'prompt_length': len(prompt)
        }
        results.append(result)

        # Print progress
        if (len(results) % 10) == 0 or len(results) <= 10:
            print(f"  Processed {len(results)} prompts... Last: tokens={token_count}, log(PPL)={log_ppl:.2f}")

    print(f"\nCompleted processing {len(results)} prompts")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")

        # Create plot - only the right plot (Log Perplexity vs Token Count)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Extract data for plotting
        token_counts = results_df['token_count'].values
        perplexities = results_df['perplexity'].values
        log_perplexities = results_df['log_perplexity'].values

        # Filter out infinite or NaN values for some plots
        valid_mask = np.isfinite(perplexities) & np.isfinite(log_perplexities)
        token_counts_valid = token_counts[valid_mask]
        perplexities_valid = perplexities[valid_mask]
        log_perplexities_valid = log_perplexities[valid_mask]

        # Plot: Log Perplexity vs Token Count
        ax.scatter(token_counts_valid, log_perplexities_valid, alpha=0.6, s=20, color='green')
        ax.set_xlabel('Token Count')
        ax.set_ylabel('Log Perplexity')
        ax.set_title('Log Perplexity vs Token Count')
        ax.grid(True, alpha=0.3)

        # Add trend line
        if len(token_counts_valid) > 1:
            z = np.polyfit(token_counts_valid, log_perplexities_valid, 1)
            p = np.poly1d(z)
            ax.plot(token_counts_valid, p(token_counts_valid), "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
            ax.legend()

        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {args.output_plot}")

        # Calculate and print statistics
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        print(f"Total prompts processed: {len(results)}")
        print(f"\nToken Count Statistics:")
        print(f"  Mean: {token_counts.mean():.2f}")
        print(f"  Std: {token_counts.std():.2f}")
        print(f"  Min: {token_counts.min()}")
        print(f"  Max: {token_counts.max()}")
        print(f"  Median: {np.median(token_counts):.2f}")

        print(f"\nPerplexity Statistics (valid values only):")
        print(f"  Mean: {perplexities_valid.mean():.2f}")
        print(f"  Std: {perplexities_valid.std():.2f}")
        print(f"  Min: {perplexities_valid.min():.2f}")
        print(f"  Max: {perplexities_valid.max():.2f}")
        print(f"  Median: {np.median(perplexities_valid):.2f}")

        print(f"\nLog Perplexity Statistics:")
        print(f"  Mean: {log_perplexities_valid.mean():.4f}")
        print(f"  Std: {log_perplexities_valid.std():.4f}")
        print(f"  Min: {log_perplexities_valid.min():.4f}")
        print(f"  Max: {log_perplexities_valid.max():.4f}")
        print(f"  Median: {np.median(log_perplexities_valid):.4f}")

        # Calculate correlation
        if len(token_counts_valid) > 1:
            corr = np.corrcoef(token_counts_valid, log_perplexities_valid)[0, 1]
            print(f"\nCorrelation between token count and log perplexity: {corr:.4f}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()