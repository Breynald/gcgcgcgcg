#!/usr/bin/env python3
"""Calculate perplexity for optimized prompts from init length tests and plot results."""

import os
import sys
import pandas as pd
import glob
from pathlib import Path
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

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


def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity for optimized prompts and plot results")
    parser.add_argument("--input-dir", default="init_length_test_results",
                        help="Directory containing optimized prompt CSV files")
    parser.add_argument("--model", default="/work/models/Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model path for perplexity calculation")
    parser.add_argument("--device", default="cuda:4",
                        help="Device for calculation")
    parser.add_argument("--output-csv", default="init_length_perplexity_results.csv",
                        help="Output CSV file with results")
    parser.add_argument("--output-plot", default="init_length_perplexity_plot.png",
                        help="Output plot file")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "jpg"],
                        help="Plot file format")

    args = parser.parse_args()

    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(args.input_dir, "optimized_length_*.csv"))
    csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")
    print(f"Loading model: {args.model}")

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

    results = []
    detailed_results = []

    for csv_file in csv_files:
        # Extract length from filename
        length = int(csv_file.split('_')[-1].split('.')[0])
        print(f"\nProcessing length {length}...")

        try:
            df = pd.read_csv(csv_file)
            perplexities = []

            for idx, row in df.iterrows():
                if 'full_optimized_prompt' in row and pd.notna(row['full_optimized_prompt']):
                    prompt = row['full_optimized_prompt']

                    # Calculate perplexity of the entire optimized prompt
                    ppl = calculate_perplexity(prompt, model, tokenizer, args.device)
                    perplexities.append(ppl)

                    # Store detailed results
                    detailed_results.append({
                        'init_length': length,
                        'row_index': idx,
                        'prompt_preview': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                        'perplexity': ppl
                    })

                    print(f"  Row {idx}: PPL = {ppl:.2f}")

            if perplexities:
                avg_ppl = sum(perplexities) / len(perplexities)
                min_ppl = min(perplexities)
                max_ppl = max(perplexities)
                std_ppl = np.std(perplexities)

                result = {
                    'init_length': length,
                    'avg_perplexity': avg_ppl,
                    'min_perplexity': min_ppl,
                    'max_perplexity': max_ppl,
                    'std_perplexity': std_ppl,
                    'num_prompts': len(perplexities)
                }
                results.append(result)

                print(f"  Average perplexity: {avg_ppl:.2f} ± {std_ppl:.2f}")
                print(f"  Min perplexity: {min_ppl:.2f}")
                print(f"  Max perplexity: {max_ppl:.2f}")
                print(f"  Processed {len(perplexities)} optimized prompts")
            else:
                print(f"  No valid optimized prompts found")

        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")

        # Save detailed results
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(args.output_csv.replace('.csv', '_detailed.csv'), index=False)
        print(f"Detailed results saved to {args.output_csv.replace('.csv', '_detailed.csv')}")

        # Create plot
        plt.figure(figsize=(12, 8))

        # Sort results by init_length
        results_sorted = sorted(results, key=lambda x: x['init_length'])
        lengths = [r['init_length'] for r in results_sorted]
        avg_ppls = [r['avg_perplexity'] for r in results_sorted]
        std_ppls = [r['std_perplexity'] for r in results_sorted]
        min_ppls = [r['min_perplexity'] for r in results_sorted]
        max_ppls = [r['max_perplexity'] for r in results_sorted]

        # Plot with error bars
        plt.errorbar(lengths, avg_ppls, yerr=std_ppls,
                    marker='o', capsize=5, capthick=2,
                    label='Average ± Std Dev', linewidth=2, markersize=8)

        # Fill between min and max
        plt.fill_between(lengths, min_ppls, max_ppls, alpha=0.2,
                         label='Min-Max Range', color='orange')

        plt.xlabel('Initial Optimization Length', fontsize=14)
        plt.ylabel('Perplexity of Optimized Prompts', fontsize=14)
        plt.title('Perplexity of Optimized Prompts by Initial Optimization Length', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Set integer ticks for x-axis
        plt.xticks(lengths)

        # Add value labels on points
        for i, (l, ppl) in enumerate(zip(lengths, avg_ppls)):
            plt.annotate(f'{ppl:.1f}', (l, ppl),
                        textcoords="offset points", xytext=(0,10), ha='center')

        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {args.output_plot}")

        # Print summary
        print("\nSummary:")
        print("Length | Avg PPL | Std Dev | Min PPL | Max PPL")
        print("---------------------------------------------")
        for r in results_sorted:
            print(f"{r['init_length']:7d} | {r['avg_perplexity']:8.2f} | {r['std_perplexity']:8.2f} | "
                  f"{r['min_perplexity']:7.2f} | {r['max_perplexity']:7.2f}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()