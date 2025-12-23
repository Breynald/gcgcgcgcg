#!/usr/bin/env python3
"""Plot perplexity distribution for multiple optimized prompt files."""

import os
import sys
import pandas as pd
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

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

def process_file(file_path, model, tokenizer, device, max_samples=1000, prompt_column="full_optimized_prompt"):
    """Process a single CSV file and return perplexities."""
    print(f"\nProcessing {os.path.basename(file_path)}...")

    # Load the CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"  Loaded {len(df)} prompts")
    except Exception as e:
        print(f"  Error loading CSV: {e}")
        return None

    # Check if the prompt column exists
    if prompt_column not in df.columns:
        print(f"  Error: Column '{prompt_column}' not found in CSV!")
        print(f"  Available columns: {list(df.columns)}")
        return None

    # Limit the number of samples if specified
    if max_samples > 0 and len(df) > max_samples:
        df = df.head(max_samples)
        print(f"  Processing first {max_samples} samples")

    perplexities = []
    log_perplexities = []

    for idx, row in df.iterrows():
        prompt = row[prompt_column]

        # Skip empty or invalid prompts
        if pd.isna(prompt) or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            continue

        # Calculate perplexity
        ppl = calculate_perplexity(prompt, model, tokenizer, device)

        if ppl != float('inf'):
            perplexities.append(ppl)
            # Calculate log perplexity (natural log)
            log_ppl = math.log(ppl)
            log_perplexities.append(log_ppl)

        # Print progress
        if (len(perplexities) % 20) == 0 or len(perplexities) <= 10:
            print(f"    Processed {len(perplexities)} valid prompts... Last: PPL={ppl:.2f}, log(PPL)={log_ppl:.2f}")

    print(f"  Completed processing {len(perplexities)} valid prompts")

    return {
        'perplexities': np.array(perplexities),
        'log_perplexities': np.array(log_perplexities)
    }

def main():
    parser = argparse.ArgumentParser(description="Plot perplexity distribution for multiple files")
    parser.add_argument("--files", nargs='+',
                        default=[
                            "/work/table-fp/nanoGCG-main/assets/optimized_prompts_1.5b.csv",
                            "/work/table-fp/nanoGCG-main/assets/optimized_prompts_7b.csv",
                            "/work/table-fp/nanoGCG-main/assets/processed_counterfactual_prompts.csv",
                            "/work/table-fp/nanoGCG-main/assets/question_ppl.csv"
                        ],
                        help="List of CSV files to process")
    parser.add_argument("--model", default="/work/models/openai-community/gpt2",
                        help="Model path for perplexity calculation")
    parser.add_argument("--device", default="cuda:5",
                        help="Device for calculation")
    parser.add_argument("--output-plot", default="multiple_files_distribution_plot.png",
                        help="Output plot file")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "jpg"],
                        help="Plot file format")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum number of samples to process per file")
    parser.add_argument("--labels", nargs='+',
                        default=["1.5b", "7b", "proflingo", "common"],
                        help="Labels for x-axis (must match number of files)")
    parser.add_argument("--prompt-column", default="full_optimized_prompt",
                        help="Column name containing prompts (default: full_optimized_prompt)")
    parser.add_argument("--plot-type", default="box", choices=["box", "violin", "swarm", "bar"],
                        help="Type of plot to generate (box, violin, swarm, or bar)")

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
    print("Multiple Files Perplexity Distribution Analysis")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output plot: {args.output_plot}")
    print(f"Max samples per file: {args.max_samples}")
    print(f"Plot type: {args.plot_type}")
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

    # Process each file
    all_data = []
    valid_data = []

    for i, (file_path, label) in enumerate(zip(args.files, args.labels)):
        data = process_file(file_path, model, tokenizer, args.device, args.max_samples, args.prompt_column)

        if data is not None and len(data['log_perplexities']) > 0:
            all_data.append({
                'label': label,
                'file': os.path.basename(file_path),
                'log_perplexities': data['log_perplexities'],
                'perplexities': data['perplexities']
            })

            # Calculate statistics
            valid_log_ppl = data['log_perplexities'][np.isfinite(data['log_perplexities'])]
            if len(valid_log_ppl) > 0:
                valid_data.append({
                    'label': label,
                    'n_samples': len(valid_log_ppl),
                    'mean_log_ppl': np.mean(valid_log_ppl),
                    'std_log_ppl': np.std(valid_log_ppl),
                    'median_log_ppl': np.median(valid_log_ppl),
                    'min_log_ppl': np.min(valid_log_ppl),
                    'max_log_ppl': np.max(valid_log_ppl),
                    'q25_log_ppl': np.percentile(valid_log_ppl, 25),
                    'q75_log_ppl': np.percentile(valid_log_ppl, 75)
                })

    if not all_data:
        print("Error: No valid data found in any file!")
        return

    # Define colors for each file
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Create the plot based on plot type
    if args.plot_type == "box":
        # For box plot with different colors
        for i, data in enumerate(all_data):
            color = colors[i % len(colors)]
            box_data = data['log_perplexities']
            box = ax.boxplot(box_data, positions=[i], widths=0.6,
                           patch_artist=True, labels=[data['label']])
            box['boxes'][0].set_facecolor(color)
            box['boxes'][0].set_alpha(0.7)
            # Set whiskers and median color
            for element in ['whiskers', 'medians', 'caps']:
                plt.setp(box[element], color=color)
    elif args.plot_type == "violin":
        # For violin plot with different colors
        for i, data in enumerate(all_data):
            color = colors[i % len(colors)]
            violin_data = data['log_perplexities']
            parts = ax.violinplot(violin_data, positions=[i], widths=0.6)
            # Color the violin
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            # Set other elements color
            for element in parts.keys():
                if element != 'bodies':
                    plt.setp(parts[element], color=color)
            ax.set_xticks(range(len(all_data)))
            ax.set_xticklabels([data['label'] for data in all_data])
    elif args.plot_type == "swarm":
        # For swarm plot with different colors
        for i, data in enumerate(all_data):
            color = colors[i % len(colors)]
            y_positions = data['log_perplexities']
            x_positions = [i] * len(y_positions)
            ax.scatter(x_positions, y_positions, c=color, alpha=0.7,
                      s=30, label=data['label'], edgecolors='black', linewidth=0.5)
        ax.set_xticks(range(len(all_data)))
        ax.set_xticklabels([data['label'] for data in all_data])
    elif args.plot_type == "bar":
        # For bar plot with different colors
        bar_colors = [colors[i % len(colors)] for i in range(len(valid_data))]
        bar_data = []
        for i, data in enumerate(valid_data):
            bar_data.append({
                'Dataset': data['label'],
                'Mean Log Perplexity': data['mean_log_ppl'],
                'Std Log Perplexity': data['std_log_ppl']
            })
        df_bar = pd.DataFrame(bar_data)
        bars = ax.bar(range(len(df_bar)), df_bar['Mean Log Perplexity'],
                     yerr=df_bar['Std Log Perplexity'], capsize=5,
                     alpha=0.7, color=bar_colors, edgecolor='black', linewidth=1)
        ax.set_xticks(range(len(df_bar)))
        ax.set_xticklabels(df_bar['Dataset'])

    # Customize plot
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Log Perplexity')
    ax.set_title(f'Log Perplexity Distribution Across Different Datasets ({args.plot_type} plot)')
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate x-axis labels if they might overlap
    if len(args.labels) > 3:
        plt.xticks(rotation=45, ha='right')

    # Add legend for swarm plot
    if args.plot_type == "swarm":
        ax.legend()

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {args.output_plot}")

    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS BY DATASET")
    print("="*60)
    for data in valid_data:
        print(f"\n{data['label']}:")
        print(f"  Samples: {data['n_samples']}")
        print(f"  Log Perplexity:")
        print(f"    Mean: {data['mean_log_ppl']:.4f} ± {data['std_log_ppl']:.4f}")
        print(f"    Median: {data['median_log_ppl']:.4f}")
        print(f"    Range: [{data['min_log_ppl']:.4f}, {data['max_log_ppl']:.4f}]")
        print(f"    IQR: [{data['q25_log_ppl']:.4f}, {data['q75_log_ppl']:.4f}]")

    # Additional statistics
    if len(valid_data) > 1:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        means = [d['mean_log_ppl'] for d in valid_data]
        labels = [d['label'] for d in valid_data]

        min_idx = np.argmin(means)
        max_idx = np.argmax(means)

        print(f"Lowest mean log perplexity: {labels[min_idx]} ({means[min_idx]:.4f})")
        print(f"Highest mean log perplexity: {labels[max_idx]} ({means[max_idx]:.4f})")

        # Perform ANOVA if we have multiple groups
        if len(valid_data) >= 3:
            from scipy import stats
            groups = [d['log_perplexities'] for d in all_data]
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                print(f"\nANOVA test:")
                print(f"  F-statistic: {f_stat:.4f}")
                print(f"  p-value: {p_value:.6f}")
                if p_value < 0.05:
                    print("  Result: Significant differences detected between groups (p < 0.05)")
                else:
                    print("  Result: No significant differences detected between groups (p >= 0.05)")
            except Exception as e:
                print(f"\nCould not perform ANOVA test: {e}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    print("\nDone!")

if __name__ == "__main__":
    main()