#!/usr/bin/env python3
"""Calculate perplexity for optimized prompts from init length tests."""

import os
import sys
import pandas as pd
import glob
from pathlib import Path
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser = argparse.ArgumentParser(description="Calculate perplexity for optimized prompts")
    parser.add_argument("--input-dir", default="init_length_test_results",
                        help="Directory containing optimized prompt CSV files")
    parser.add_argument("--model", default="/work/models/Qwen/Qwen2.5-1.5B",
                        help="Model path for perplexity calculation")
    parser.add_argument("--device", default="cuda:4",
                        help="Device for calculation")
    parser.add_argument("--output", default="perplexity_results.csv",
                        help="Output CSV file")

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
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    results = []

    for csv_file in csv_files:
        # Extract length from filename
        length = int(csv_file.split('_')[-1].split('.')[0])
        print(f"\nProcessing length {length}...")

        try:
            df = pd.read_csv(csv_file)
            perplexities = []

            for _, row in df.iterrows():
                if 'full_optimized_prompt' in row and pd.notna(row['full_optimized_prompt']):
                    prompt = row['full_optimized_prompt']
                    ppl = calculate_perplexity(prompt, model, tokenizer, args.device)
                    perplexities.append(ppl)
                elif 'prompt' in row and pd.notna(row['prompt']):
                    prompt = row['prompt']
                    ppl = calculate_perplexity(prompt, model, tokenizer, args.device)
                    perplexities.append(ppl)

            if perplexities:
                avg_ppl = sum(perplexities) / len(perplexities)
                min_ppl = min(perplexities)
                max_ppl = max(perplexities)

                result = {
                    'init_length': length,
                    'avg_perplexity': avg_ppl,
                    'min_perplexity': min_ppl,
                    'max_perplexity': max_ppl,
                    'num_prompts': len(perplexities)
                }
                results.append(result)

                print(f"  Average perplexity: {avg_ppl:.2f}")
                print(f"  Min perplexity: {min_ppl:.2f}")
                print(f"  Max perplexity: {max_ppl:.2f}")
            else:
                print(f"  No valid prompts found")

        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

        # Print summary
        print("\nSummary:")
        print("Length | Avg PPL | Min PPL | Max PPL")
        print("----------------------------------")
        for r in results:
            print(f"{r['init_length']:7d} | {r['avg_perplexity']:8.2f} | {r['min_perplexity']:7.2f} | {r['max_perplexity']:7.2f}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()