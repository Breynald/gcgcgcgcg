#!/usr/bin/env python3
"""Process counterfactual CSV to create full prompts by replacing {optim_str} with control text."""

import os
import sys
import pandas as pd
import argparse
import re

def process_prompts(input_file, output_file):
    """Process the counterfactual CSV to create prompts with control text."""

    # Load the CSV
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")

    # Check required columns
    required_columns = ['goals', 'targets', 'control']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Process each row
    processed_prompts = []

    for idx, row in df.iterrows():
        goal = row['goals']
        target = row['targets']
        control = row['control']

        # Replace {optim_str} with control text
        if '{optim_str}' in goal:
            prompt = goal.replace('{optim_str}', control)
        else:
            # If no {optim_str} placeholder, just concatenate
            prompt = f"{control} {goal}"

        # Clean up the prompt - handle multiple spaces and newlines
        prompt = re.sub(r'\s+', ' ', prompt.strip())

        processed_prompts.append({
            'id': row.get('id', idx),
            'prompt': prompt,
            'target': target,
            'control': control,
            'loss': row.get('loss', ''),
            'step': row.get('step', ''),
            'keyword': row.get('keyword', ''),
            'original_goal': goal
        })

    # Create new DataFrame
    result_df = pd.DataFrame(processed_prompts)

    # Save to CSV
    print(f"Saving processed prompts to: {output_file}")
    result_df.to_csv(output_file, index=False)
    print(f"Saved {len(result_df)} processed prompts")

    # Show some examples
    print("\nFirst few processed prompts:")
    print("-" * 80)
    for i in range(min(3, len(result_df))):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {result_df.iloc[i]['prompt'][:200]}...")
        print(f"Target: {result_df.iloc[i]['target']}")
        print(f"Control length: {len(result_df.iloc[i]['control'])} chars")

    return result_df

def main():
    parser = argparse.ArgumentParser(description="Process counterfactual prompts")
    parser.add_argument("--input", default="/work/table-fp/nanoGCG-main/assets/counterfactual_base_finance-64.csv",
                        help="Input CSV file")
    parser.add_argument("--output", default="processed_counterfactual_prompts.csv",
                        help="Output CSV file with processed prompts")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return

    # Process the file
    result_df = process_prompts(args.input, args.output)

    if result_df is not None:
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Input file: {args.input}")
        print(f"Output file: {args.output}")
        print(f"Total prompts processed: {len(result_df)}")

        # Calculate some statistics
        prompt_lengths = [len(p) for p in result_df['prompt']]
        control_lengths = [len(c) for c in result_df['control']]

        print(f"\nPrompt length statistics:")
        print(f"  Mean: {sum(prompt_lengths)/len(prompt_lengths):.0f} chars")
        print(f"  Min: {min(prompt_lengths)} chars")
        print(f"  Max: {max(prompt_lengths)} chars")

        print(f"\nControl text length statistics:")
        print(f"  Mean: {sum(control_lengths)/len(control_lengths):.0f} chars")
        print(f"  Min: {min(control_lengths)} chars")
        print(f"  Max: {max(control_lengths)} chars")

if __name__ == "__main__":
    main()