#!/usr/bin/env python3
"""
Extract failed questions from test results and save them to a new CSV file.
This script reads the optimized prompts CSV and identifies which questions would fail
based on the same criteria used in test_dataset_prompts.py.
"""

import argparse
import pandas as pd
import torch
import sys
from pathlib import Path

# Add the parent directory to the path to import nanogcg
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer


def test_single_prompt(model, tokenizer, prompt: str, target: str, max_new_tokens: int = None):
    """
    Test a single optimized prompt (same logic as in test_dataset_prompts.py).

    Returns:
        bool: True if the test passes (successful generation and target match)
    """
    try:
        # Calculate the number of tokens needed for the target
        target_tokens = tokenizer.encode(target, return_tensors="pt").shape[1]

        # Use provided max_new_tokens or calculated target tokens, whichever is larger
        final_max_tokens = max(max_new_tokens or 0, target_tokens)

        # Apply chat template if available (for models without chat_template like Llama 2)
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # For models without chat_template, use prompt directly
            final_prompt = prompt

        # Handle BOS token (consistent with multigcg.py:281-282)
        if tokenizer.bos_token and final_prompt.startswith(tokenizer.bos_token):
            final_prompt = final_prompt.replace(tokenizer.bos_token, "")

        # Tokenize input
        inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=final_max_tokens,
                do_sample=False,  # Use greedy decoding for deterministic results
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (remove the input prompt)
        input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        response = generated_text[len(input_text):].strip()

        # Check if response starts with target (prefix match)
        target_in_response = response.startswith(target.strip())

        return target_in_response

    except Exception as e:
        print(f"Error testing prompt: {e}")
        return False


def extract_failed_questions(df, model, tokenizer, max_new_tokens: int = None):
    """
    Identify failed questions from the DataFrame.

    Args:
        df: DataFrame with optimized prompts
        model: The language model
        tokenizer: The tokenizer
        max_new_tokens: Maximum new tokens to generate

    Returns:
        DataFrame: Contains only the failed questions in the same format as question.csv (question,answer,keyword)
    """
    failed_rows = []

    print(f"\nTesting {len(df)} prompts to identify failures...")
    print("=" * 60)

    for idx, row in df.iterrows():
        print(f"\n[{idx+1}/{len(df)}] Testing prompt...")
        print(f"Question: {row['question'][:80]}{'...' if len(row['question']) > 80 else ''}")

        prompt = row['full_optimized_prompt']
        target = row['target_answer']

        # Test the prompt
        is_success = test_single_prompt(model, tokenizer, prompt, target, max_new_tokens)

        if not is_success:
            # This is a failed question, add it to our list in question.csv format
            failed_row = {
                'question': row['question'],
                'answer': row['target_answer'],  # target_answer becomes answer in question.csv format
                'keyword': row.get('keyword', '')
            }
            failed_rows.append(failed_row)
            print(f"✗ FAILED - Added to extraction list")
        else:
            print(f"✓ PASSED")

    return pd.DataFrame(failed_rows)


def main():
    parser = argparse.ArgumentParser(description="Extract failed questions from optimized prompts")
    parser.add_argument("--input-csv", default="../assets/optimized_prompts_1.5b.csv",
                       help="Path to CSV file with optimized prompts")
    parser.add_argument("--output-csv", default="../assets/question2.csv",
                       help="Path to save failed questions")
    parser.add_argument("--model", default="/work/models/Qwen/Qwen2.5-1.5B",
                       help="Model to use for testing")
    parser.add_argument("--device", default="cuda:4",
                       help="Device to use for testing")
    parser.add_argument("--dtype", default="float16",
                       help="Data type for model")
    parser.add_argument("--max-rows", type=int, default=None,
                       help="Maximum number of prompts to test")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                       help="Maximum new tokens to generate (auto-calculated if not specified)")

    args = parser.parse_args()

    print(f"Loading optimized prompts from: {args.input_csv}")
    print(f"Output will be saved to: {args.output_csv}")

    # Check if input file exists
    if not Path(args.input_csv).exists():
        print(f"Error: Input CSV file '{args.input_csv}' not found!")
        sys.exit(1)

    # Load optimized prompts
    df = pd.read_csv(args.input_csv)

    if args.max_rows is not None and args.max_rows > 0:
        df = df.head(args.max_rows)
        print(f"Limited to {args.max_rows} rows for testing")

    print(f"Loaded {len(df)} optimized prompts")

    if len(df) == 0:
        print("No prompts found in the CSV file!")
        return

    print(f"\nLoading model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")

    # Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=getattr(torch, args.dtype),
            device_map=args.device
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract failed questions
    failed_df = extract_failed_questions(df, model, tokenizer, args.max_new_tokens)

    # Save failed questions to CSV
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompts tested: {len(df)}")
    print(f"Failed prompts: {len(failed_df)} ({len(failed_df)/len(df)*100:.1f}%)")

    if len(failed_df) > 0:
        # Create output directory if it doesn't exist
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        failed_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved {len(failed_df)} failed questions to: {args.output_csv}")

        # Show first few failed questions
        print(f"\nFirst 5 failed questions:")
        for i, (_, row) in enumerate(failed_df.head(5).iterrows()):
            print(f"\n{i+1}. Question: {row['question']}")
            print(f"   Answer: {row['answer']}")
            print(f"   Keyword: {row['keyword']}")
    else:
        print("\nNo failed questions found! All prompts passed the test.")


if __name__ == "__main__":
    main()