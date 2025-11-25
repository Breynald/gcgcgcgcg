#!/usr/bin/env python3
"""
Test script to validate the optimized prompts saved in the CSV file.
This script loads the optimized prompts and tests if they can generate the target responses.
"""

import argparse
import os
import sys
import pandas as pd
import torch
from pathlib import Path

# Add the parent directory to the path to import nanogcg
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_optimized_prompts(csv_path: str, max_rows: int = None):
    """
    Load optimized prompts from CSV file.

    Args:
        csv_path: Path to the CSV file
        max_rows: Maximum number of rows to load for testing

    Returns:
        DataFrame with optimized prompts
    """
    df = pd.read_csv(csv_path)

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
        print(f"Limited to {max_rows} rows for testing")

    print(f"Loaded {len(df)} optimized prompts from {csv_path}")
    return df


def test_single_prompt(model, tokenizer, prompt: str, target: str, max_new_tokens: int = None):
    """
    Test a single optimized prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The optimized prompt to test
        target: The expected target response
        max_new_tokens: Maximum number of tokens to generate (auto-calculated if None)

    Returns:
        dict: Test results
    """
    try:
        # Calculate the number of tokens needed for the target
        target_tokens = tokenizer.encode(target, return_tensors="pt").shape[1]

        # Use provided max_new_tokens or calculated target tokens, whichever is larger
        final_max_tokens = max(max_new_tokens or 0, target_tokens)

        # Apply chat template (consistent with multigcg.py)
        messages = [{"role": "user", "content": prompt}]
        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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

        return {
            "success": True,
            "response": response,
            "target_in_response": target_in_response,
            "response_length": len(response),
            "target_tokens": target_tokens,
            "generated_tokens": final_max_tokens,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "response": None,
            "target_in_response": False,
            "response_length": 0,
            "target_tokens": 0,
            "generated_tokens": 0,
            "error": str(e)
        }


def test_all_prompts(df, model, tokenizer, max_new_tokens: int = None):
    """
    Test all optimized prompts.

    Args:
        df: DataFrame with optimized prompts
        model: The language model
        tokenizer: The tokenizer
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        list: Test results for all prompts
    """
    results = []

    for idx, row in df.iterrows():
        print(f"\n[{idx+1}/{len(df)}] Testing prompt...")
        print(f"Question: {row['question']}")
        print(f"Target: {row['target_answer']}")

        prompt = row['full_optimized_prompt']
        target = row['target_answer']
        best_loss = row['best_loss']

        result = test_single_prompt(model, tokenizer, prompt, target, max_new_tokens)
        result.update({
            'question': row['question'],
            'target_answer': target,
            'best_loss': best_loss,
            'keyword': row.get('keyword', ''),
            'table_size': row['table_size']
        })

        results.append(result)

        if result['success']:
            print(f"✓ Generated: '{result['response'][:100]}{'...' if len(result['response']) > 100 else ''}'")
            print(f"Response starts with target: {'✓' if result['target_in_response'] else '✗'}")
        else:
            print(f"✗ Error: {result['error']}")

    return results




def analyze_results(results):
    """
    Analyze and print test results summary.

    Args:
        results: List of test results
    """
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    target_match = sum(1 for r in results if r['target_in_response'])

    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompts tested: {total}")
    print(f"Successfully generated: {successful} ({successful/total*100:.1f}%)")
    print(f"Prefix matches (response starts with target): {target_match} ({target_match/total*100:.1f}%)")

    
    print(f"\nSuccessful examples:")
    for i, r in enumerate(results):
        if r['success'] and r['target_in_response']:
            print(f"  {i+1}. Question: {r['question'][:50]}{'...' if len(r['question']) > 50 else ''}")
            print(f"     Generated: '{r['response'][:80]}{'...' if len(r['response']) > 80 else ''}'")
            print(f"     Target: '{r['target_answer']}'")
            print(f"     Best loss: {r['best_loss']:.4f}")
            if i >= 2:  # Show max 3 examples
                break

    print(f"\nFailed examples:")
    failed_examples = [r for r in results if not r['success'] or not r['target_in_response']]
    for i, r in enumerate(failed_examples[:3]):  # Show max 3 examples
        print(f"  {i+1}. Question: {r['question'][:50]}{'...' if len(r['question']) > 50 else ''}")
        if r['success']:
            print(f"     Generated: '{r['response'][:80]}{'...' if len(r['response']) > 80 else ''}'")
            print(f"     Target: '{r['target_answer']}' (not a prefix of response)")
        else:
            print(f"     Error: {r['error']}")


def main():
    parser = argparse.ArgumentParser(description="Test optimized prompts")
    parser.add_argument("--input-csv", default="assets/optimized_prompts.csv",
                       help="Path to CSV file with optimized prompts")
    parser.add_argument("--model", default="/work/models/Qwen/Qwen2.5-1.5B-Instruct",
                       help="Model to use for testing")
    parser.add_argument("--device", default="cuda:7",
                       help="Device to use for testing")
    parser.add_argument("--dtype", default="float16",
                       help="Data type for model")
    parser.add_argument("--max-rows", type=int, default=None,
                       help="Maximum number of prompts to test")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                       help="Maximum new tokens to generate (auto-calculated if not specified)")

    args = parser.parse_args()

    print(f"Loading optimized prompts from: {args.input_csv}")

    # Load optimized prompts
    df = load_optimized_prompts(args.input_csv, args.max_rows)

    if len(df) == 0:
        print("No prompts found in the CSV file!")
        return

    print(f"Loading model: {args.model}")
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

    # Test prompts
    print(f"\nTesting {len(df)} prompts...")
    results = test_all_prompts(df, model, tokenizer, args.max_new_tokens)

    # Analyze and print summary
    analyze_results(results)


if __name__ == "__main__":
    main()