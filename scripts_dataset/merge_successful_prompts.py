#!/usr/bin/env python3
"""
Merge successful optimized prompts from the iterative optimization results back to the main results file.
This script reads both the main optimized prompts file and the latest iterative optimization results,
identifies which samples were successful in the iterative results, and updates the main file with these
successful optimizations.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the parent directory to the path to import nanogcg
sys.path.append(str(Path(__file__).parent.parent))

def test_single_prompt(model, tokenizer, prompt: str, target: str, max_new_tokens: int = None):
    """
    Test a single optimized prompt to verify success.

    Returns:
        bool: True if the test passes (successful generation and target match)
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

        return target_in_response

    except Exception as e:
        print(f"Error testing prompt: {e}")
        return False


def merge_successful_prompts(main_file, iterative_file, output_file, model_path, device, dtype, max_new_tokens=None):
    """
    Merge successful prompts from iterative optimization back to the main file.

    Args:
        main_file: Path to the main optimized prompts file (e.g., optimized_prompts_1.5b.csv)
        iterative_file: Path to the iterative optimization results (e.g., optimized_prompts_1.5b_2.csv)
        output_file: Path to save the updated main file
        model_path: Path to the model for testing
        device: Device to use for testing
        dtype: Data type for model
        max_new_tokens: Maximum new tokens for generation
    """

    print("Loading models...")
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, dtype),
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading data files...")
    # Load the data files
    main_df = pd.read_csv(main_file)
    iterative_df = pd.read_csv(iterative_file)

    print(f"Main file has {len(main_df)} entries")
    print(f"Iterative file has {len(iterative_df)} entries")

    # Create a copy of main_df for updating
    updated_df = main_df.copy()

    # Track updates
    updates_count = 0

    print("\nTesting iterative optimization results...")
    print("=" * 60)

    # Test each entry in the iterative results
    for idx, row in iterative_df.iterrows():
        print(f"\n[{idx+1}/{len(iterative_df)}] Testing iterative result...")
        print(f"Question: {row['question'][:80]}{'...' if len(row['question']) > 80 else ''}")

        prompt = row['full_optimized_prompt']
        target = row['target_answer']

        # Test the prompt from iterative results
        is_success = test_single_prompt(model, tokenizer, prompt, target, max_new_tokens)

        if is_success:
            print("✓ SUCCESS: This prompt works!")

            # Find the corresponding entry in the main file
            # Match based on question, target_answer, and keyword
            mask = (
                (updated_df['question'] == row['question']) &
                (updated_df['target_answer'] == row['target_answer']) &
                (updated_df['keyword'] == row['keyword'])
            )

            matches = updated_df[mask]

            if len(matches) > 0:
                # Update the first matching entry
                update_idx = matches.index[0]

                # Test the original prompt to see if it was already working
                original_prompt = updated_df.loc[update_idx, 'full_optimized_prompt']
                original_success = test_single_prompt(model, tokenizer, original_prompt, target, max_new_tokens)

                if not original_success:
                    # Update with the successful iterative result
                    updated_df.loc[update_idx, 'full_optimized_prompt'] = row['full_optimized_prompt']
                    updated_df.loc[update_idx, 'best_loss'] = row['best_loss']
                    # Add status column if it doesn't exist
                    if 'status' not in updated_df.columns:
                        updated_df['status'] = 'unknown'
                    updated_df.loc[update_idx, 'status'] = 'success'

                    print(f"✓ UPDATED entry at index {update_idx} in main file")
                    updates_count += 1
                else:
                    print(f"- Original prompt was already working, no update needed")
            else:
                print(f"- WARNING: No matching entry found in main file")
        else:
            print("✗ FAILED: This prompt still doesn't work")

    print("\n" + "=" * 60)
    print(f"Merge complete!")
    print(f"Total updates made: {updates_count}")
    print(f"Success rate: {updates_count}/{len(iterative_df)} ({100*updates_count/len(iterative_df):.1f}%)")

    # Save the updated main file
    updated_df.to_csv(output_file, index=False)
    print(f"Updated main file saved to: {output_file}")

    # Test the updated main file to verify overall success
    print("\nVerifying overall success rate in updated file...")
    success_count = 0
    total_count = len(updated_df)

    for idx, row in updated_df.iterrows():
        if idx % 10 == 0:
            print(f"Testing... {idx}/{total_count}")

        prompt = row['full_optimized_prompt']
        target = row['target_answer']

        if test_single_prompt(model, tokenizer, prompt, target, max_new_tokens):
            success_count += 1

    final_success_rate = 100 * success_count / total_count
    print(f"\nFinal verification:")
    print(f"Successful prompts: {success_count}/{total_count}")
    print(f"Overall success rate: {final_success_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Merge successful iterative optimization results back to main file")
    parser.add_argument("--main-file", type=str, default="../assets/optimized_prompts_1.5b.csv",
                       help="Path to the main optimized prompts file")
    parser.add_argument("--iterative-file", type=str, default="../assets/optimized_prompts_1.5b_2.csv",
                       help="Path to the iterative optimization results")
    parser.add_argument("--output-file", type=str, default="../assets/optimized_prompts_1.5b.csv",
                       help="Path to save the updated main file (can be same as main-file for in-place update)")
    parser.add_argument("--model", type=str, default="/work/models/Qwen/Qwen2.5-1.5B",
                       help="Model to use for testing")
    parser.add_argument("--device", type=str, default="cuda:4",
                       help="Device to use")
    parser.add_argument("--dtype", type=str, default="float16",
                       help="Data type for model")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                       help="Maximum new tokens to generate (auto-calculated if not specified)")

    args = parser.parse_args()

    # Check if input files exist
    if not Path(args.main_file).exists():
        print(f"Error: Main file '{args.main_file}' not found!")
        sys.exit(1)

    if not Path(args.iterative_file).exists():
        print(f"Error: Iterative file '{args.iterative_file}' not found!")
        sys.exit(1)

    # Merge the successful prompts
    merge_successful_prompts(
        args.main_file,
        args.iterative_file,
        args.output_file,
        args.model,
        args.device,
        args.dtype,
        args.max_new_tokens
    )


if __name__ == "__main__":
    main()