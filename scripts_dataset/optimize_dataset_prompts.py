#!/usr/bin/env python3
"""
Script to optimize all questions from a CSV dataset using nanoGCG.
Optimized prompts are saved to a separate CSV file.
Uses the same prompt generation as batch_optimization.py but replaces
the final instruction with dataset questions.
"""

import argparse
import os
import sys
import pandas as pd
from typing import List, Optional
import time
from pathlib import Path
import torch

# Add the parent directory to the path to import nanogcg
sys.path.append(str(Path(__file__).parent.parent))

import nanogcg
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanogcg import GCGConfig
from nanogcg.multigcg import GCGResult


def load_csv_data(csv_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load CSV data with questions, answers, and keywords.

    Args:
        csv_path: Path to the CSV file
        max_rows: Maximum number of rows to load (for testing)

    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_csv(csv_path)

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
        print(f"Limited to {max_rows} rows for testing")

    print(f"Loaded {len(df)} questions from {csv_path}")
    return df


def generate_question_table_prompt(table_rows: int, table_cols: int, question: str) -> dict:
    """
    Generate a table prompt with the dataset question as the final instruction.
    This follows the same pattern as batch_optimization.py but replaces the final line.

    Args:
        table_rows: Number of rows in the table
        table_cols: Number of columns in the table
        question: The question from the dataset to use as final instruction

    Returns:
        dict: Contains the prompt template and list of optim_str placeholders
    """
    # Generate column headers (same as batch_optimization)
    col_headers = []
    for i in range(table_cols):
        col_headers.append(f"Metric {chr(65 + i)}")  # A, B, C, D...

    # Start building the table (same as batch_optimization)
    lines = ["Here is a table:\n"]

    # Header row
    header_row = "| | " + " | ".join(col_headers) + " |"
    lines.append(header_row)

    # Separator row
    separator_row = "|" + "---|" * (table_cols + 1)
    lines.append(separator_row)

    # Data rows with placeholders (same as batch_optimization)
    placeholder_counter = 1
    optim_str_placeholders = []

    for row_idx in range(table_rows):
        row_cells = []
        for col_idx in range(table_cols):
            placeholder = f"{{optim_str_{placeholder_counter}}}"
            optim_str_placeholders.append(placeholder)
            placeholder_counter += 1
            row_cells.append(placeholder)

        row_name = f"Data {row_idx + 1}"
        data_row = f"| {row_name} | " + " | ".join(row_cells) + " |"
        lines.append(data_row)

    # Replace the final instruction with the dataset question
    lines.append(f"\n{question}")

    template_prompt = "\n".join(lines)

    return {
        "prompt": template_prompt,
        "optim_str_placeholders": optim_str_placeholders,
    }


def optimize_single_question(
    model,
    tokenizer,
    question: str,
    target_answer: str,
    table_rows: int,
    table_cols: int,
    config: GCGConfig
) -> GCGResult:
    """
    Optimize a single question using nanoGCG.

    Args:
        model: The language model
        tokenizer: The tokenizer
        question: The question to optimize
        target_answer: The target answer
        table_rows: Number of rows in the table
        table_cols: Number of columns in the table
        config: GCG configuration

    Returns:
        GCGResult with optimization results
    """
    # Generate table prompt with dataset question
    prompt_data = generate_question_table_prompt(table_rows, table_cols, question)
    full_prompt = prompt_data["prompt"]
    placeholders = prompt_data["optim_str_placeholders"]

    # Create messages format for multi-gcg
    messages = [{"role": "user", "content": full_prompt}]

    try:
        # Run multi-GCG optimization
        result = nanogcg.run_multigcg(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            target=target_answer,
            config=config,
            optim_str_placeholders=placeholders
        )
        return result
    except Exception as e:
        print(f"Error optimizing question '{question}': {e}")
        # Return a dummy result
        return GCGResult(
            best_loss=float('inf'),
            best_strings=["" for _ in placeholders],
            losses=[],
            strings=[]
        )


def save_optimized_results(
    original_df: pd.DataFrame,
    optimized_results: List[GCGResult],
    output_csv: str,
    table_rows: int,
    table_cols: int
):
    """
    Save optimized results to CSV.

    Args:
        original_df: Original DataFrame with questions
        optimized_results: List of optimization results
        output_csv: Output CSV path
        table_rows: Number of rows in the table
        table_cols: Number of columns in the table
    """
    results_data = []

    for idx, (_, row) in enumerate(original_df.iterrows()):
        if idx >= len(optimized_results):
            break

        result = optimized_results[idx]
        question = row['question']

        # Generate placeholders for this question to match the optimization
        prompt_data = generate_question_table_prompt(table_rows, table_cols, question)
        placeholders = prompt_data["optim_str_placeholders"]

        # Create base result entry
        result_entry = {
            'question': row['question'],
            'target_answer': row['answer'],
            'keyword': row.get('keyword', ''),
            'best_loss': result.best_loss,
            'num_placeholders': len(placeholders),
            'table_size': f"{table_rows}x{table_cols}"
        }

        # Note: Individual optimized strings are stored in full_optimized_prompt, no need to save separate columns

        # Add the complete optimized prompt
        # Reconstruct the table with optimized strings using the same format as batch_optimization
        col_headers = []
        for i in range(table_cols):
            col_headers.append(f"Metric {chr(65 + i)}")

        lines = ["Here is a table:\n"]

        # Header row
        header_row = "| | " + " | ".join(col_headers) + " |"
        lines.append(header_row)

        # Separator row
        separator_row = "|" + "---|" * (table_cols + 1)
        lines.append(separator_row)

        # Data rows with optimized strings
        placeholder_counter = 1
        string_index = 0
        for row_idx in range(table_rows):
            row_cells = []
            for col_idx in range(table_cols):
                if string_index < len(result.best_strings):
                    optimized_str = result.best_strings[string_index]
                else:
                    optimized_str = ""
                row_cells.append(optimized_str)
                string_index += 1
                placeholder_counter += 1

            row_name = f"Data {row_idx + 1}"
            data_row = f"| {row_name} | " + " | ".join(row_cells) + " |"
            lines.append(data_row)

        # Add the original question
        lines.append(f"\n{question}")

        full_optimized_prompt = "\n".join(lines)
        result_entry['full_optimized_prompt'] = full_optimized_prompt

        results_data.append(result_entry)

    # Save to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved {len(results_df)} optimized results to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Optimize questions from CSV dataset using nanoGCG")
    parser.add_argument("--input-csv", default="assets/question.csv",
                       help="Path to input CSV file with questions")
    parser.add_argument("--output-csv", default="assets/optimized_prompts.csv",
                       help="Path to output CSV file for optimized prompts")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3",
                       help="Model to use for optimization")
    parser.add_argument("--table-rows", type=int, default=2,
                       help="Number of rows in the optimization table")
    parser.add_argument("--table-cols", type=int, default=2,
                       help="Number of columns in the optimization table")
    parser.add_argument("--max-rows", type=int, default=None,
                       help="Maximum number of questions to optimize (for testing)")
    parser.add_argument("--num-steps", type=int, default=250,
                       help="Number of optimization steps")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use (cuda:0, cuda:1, cpu)")
    parser.add_argument("--dtype", default="float16",
                       help="Data type for model (float16/float32)")
    parser.add_argument("--early-stop-confidence", type=float, default=None,
                       help="Confidence threshold for early stopping (0.0-1.0, only used if early_stop=True)")

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file '{args.input_csv}' not found!")
        sys.exit(1)

    print(f"Starting optimization with {args.table_rows}x{args.table_cols} tables")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")

    # Load data
    df = load_csv_data(args.input_csv, args.max_rows)

    # Setup model and tokenizer
    print("Loading model and tokenizer...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=getattr(torch, args.dtype),
            device_map=args.device
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model and tokenizer loaded successfully")
        print(f"Model device: {model.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Setup GCG configuration
    config = GCGConfig(
        num_steps=args.num_steps,
        buffer_size=5,
        use_mellowmax=True,
        early_stop=True,
        early_stop_confidence=args.early_stop_confidence,
        use_prefix_cache=False,  # Disable prefix cache to avoid issues
        filter_ids=False  # Disable token filtering to prevent optimization failures
    )

    # Optimize each question
    optimized_results = []

    total_questions = len(df)
    start_time = time.time()

    for idx, (_, row) in enumerate(df.iterrows()):
        question = row['question']
        target_answer = row['answer']

        print(f"\n[{idx+1}/{total_questions}] Optimizing: {question}")
        print(f"Target: {target_answer}")

        try:
            # Optimize the question
            result = optimize_single_question(
                model=model,
                tokenizer=tokenizer,
                question=question,
                target_answer=target_answer,
                table_rows=args.table_rows,
                table_cols=args.table_cols,
                config=config
            )

            optimized_results.append(result)

            print(f"Best loss: {result.best_loss:.4f}")
            if result.best_loss < 0.1:  # Good optimization
                print("✓ Good optimization achieved!")

        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            # Generate placeholders to create empty result
            prompt_data = generate_question_table_prompt(args.table_rows, args.table_cols, question)
            placeholders = prompt_data["optim_str_placeholders"]

            # Add empty result to maintain alignment
            empty_result = GCGResult(
                best_loss=float('inf'),
                best_strings=["" for _ in placeholders],
                losses=[],
                strings=[]
            )
            optimized_results.append(empty_result)

    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nOptimization completed in {total_time:.2f} seconds")
    print(f"Average time per question: {total_time/total_questions:.2f} seconds")

    # Save results
    save_optimized_results(
        original_df=df,
        optimized_results=optimized_results,
        output_csv=args.output_csv,
        table_rows=args.table_rows,
        table_cols=args.table_cols
    )

    print(f"\nOptimization complete! Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()