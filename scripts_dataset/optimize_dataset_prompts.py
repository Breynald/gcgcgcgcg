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
import subprocess
import math

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


def split_dataset_for_multi_gpu(df: pd.DataFrame, num_jobs: int, temp_dir: str) -> List[str]:
    """
    Split dataset into multiple parts for multi-GPU processing.

    Args:
        df: Input DataFrame
        num_jobs: Number of parallel jobs
        temp_dir: Temporary directory to save split files

    Returns:
        List of paths to split CSV files
    """
    split_files = []
    total_rows = len(df)
    rows_per_job = math.ceil(total_rows / num_jobs)

    print(f"Splitting {total_rows} rows into {num_jobs} parts (~{rows_per_job} rows each)")

    for i in range(num_jobs):
        start_idx = i * rows_per_job
        end_idx = min((i + 1) * rows_per_job, total_rows)

        if start_idx < total_rows:
            split_df = df.iloc[start_idx:end_idx].copy()
            split_file = os.path.join(temp_dir, f"split_{i:02d}.csv")
            split_df.to_csv(split_file, index=False)
            split_files.append(split_file)
            print(f"  Part {i+1}: rows {start_idx+1}-{end_idx} -> {split_file}")

    return split_files


def merge_multi_gpu_results(split_files: List[str], output_csv: str) -> None:
    """
    Merge results from multiple GPU jobs into a single CSV file.

    Args:
        split_files: List of paths to result CSV files
        output_csv: Path to final merged output CSV file
    """
    print("Merging results from multiple GPUs...")

    all_dfs = []
    for i, result_file in enumerate(split_files):
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            all_dfs.append(df)
            print(f"  Loaded {len(df)} results from GPU {i}")
        else:
            print(f"  Warning: Result file {result_file} not found")

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(output_csv, index=False)
        print(f"✓ Merged {len(merged_df)} total results to {output_csv}")
    else:
        print("❌ No result files found to merge")
        raise FileNotFoundError("No result files found for merging")


def run_single_gpu_job(args, gpu_id: int, input_csv: str, output_csv: str) -> None:
    """
    Run optimization on a single GPU with a subset of data.

    Args:
        args: Command line arguments
        gpu_id: GPU ID to use
        input_csv: Input CSV file for this job
        output_csv: Output CSV file for this job
    """
    # Build command for single GPU job
    cmd = [
        sys.executable, __file__,
        "--input-csv", input_csv,
        "--output-csv", output_csv,
        "--model", args.model,
        "--table-rows", str(args.table_rows),
        "--table-cols", str(args.table_cols),
        "--num-steps", str(args.num_steps),
        "--device", f"cuda:{gpu_id}",
        "--dtype", args.dtype,
        "--buffer-size", str(args.buffer_size),
        "--use-mellowmax", str(args.use_mellowmax).lower()
    ]

    if args.max_rows:
        cmd.extend(["--max-rows", str(args.max_rows)])

    if args.early_stop:
        cmd.extend(["--early-stop", args.early_stop])

    if args.early_stop_confidence is not None:
        cmd.extend(["--early-stop-confidence", str(args.early_stop_confidence)])

    if args.test_best_response:
        cmd.extend(["--test-best-response", str(args.test_best_response).lower()])

    print(f"Starting job on GPU {gpu_id} with input: {input_csv}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ GPU {gpu_id} job completed successfully")
        if result.stdout:
            print(f"  Output: {result.stdout[-200:]}")  # Show last 200 chars
    except subprocess.CalledProcessError as e:
        print(f"❌ GPU {gpu_id} job failed:")
        print(f"  Error: {e.stderr}")
        raise


def run_multi_gpu_optimization_improved(args, gpu_ids: List[int], num_jobs: int) -> None:
    """
    Improved multi-GPU optimization with dynamic load balancing.

    This function implements true parallel processing with dynamic load balancing
    where completed GPUs can immediately take on new tasks.

    Args:
        args: Command line arguments
        gpu_ids: List of GPU IDs to use
        num_jobs: Number of parallel jobs
    """
    print("=" * 60)
    print("IMPROVED MULTI-GPU OPTIMIZATION MODE WITH DYNAMIC LOAD BALANCING")
    print("=" * 60)

    # Ensure temp directory exists
    os.makedirs(args.temp_dir, exist_ok=True)

    try:
        # Load the complete dataset
        print("Loading complete dataset...")
        df = load_csv_data(args.input_csv, args.max_rows)

        # Create optimal task chunks for better GPU utilization
        total_rows = len(df)

        # Calculate optimal chunk size: aim for 1-2 chunks per GPU for better load balancing
        # But ensure chunks aren't too small (inefficient) or too large (poor load balancing)
        optimal_chunks_per_gpu = 2
        total_chunks = min(num_jobs * optimal_chunks_per_gpu, total_rows)  # Don't create more chunks than rows
        chunk_size = max(1, total_rows // total_chunks)

        print(f"Splitting {total_rows} rows into {total_chunks} chunks of ~{chunk_size} rows each ({optimal_chunks_per_gpu} chunks per GPU)")

        # Create all task chunks
        all_chunks = []
        chunk_count = 0
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            if start_idx < total_rows:
                chunk_df = df.iloc[start_idx:end_idx].copy()
                chunk_file = os.path.join(args.temp_dir, f"chunk_{chunk_count:03d}.csv")
                chunk_df.to_csv(chunk_file, index=False)
                all_chunks.append(chunk_file)
                chunk_count += 1

        print(f"Created {len(all_chunks)} task chunks for dynamic load balancing")

        # Initialize task queue and result tracking
        pending_chunks = all_chunks.copy()
        completed_chunks = []
        failed_chunks = []
        active_processes = {}

        start_time = time.time()

        # Show progress header
        print(f"\n{'GPU':<6} {'Status':<15} {'Task':<10} {'Runtime':<10} {'Completed'}")
        print("-" * 70)

        # Main load balancing loop
        while len(completed_chunks) + len(failed_chunks) < len(all_chunks):
            # Fill available GPUs with pending tasks
            while len(active_processes) < len(gpu_ids) and pending_chunks:
                # Find next available GPU
                available_gpu = None
                for gpu_id in gpu_ids:
                    if gpu_id not in {p['gpu_id'] for p in active_processes.values()}:
                        available_gpu = gpu_id
                        break

                if available_gpu is not None:
                    # Assign next chunk to available GPU
                    chunk_file = pending_chunks.pop(0)
                    chunk_id = os.path.basename(chunk_file).replace('.csv', '')
                    result_file = os.path.join(args.temp_dir, f"result_{chunk_id}.csv")

                    try:
                        # 使用文件记录输出，避免管道缓冲区死锁
                        stdout_file = os.path.join(args.temp_dir, f"stdout_{chunk_id}.log")
                        stderr_file = os.path.join(args.temp_dir, f"stderr_{chunk_id}.log")

                        process = subprocess.Popen(
                            [sys.executable, __file__,
                             "--input-csv", chunk_file,
                             "--output-csv", result_file,
                             "--model", args.model,
                             "--table-rows", str(args.table_rows),
                             "--table-cols", str(args.table_cols),
                             "--num-steps", str(args.num_steps),
                             "--device", f"cuda:{available_gpu}",
                             "--dtype", args.dtype,
                             "--buffer-size", str(args.buffer_size),
                             "--use-mellowmax", str(args.use_mellowmax).lower()] +
                            (["--max-rows", str(args.max_rows)] if args.max_rows else []) +
                            ["--early-stop", args.early_stop] +
                            (["--early-stop-confidence", str(args.early_stop_confidence)] if args.early_stop_confidence is not None else []) +
                            ["--dynamic-confidence", str(args.dynamic_confidence).lower()] +
                            ["--test-best-response", str(args.test_best_response).lower()],
                            stdout=open(stdout_file, 'w'),
                            stderr=open(stderr_file, 'w'),
                            text=True
                        )

                        active_processes[process.pid] = {
                            'process': process,
                            'gpu_id': available_gpu,
                            'chunk_file': chunk_file,
                            'result_file': result_file,
                            'chunk_id': chunk_id,
                            'start_time': time.time(),
                            'stdout_file': stdout_file,
                            'stderr_file': stderr_file
                        }

                        print(f"GPU {available_gpu:3d} Started   {chunk_id:<10} 0s         {len(completed_chunks)}/{len(all_chunks)}")

                    except Exception as e:
                        print(f"❌ Failed to start task {chunk_id} on GPU {available_gpu}: {e}")
                        failed_chunks.append(chunk_file)
                        # 清理可能已创建的文件
                        try:
                            if 'stdout_file' in locals():
                                os.unlink(stdout_file)
                            if 'stderr_file' in locals():
                                os.unlink(stderr_file)
                        except:
                            pass
                else:
                    break  # No available GPUs

            # Monitor active processes with file-based output (no deadlocks)
            completed_pids = []
            current_time = time.time()

            for pid, proc_info in active_processes.items():
                process = proc_info['process']
                return_code = process.poll()
                runtime = int(current_time - proc_info['start_time'])

                # For long-running tasks, we don't set timeout - let them run naturally

                if return_code is not None:
                    # Process completed naturally
                    try:
                        # Read output files without blocking
                        stdout_content = ""
                        stderr_content = ""

                        try:
                            with open(proc_info['stdout_file'], 'r') as f:
                                stdout_content = f.read().strip()
                        except:
                            pass

                        try:
                            with open(proc_info['stderr_file'], 'r') as f:
                                stderr_content = f.read().strip()
                        except:
                            pass

                        if return_code == 0:
                            # Show final status with runtime in readable format
                            hours = runtime // 3600
                            minutes = (runtime % 3600) // 60
                            seconds = runtime % 60
                            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                            print(f"GPU {proc_info['gpu_id']:3d} ✓Done     {proc_info['chunk_id']:<10} {time_str} {len(completed_chunks)+1}/{len(all_chunks)}")
                            completed_chunks.append(proc_info['result_file'])
                        else:
                            print(f"GPU {proc_info['gpu_id']:3d} ✗Failed   {proc_info['chunk_id']:<10} {runtime:8d}s {len(completed_chunks)}/{len(all_chunks)}")
                            if stderr_content:
                                print(f"  Error: {stderr_content[:300]}...")
                            failed_chunks.append(proc_info['chunk_file'])

                        # Keep log files for debugging long-running tasks
                        # Don't delete them automatically

                    except Exception as e:
                        print(f"❌ Error monitoring task {proc_info['chunk_id']}: {e}")
                        failed_chunks.append(proc_info['chunk_file'])

                    completed_pids.append(pid)
                else:
                    # Process is still running - check for heartbeat in output
                    try:
                        if os.path.exists(proc_info['stdout_file']):
                            file_mtime = os.path.getmtime(proc_info['stdout_file'])
                            time_since_output = current_time - file_mtime

                            # If no output for more than 30 minutes, show warning
                            if time_since_output > 1800:  # 30 minutes
                                hours_stuck = int(time_since_output // 3600)
                                minutes_stuck = int((time_since_output % 3600) // 60)
                                print(f"GPU {proc_info['gpu_id']:3d} ⚠NoOutput {proc_info['chunk_id']:<10} {runtime:8d}s (No output for {hours_stuck}h{minutes_stuck}m)")
                    except:
                        pass

            # Remove completed processes from active list
            for pid in completed_pids:
                del active_processes[pid]

            # Show status update less frequently for long-running tasks
            if active_processes:
                total_progress = len(completed_chunks) + len(failed_chunks)
                progress_pct = total_progress / len(all_chunks) * 100

                # Only show detailed progress every 10 cycles to reduce overhead
                if len(completed_chunks) % 10 == 0 or len(failed_chunks) > 0:
                    print(f"Active: {len(active_processes)} GPUs, Progress: {total_progress}/{len(all_chunks)} ({progress_pct:.1f}%)")

            # Adaptive sleep: longer intervals for stable long-running tasks
            if len(active_processes) > 0:
                # Check every 30 seconds to reduce overhead while maintaining responsiveness
                time.sleep(30)
            else:
                # No active processes, check more frequently
                time.sleep(5)

        # Final summary
        print("\n" + "=" * 70)
        print("DYNAMIC LOAD BALANCING COMPLETED")
        print("=" * 70)
        print(f"✅ Completed chunks: {len(completed_chunks)}")
        print(f"❌ Failed chunks: {len(failed_chunks)}")
        print(f"⏱️  Total runtime: {int(time.time() - start_time)}s")

        if completed_chunks:
            # Merge all successful results
            print("\nMerging results from completed tasks...")
            merge_multi_gpu_results(completed_chunks, args.output_csv)
            print(f"\n🎉 Multi-GPU optimization completed successfully!")
            print(f"Final results saved to: {args.output_csv}")
        else:
            print("\n❌ No successful results to merge!")
            raise RuntimeError("All tasks failed")

    except Exception as e:
        print(f"\n❌ Improved Multi-GPU optimization failed: {e}")
        raise
    finally:
        # Optional: clean up temp directory
        try:
            pass  # Keep temp files for debugging
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up temp directory: {cleanup_error}")


def run_multi_gpu_optimization(args, gpu_ids: List[int], num_jobs: int) -> None:
    """
    Run optimization using multiple GPUs in parallel.

    Args:
        args: Command line arguments
        gpu_ids: List of GPU IDs to use
        num_jobs: Number of parallel jobs
    """
    print("=" * 60)
    print("MULTI-GPU OPTIMIZATION MODE")
    print("=" * 60)

    # Ensure temp directory exists
    os.makedirs(args.temp_dir, exist_ok=True)

    try:
        # Load the complete dataset
        print("Loading complete dataset...")
        df = load_csv_data(args.input_csv, args.max_rows)

        # Split dataset for parallel processing
        split_input_files = split_dataset_for_multi_gpu(df, num_jobs, args.temp_dir)

        # Prepare output file paths for each job
        split_output_files = []
        for i in range(len(split_input_files)):
            output_file = os.path.join(args.temp_dir, f"result_{i:02d}.csv")
            split_output_files.append(output_file)

        # Launch parallel jobs
        print(f"\nLaunching {len(split_input_files)} parallel jobs...")
        processes = []

        for i, (input_file, output_file) in enumerate(zip(split_input_files, split_output_files)):
            gpu_id = gpu_ids[i % len(gpu_ids)]

            try:
                # Use file-based output to prevent deadlocks for long-running tasks
                stdout_file = os.path.join(args.temp_dir, f"job_{i:02d}_stdout.log")
                stderr_file = os.path.join(args.temp_dir, f"job_{i:02d}_stderr.log")

                # Start the process in background
                process = subprocess.Popen(
                    [sys.executable, __file__,
                     "--input-csv", input_file,
                     "--output-csv", output_file,
                     "--model", args.model,
                     "--table-rows", str(args.table_rows),
                     "--table-cols", str(args.table_cols),
                     "--num-steps", str(args.num_steps),
                     "--device", f"cuda:{gpu_id}",
                     "--dtype", args.dtype,
                     "--buffer-size", str(args.buffer_size),
                     "--use-mellowmax", str(args.use_mellowmax).lower()] +
                    (["--max-rows", str(args.max_rows)] if args.max_rows else []) +
                    ["--early-stop", args.early_stop] +
                    (["--early-stop-confidence", str(args.early_stop_confidence)] if args.early_stop_confidence is not None else []) +
                    ["--dynamic-confidence", str(args.dynamic_confidence).lower()] +
                    ["--test-best-response", str(args.test_best_response).lower()],
                    stdout=open(stdout_file, 'w'),
                    stderr=open(stderr_file, 'w'),
                    text=True
                )
                processes.append((process, gpu_id, i, stdout_file, stderr_file))
                print(f"  Started job {i+1} on GPU {gpu_id}")
            except Exception as e:
                print(f"❌ Failed to start job {i+1} on GPU {gpu_id}: {e}")
                continue

        # Monitor processes with real-time parallel monitoring
        print("\nMonitoring parallel jobs with real-time progress...")
        failed_jobs = []
        completed_jobs = []
        start_time = time.time()

        # Show progress header
        print(f"{'Job':<6} {'GPU':<6} {'Status':<15} {'Runtime':<10} {'Progress'}")
        print("-" * 60)

        while len(completed_jobs) < len(processes):
            active_jobs = []
            current_time = time.time()

            for i, (process, gpu_id, job_id, stdout_file, stderr_file) in enumerate(processes):
                if i in completed_jobs:
                    continue

                # Check if process is still running
                return_code = process.poll()

                if return_code is None:
                    # Process is still running
                    runtime = int(current_time - start_time)
                    # Check for heartbeat in output files
                    try:
                        if os.path.exists(stdout_file):
                            file_mtime = os.path.getmtime(stdout_file)
                            time_since_output = current_time - file_mtime
                            if time_since_output > 1800:  # 30 minutes no output
                                hours_stuck = int(time_since_output // 3600)
                                minutes_stuck = int((time_since_output % 3600) // 60)
                                active_jobs.append(f"Job {job_id+1:3d} GPU {gpu_id:3d} ⚠Stuck    {runtime:8d}s  No output {hours_stuck}h{minutes_stuck}m")
                            else:
                                active_jobs.append(f"Job {job_id+1:3d} GPU {gpu_id:3d} Running   {runtime:8d}s  Active")
                        else:
                            active_jobs.append(f"Job {job_id+1:3d} GPU {gpu_id:3d} Running   {runtime:8d}s  Starting...")
                    except:
                        active_jobs.append(f"Job {job_id+1:3d} GPU {gpu_id:3d} Running   {runtime:8d}s  Unknown status")
                else:
                    # Process has finished - read output files without blocking
                    try:
                        stdout_content = ""
                        stderr_content = ""

                        try:
                            with open(stdout_file, 'r') as f:
                                stdout_content = f.read().strip()
                        except:
                            pass

                        try:
                            with open(stderr_file, 'r') as f:
                                stderr_content = f.read().strip()
                        except:
                            pass

                        runtime = int(current_time - start_time)

                        if return_code == 0:
                            # Format time for long-running tasks
                            hours = runtime // 3600
                            minutes = (runtime % 3600) // 60
                            time_str = f"{hours:02d}:{minutes:02d}:{runtime % 60:02d}"
                            print(f"Job {job_id+1:3d} GPU {gpu_id:3d} ✓Done    {time_str}  Completed successfully")
                            completed_jobs.append(i)
                        else:
                            print(f"Job {job_id+1:3d} GPU {gpu_id:3d} ✗Failed   {runtime:8d}s  Return code: {return_code}")
                            if stderr_content:
                                print(f"  Error: {stderr_content[:300]}...")  # Show more for debugging
                            failed_jobs.append(job_id)
                            completed_jobs.append(i)

                        # Keep log files for debugging
                    except Exception as e:
                        print(f"❌ Error reading output for job {job_id+1}: {e}")
                        failed_jobs.append(job_id)
                        completed_jobs.append(i)

            # Show active jobs status less frequently for long-running tasks
            if active_jobs and len(completed_jobs) < 5:  # Only show at beginning
                for job_status in active_jobs:
                    print(job_status)
                print("-" * 60)

            # Adaptive sleep for long-running tasks
            if len(active_jobs) > 0:
                time.sleep(60)  # Check every minute for stable long-running tasks
            else:
                time.sleep(10)  # Check more frequently when no active jobs

            # Show completion progress less frequently for long-running tasks
            if len(completed_jobs) > 0 and len(completed_jobs) % 5 == 0:  # Only every 5 completions
                progress = len(completed_jobs) / len(processes) * 100
                current_time = time.time()
                elapsed_hours = int((current_time - start_time) // 3600)
                elapsed_minutes = int(((current_time - start_time) % 3600) // 60)

                print(f"Progress: {len(completed_jobs)}/{len(processes)} jobs ({progress:.1f}%) - Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}")

                # Show remaining active jobs
                still_running = len(processes) - len(completed_jobs)
                if still_running > 0:
                    print(f"Still running: {still_running} jobs")
                print("-" * 60)

        # Check if any jobs failed
        if failed_jobs:
            print(f"\n⚠ Warning: {len(failed_jobs)} jobs failed")
            print("Failed job IDs:", [job_id+1 for job_id in failed_jobs])

        # Merge results
        print("\nMerging results from all jobs...")
        successful_results = []
        for i, output_file in enumerate(split_output_files):
            if i not in failed_jobs and os.path.exists(output_file):
                successful_results.append(output_file)
                print(f"  Found result file: {output_file}")
            else:
                print(f"  Missing result file: {output_file}")

        if successful_results:
            merge_multi_gpu_results(successful_results, args.output_csv)
            print(f"\n🎉 Multi-GPU optimization completed successfully!")
            print(f"Final results saved to: {args.output_csv}")
            print(f"Used {len(successful_results)}/{len(split_output_files)} job results")
        else:
            print("\n❌ No successful results to merge!")
            raise RuntimeError("All jobs failed")

    except Exception as e:
        print(f"\n❌ Multi-GPU optimization failed: {e}")
        raise
    finally:
        # Clean up temp directory (optional - comment out if you want to keep intermediate files)
        try:
            import shutil
            shutil.rmtree(args.temp_dir)
            print(f"Cleaned up temporary directory: {args.temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temp directory {args.temp_dir}: {e}")


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
    lines = []

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

        lines = []

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
    parser.add_argument("--early-stop", type=str, default="True",
                       choices=["True", "False"],
                       help="Enable early stopping (default: True)")
    parser.add_argument("--early-stop-confidence", type=float, default=None,
                       help="Confidence threshold for early stopping (0.0-1.0, only used if early_stop=True)")
    parser.add_argument("--dynamic-confidence", type=lambda x: x.lower() == 'true', default=False,
                       help="Use dynamic confidence that decreases over steps (default: False)")
    parser.add_argument("--test-best-response", type=lambda x: x.lower() == 'true', default=False,
                       help="Test current best response during optimization (default: False)")
    parser.add_argument("--buffer-size", type=int, default=3,
                       help="Buffer size for optimization (default: 3)")
    parser.add_argument("--use-mellowmax", type=lambda x: x.lower() == 'true', default=False,
                       help="Use mellowmax loss function (default: False)")

    # Multi-GPU parameters
    parser.add_argument("--multi-gpu", action="store_true",
                       help="Enable multi-GPU parallel processing")
    parser.add_argument("--gpu-ids", type=str,
                       help="Comma-separated GPU IDs for parallel processing (e.g., '0,1,2,3')")
    parser.add_argument("--parallel-jobs", type=int,
                       help="Number of parallel jobs (default: auto-detect from GPU count)")
    parser.add_argument("--temp-dir", type=str,
                       help="Temporary directory for split data and intermediate results")

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file '{args.input_csv}' not found!")
        sys.exit(1)

    # Check if multi-GPU mode is enabled
    if args.multi_gpu:
        if not args.gpu_ids:
            print("Error: --gpu-ids is required when using --multi-gpu")
            sys.exit(1)

        if not args.temp_dir:
            print("Error: --temp-dir is required when using --multi-gpu")
            sys.exit(1)

        # Parse GPU IDs
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]
        num_jobs = args.parallel_jobs if args.parallel_jobs else len(gpu_ids)

        print(f"Starting multi-GPU optimization with {args.table_rows}x{args.table_cols} tables")
        print(f"Model: {args.model}")
        print(f"Data type: {args.dtype}")
        print(f"GPU IDs: {gpu_ids}")
        print(f"Parallel jobs: {num_jobs}")
        print(f"Temporary directory: {args.temp_dir}")

        # Load and split data, then run improved parallel optimization with dynamic load balancing
        run_multi_gpu_optimization_improved(args, gpu_ids, num_jobs)
        return

    print(f"Starting single-GPU optimization with {args.table_rows}x{args.table_cols} tables")
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
        buffer_size=args.buffer_size,
        use_mellowmax=args.use_mellowmax,
        early_stop=args.early_stop == "True",
        early_stop_confidence=args.early_stop_confidence,
        dynamic_confidence=args.dynamic_confidence,  # 启用动态置信度
        test_best_response=args.test_best_response,  # 启用实时测试最佳回答
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