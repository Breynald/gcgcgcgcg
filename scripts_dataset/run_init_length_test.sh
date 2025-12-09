#!/bin/bash
# Test different optim_str_init lengths (3-16) on first 3 questions

# Configuration
INPUT_CSV="../assets/question.csv"
MODEL="/work/models/Qwen/Qwen2.5-1.5B"
TABLE_ROWS=1
TABLE_COLS=3
MAX_ROWS=3
NUM_STEPS=1500
DEVICE="cuda"
GPU_ID="4"  # Single GPU (backward compatibility)
GPU_IDS="0,1,3"   # Multiple GPUs
PARALLEL_JOBS=""  # Auto-detect based on GPU count
DTYPE="float16"
EARLY_STOP="True"
EARLY_STOP_LOSS_THRESHOLD="0.05"
DYNAMIC_CONFIDENCE="True"
TEST_BEST_RESPONSE="True"
BUFFER_SIZE="3"
USE_MELLOWMAX="False"

# Output directory
OUTPUT_DIR="init_length_test_results"
mkdir -p $OUTPUT_DIR

echo "Testing optim_str_init lengths from 3 to 16"
echo "=========================================="
echo "Model: $MODEL"
echo "Questions: first $MAX_ROWS from $INPUT_CSV"
echo "Table size: ${TABLE_ROWS}x${TABLE_COLS}"
echo "Optimization steps: $NUM_STEPS"
echo ""

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --gpu-id ID             Single GPU ID to use (default: $GPU_ID)"
    echo "  --gpu-ids IDS           Comma-separated GPU IDs for parallel processing (e.g., \"0,1,2,3\")"
    echo "  --parallel-jobs N       Number of parallel jobs (default: auto-detect from GPU count)"
    echo "  --min-length N          Minimum init string length (default: 3)"
    echo "  --max-length N          Maximum init string length (default: 16)"
    echo "  --num-steps N           Number of optimization steps (default: $NUM_STEPS)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings (single GPU)"
    echo "  $0"
    echo ""
    echo "  # Use multiple GPUs for parallel processing"
    echo "  $0 --gpu-ids \"0,1,2,3\""
    echo ""
    echo "  # Test different length range with custom step count"
    echo "  $0 --min-length 5 --max-length 10 --num-steps 1000"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --min-length)
            MIN_LENGTH="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Set default length range if not specified
MIN_LENGTH=${MIN_LENGTH:-3}
MAX_LENGTH=${MAX_LENGTH:-16}

# Determine GPU configuration
if [[ -n "$GPU_IDS" ]]; then
    # Multi-GPU mode
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    NUM_GPUS=${#GPU_ARRAY[@]}

    # Set parallel jobs count
    if [[ -n "$PARALLEL_JOBS" ]]; then
        PARALLEL_COUNT=$PARALLEL_JOBS
    else
        PARALLEL_COUNT=$NUM_GPUS
    fi

    # Ensure we don't exceed available GPUs
    if [[ $PARALLEL_COUNT -gt $NUM_GPUS ]]; then
        PARALLEL_COUNT=$NUM_GPUS
    fi

    MULTI_GPU=true
    echo "Multi-GPU Optimization Mode:"
    echo "  Available GPUs: ${GPU_ARRAY[@]}"
    echo "  Parallel jobs: $PARALLEL_COUNT"
else
    # Single GPU mode
    MULTI_GPU=false
    echo "Single-GPU Optimization Mode:"
    echo "  GPU ID: $GPU_ID"
fi

echo "  Input CSV: $INPUT_CSV"
echo "  Model: $MODEL"
echo "  Length range: $MIN_LENGTH to $MAX_LENGTH"
echo "  Optimization steps: $NUM_STEPS"
echo ""

# Loop through lengths MIN_LENGTH to MAX_LENGTH
for LENGTH in $(seq $MIN_LENGTH $MAX_LENGTH); do
    echo "Testing optim_str_init length: $LENGTH"

    # Create init string with X tokens (space-separated)
    INIT_STR=$(for i in $(seq 1 $LENGTH); do echo -n "X "; done | sed 's/ $//')

    # Output file for this length
    OUTPUT_CSV="${OUTPUT_DIR}/optimized_length_${LENGTH}.csv"

    echo "  Init string: $INIT_STR"
    echo "  Output: $OUTPUT_CSV"

    if [[ "$MULTI_GPU" == true ]]; then
        # Multi-GPU mode - use run_dataset_optimization.sh
        CMD="./run_dataset_optimization.sh"
        CMD="$CMD --input-csv '$INPUT_CSV'"
        CMD="$CMD --output-csv '$OUTPUT_CSV'"
        CMD="$CMD --model '$MODEL'"
        CMD="$CMD --table-rows $TABLE_ROWS"
        CMD="$CMD --table-cols $TABLE_COLS"
        CMD="$CMD --max-rows $MAX_ROWS"
        CMD="$CMD --num-steps $NUM_STEPS"
        CMD="$CMD --dtype '$DTYPE'"
        CMD="$CMD --gpu-ids '$GPU_IDS'"
        CMD="$CMD --parallel-jobs $PARALLEL_COUNT"
        CMD="$CMD --early-stop $EARLY_STOP"
        CMD="$CMD --early-stop-loss-threshold $EARLY_STOP_LOSS_THRESHOLD"
        CMD="$CMD --test-best-response $TEST_BEST_RESPONSE"
        CMD="$CMD --buffer-size $BUFFER_SIZE"
        CMD="$CMD --use-mellowmax $USE_MELLOWMAX"
        CMD="$CMD --dynamic-confidence $DYNAMIC_CONFIDENCE"
        CMD="$CMD --optim-str-init '$INIT_STR'"
    else
        # Single GPU mode - direct Python call
        CMD="python optimize_dataset_prompts.py"
        CMD="$CMD --input-csv '$INPUT_CSV'"
        CMD="$CMD --output-csv '$OUTPUT_CSV'"
        CMD="$CMD --model '$MODEL'"
        CMD="$CMD --table-rows $TABLE_ROWS"
        CMD="$CMD --table-cols $TABLE_COLS"
        CMD="$CMD --max-rows $MAX_ROWS"
        CMD="$CMD --num-steps $NUM_STEPS"
        CMD="$CMD --device '$DEVICE:$GPU_ID'"
        CMD="$CMD --dtype '$DTYPE'"
        CMD="$CMD --buffer-size $BUFFER_SIZE"
        CMD="$CMD --use-mellowmax $USE_MELLOWMAX"
        CMD="$CMD --dynamic-confidence $DYNAMIC_CONFIDENCE"
        CMD="$CMD --early-stop $EARLY_STOP"
        CMD="$CMD --early-stop-loss-threshold $EARLY_STOP_LOSS_THRESHOLD"
        CMD="$CMD --test-best-response $TEST_BEST_RESPONSE"
        CMD="$CMD --optim-str-init '$INIT_STR'"
    fi

    # Run optimization
    eval $CMD

    # Check if optimization succeeded
    if [ $? -eq 0 ]; then
        echo "  ✓ Optimization completed"
        # Show best loss if available
        if [ -f "$OUTPUT_CSV" ]; then
            BEST_LOSS=$(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f4 | head -n 1)
            if [ -n "$BEST_LOSS" ] && [ "$BEST_LOSS" != "" ]; then
                echo "  Best loss: $BEST_LOSS"
            fi
        fi
    else
        echo "  ❌ Optimization failed"
    fi
    echo ""
done

echo "All tests completed!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To calculate perplexity for each result, run:"
echo "python calculate_perplexities.py --input-dir $OUTPUT_DIR --model $MODEL"

# Clean up temporary file if it exists
if [ -f "${OUTPUT_DIR}/questions_3.csv" ]; then
    rm "${OUTPUT_DIR}/questions_3.csv"
fi