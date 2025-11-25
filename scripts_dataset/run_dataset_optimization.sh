#!/bin/bash
# Shell script to run the dataset optimization with multi-GPU support

# Default parameters
INPUT_CSV="../assets/question2.csv"
OUTPUT_CSV="../assets/optimized_prompts2.csv"
MODEL="/work/models/Qwen/Qwen2.5-1.5B-Instruct"
TABLE_ROWS=1
TABLE_COLS=3
MAX_ROWS=""  # Empty means no limit
NUM_STEPS=1000
DEVICE="cuda"
DTYPE="float16"
GPU_ID="4"  # Single GPU (backward compatibility)
GPU_IDS="1,2,6,7"   # Multiple GPUs (new feature)
PARALLEL_JOBS=""  # Auto-detect based on GPU count
EARLY_STOP="True"
EARLY_STOP_CONFIDENCE="0.2"
DYNAMIC_CONFIDENCE="True"
TEST_BEST_RESPONSE="True"
BUFFER_SIZE="3"
USE_MELLOWMAX="False"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --input-csv PATH        Input CSV file (default: $INPUT_CSV)"
    echo "  --output-csv PATH       Output CSV file (default: $OUTPUT_CSV)"
    echo "  --model MODEL           Model to use (default: $MODEL)"
    echo "  --table-rows N          Number of table rows (default: $TABLE_ROWS)"
    echo "  --table-cols N          Number of table columns (default: $TABLE_COLS)"
    echo "  --max-rows N            Maximum questions to optimize for testing"
    echo "  --num-steps N           Number of optimization steps (default: $NUM_STEPS)"
    echo "  --device DEVICE         Device to use (default: $DEVICE)"
    echo "  --gpu-id ID             Single GPU ID to use (default: $GPU_ID)"
    echo "  --gpu-ids IDS           Comma-separated GPU IDs for parallel processing (e.g., \"0,1,2,3\")"
    echo "  --parallel-jobs N       Number of parallel jobs (default: auto-detect from GPU count)"
    echo "  --dtype DTYPE           Data type (default: $DTYPE)"
    echo "  --early-stop BOOL       Enable early stopping (default: $EARLY_STOP)"
    echo "  --early-stop-confidence NUM  Confidence threshold for early stop (0.0-1.0, default: $EARLY_STOP_CONFIDENCE)"
    echo "  --test-best-response BOOL   Test current best response during optimization (default: $TEST_BEST_RESPONSE)"
    echo "  --buffer-size N         Buffer size for optimization (default: $BUFFER_SIZE)"
    echo "  --use-mellowmax BOOL    Use mellowmax loss (default: $USE_MELLOWMAX)"
    echo "  --dynamic-confidence BOOL   Use dynamic confidence that decreases over steps (default: $DYNAMIC_CONFIDENCE)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings (single GPU)"
    echo "  $0"
    echo ""
    echo "  # Test with only 5 questions and 1x1 table on single GPU"
    echo "  $0 --max-rows 5 --table-rows 1 --table-cols 1"
    echo ""
    echo "  # Use multiple GPUs for parallel processing"
    echo "  $0 --gpu-ids \"0,1,2,3\""
    echo ""
    echo "  # Use specific GPUs with custom parallel job count"
    echo "  $0 --gpu-ids \"4,5,6,7\" --parallel-jobs 4"
    echo ""
    echo "  # Disable early stopping for full optimization"
    echo "  $0 --early-stop False"
    echo ""
    echo "  # Use confidence threshold for early stop (90% confidence)"
    echo "  $0 --early-stop True --early-stop-confidence 0.9"
    echo ""
    echo "  # Use higher confidence threshold (95% confidence) with multiple GPUs"
    echo "  $0 --gpu-ids \"0,1,2,3\" --early-stop True --early-stop-confidence 0.95 --table-rows 2 --table-cols 2"
    echo ""
    echo "  # Enable real-time testing of best responses during optimization"
    echo "  $0 --test-best-response True"
    echo ""
    echo "  # Enable dynamic confidence threshold (starts high, decreases over time)"
    echo "  $0 --early-stop True --early-stop-confidence 0.9 --dynamic-confidence True"
    echo ""
    echo "  # Combine multi-GPU with dynamic confidence"
    echo "  $0 --gpu-ids \"0,1,2,3\" --early-stop True --early-stop-confidence 0.8 --dynamic-confidence True"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-csv)
            INPUT_CSV="$2"
            shift 2
            ;;
        --output-csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --table-rows)
            TABLE_ROWS="$2"
            shift 2
            ;;
        --table-cols)
            TABLE_COLS="$2"
            shift 2
            ;;
        --max-rows)
            MAX_ROWS="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
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
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --early-stop)
            EARLY_STOP="$2"
            shift 2
            ;;
        --early-stop-confidence)
            EARLY_STOP_CONFIDENCE="$2"
            shift 2
            ;;
        --test-best-response)
            TEST_BEST_RESPONSE="$2"
            shift 2
            ;;
        --buffer-size)
            BUFFER_SIZE="$2"
            shift 2
            ;;
        --use-mellowmax)
            USE_MELLOWMAX="$2"
            shift 2
            ;;
        --dynamic-confidence)
            DYNAMIC_CONFIDENCE="$2"
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

# Determine GPU configuration and execution mode
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

    echo "Multi-GPU Optimization Mode:"
    echo "  Available GPUs: ${GPU_ARRAY[@]}"
    echo "  Parallel jobs: $PARALLEL_COUNT"
    echo "  Input CSV: $INPUT_CSV"
    echo "  Output CSV: $OUTPUT_CSV"
    echo "  Model: $MODEL"
    echo "  Table size: ${TABLE_ROWS}x${TABLE_COLS}"
    echo "  Optimization steps: $NUM_STEPS"
    echo "  Data type: $DTYPE"
    echo "  Early stopping: $EARLY_STOP"
    echo "  Early stop confidence: $EARLY_STOP_CONFIDENCE"
    echo "  Test best response: $TEST_BEST_RESPONSE"
    echo "  Buffer size: $BUFFER_SIZE"
    echo "  Use mellowmax: $USE_MELLOWMAX"
    echo "  Dynamic confidence: $DYNAMIC_CONFIDENCE"
    if [[ -n "$MAX_ROWS" ]]; then
        echo "  Max rows: $MAX_ROWS (testing mode)"
    fi
    echo ""

    # Create temporary directory for split outputs
    TEMP_DIR=$(mktemp -d)
    echo "Using temporary directory: $TEMP_DIR"

    # Prepare data splitting command
    SPLIT_SCRIPT="python optimize_dataset_prompts.py"
    SPLIT_CMD="$SPLIT_SCRIPT --input-csv '$INPUT_CSV' --output-csv '$OUTPUT_CSV' --model '$MODEL'"
    SPLIT_CMD="$SPLIT_CMD --table-rows $TABLE_ROWS --table-cols $TABLE_COLS --num-steps $NUM_STEPS"
    SPLIT_CMD="$SPLIT_CMD --dtype '$DTYPE' --buffer-size $BUFFER_SIZE --use-mellowmax $USE_MELLOWMAX"
    SPLIT_CMD="$SPLIT_CMD --dynamic-confidence $DYNAMIC_CONFIDENCE"

    if [[ -n "$MAX_ROWS" ]]; then
        SPLIT_CMD="$SPLIT_CMD --max-rows $MAX_ROWS"
    fi

    if [[ -n "$EARLY_STOP" ]]; then
        SPLIT_CMD="$SPLIT_CMD --early-stop $EARLY_STOP"
    fi

    if [[ -n "$EARLY_STOP_CONFIDENCE" ]]; then
        SPLIT_CMD="$SPLIT_CMD --early-stop-confidence $EARLY_STOP_CONFIDENCE"
    fi

    if [[ -n "$TEST_BEST_RESPONSE" ]]; then
        SPLIT_CMD="$SPLIT_CMD --test-best-response $TEST_BEST_RESPONSE"
    fi

    # Add multi-GPU specific parameters
    SPLIT_CMD="$SPLIT_CMD --multi-gpu --gpu-ids '$GPU_IDS' --parallel-jobs $PARALLEL_COUNT --temp-dir '$TEMP_DIR'"

    echo "Starting multi-GPU optimization..."
    echo "Command: $SPLIT_CMD"
    echo ""

    # Execute multi-GPU optimization
    eval $SPLIT_CMD

    # Check if optimization was successful
    if [[ $? -eq 0 ]]; then
        echo ""
        echo "Multi-GPU optimization completed successfully!"
        echo "Results saved to: $OUTPUT_CSV"

        # Clean up temporary directory
        rm -rf "$TEMP_DIR"
    else
        echo ""
        echo "Error: Multi-GPU optimization failed!"
        echo "Temporary directory preserved for debugging: $TEMP_DIR"
        exit 1
    fi

else
    # Single GPU mode (original behavior)
    echo "Single-GPU Optimization Mode:"
    echo "  Input CSV: $INPUT_CSV"
    echo "  Output CSV: $OUTPUT_CSV"
    echo "  Model: $MODEL"
    echo "  Table size: ${TABLE_ROWS}x${TABLE_COLS}"
    echo "  Optimization steps: $NUM_STEPS"
    echo "  Device: $DEVICE:$GPU_ID"
    echo "  Data type: $DTYPE"
    echo "  Early stopping: $EARLY_STOP"
    echo "  Early stop confidence: $EARLY_STOP_CONFIDENCE"
    echo "  Test best response: $TEST_BEST_RESPONSE"
    echo "  Buffer size: $BUFFER_SIZE"
    echo "  Use mellowmax: $USE_MELLOWMAX"
    echo "  Dynamic confidence: $DYNAMIC_CONFIDENCE"
    if [[ -n "$MAX_ROWS" ]]; then
        echo "  Max rows: $MAX_ROWS (testing mode)"
    fi
    echo ""

    # Build the single-GPU command
    CMD="python optimize_dataset_prompts.py"
    CMD="$CMD --input-csv '$INPUT_CSV'"
    CMD="$CMD --output-csv '$OUTPUT_CSV'"
    CMD="$CMD --model '$MODEL'"
    CMD="$CMD --table-rows $TABLE_ROWS"
    CMD="$CMD --table-cols $TABLE_COLS"
    CMD="$CMD --num-steps $NUM_STEPS"
    CMD="$CMD --device '$DEVICE:$GPU_ID'"
    CMD="$CMD --dtype '$DTYPE'"
    CMD="$CMD --buffer-size $BUFFER_SIZE"
    CMD="$CMD --use-mellowmax $USE_MELLOWMAX"
    CMD="$CMD --dynamic-confidence $DYNAMIC_CONFIDENCE"

    if [[ -n "$MAX_ROWS" ]]; then
        CMD="$CMD --max-rows $MAX_ROWS"
    fi

    if [[ -n "$EARLY_STOP" ]]; then
        CMD="$CMD --early-stop $EARLY_STOP"
    fi

    if [[ -n "$EARLY_STOP_CONFIDENCE" ]]; then
        CMD="$CMD --early-stop-confidence $EARLY_STOP_CONFIDENCE"
    fi

    if [[ -n "$TEST_BEST_RESPONSE" ]]; then
        CMD="$CMD --test-best-response $TEST_BEST_RESPONSE"
    fi

    echo "Starting single-GPU optimization..."
    echo "Command: $CMD"
    echo ""

    # Execute the command
    eval $CMD
fi