#!/bin/bash
# Shell script to run the dataset optimization

# Default parameters
INPUT_CSV="/work/table-fp/nanoGCG-main/assets/question.csv"
OUTPUT_CSV="/work/table-fp/nanoGCG-main/assets/optimized_prompts.csv"
MODEL="/work/models/Qwen/Qwen2.5-1.5B-Instruct"
TABLE_ROWS=1
TABLE_COLS=3
MAX_ROWS=""  # Empty means no limit
NUM_STEPS=1000
DEVICE="cuda"
DTYPE="float16"
GPU_ID="7"
EARLY_STOP_CONFIDENCE="0.3"

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
    echo "  --gpu-id ID             GPU ID to use (default: $GPU_ID)"
    echo "  --dtype DTYPE           Data type (default: $DTYPE)"
    echo "  --early-stop-confidence NUM  Confidence threshold for early stop (0.0-1.0)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings (greedy early stop)"
    echo "  $0"
    echo ""
    echo "  # Test with only 5 questions and 1x1 table"
    echo "  $0 --max-rows 5 --table-rows 1 --table-cols 1"
    echo ""
    echo "  # Use confidence threshold for early stop (90% confidence)"
    echo "  $0 --early-stop-confidence 0.9"
    echo ""
    echo "  # Use higher confidence threshold (95% confidence)"
    echo "  $0 --early-stop-confidence 0.95 --table-rows 2 --table-cols 2"
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
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --early-stop-confidence)
            EARLY_STOP_CONFIDENCE="$2"
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

# Build the command
CMD="python optimize_dataset_prompts.py"
CMD="$CMD --input-csv '$INPUT_CSV'"
CMD="$CMD --output-csv '$OUTPUT_CSV'"
CMD="$CMD --model '$MODEL'"
CMD="$CMD --table-rows $TABLE_ROWS"
CMD="$CMD --table-cols $TABLE_COLS"
CMD="$CMD --num-steps $NUM_STEPS"
CMD="$CMD --device '$DEVICE:$GPU_ID'"
CMD="$CMD --dtype '$DTYPE'"

if [[ -n "$MAX_ROWS" ]]; then
    CMD="$CMD --max-rows $MAX_ROWS"
fi

if [[ -n "$EARLY_STOP_CONFIDENCE" ]]; then
    CMD="$CMD --early-stop-confidence $EARLY_STOP_CONFIDENCE"
fi

echo "Running optimization with the following settings:"
echo "  Input CSV: $INPUT_CSV"
echo "  Output CSV: $OUTPUT_CSV"
echo "  Model: $MODEL"
echo "  Table size: ${TABLE_ROWS}x${TABLE_COLS}"
echo "  Optimization steps: $NUM_STEPS"
echo "  Device: $DEVICE:$GPU_ID"
echo "  Data type: $DTYPE"
if [[ -n "$MAX_ROWS" ]]; then
    echo "  Max rows: $MAX_ROWS (testing mode)"
fi
if [[ -n "$EARLY_STOP_CONFIDENCE" ]]; then
    echo "  Early stop confidence: $EARLY_STOP_CONFIDENCE"
else
    echo "  Early stop confidence: disabled (greedy match only)"
fi
echo ""

# Execute the command
eval $CMD