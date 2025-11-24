#!/bin/bash
# Shell script to test optimized prompts

# Default parameters
INPUT_CSV="/work/table-fp/nanoGCG-main/assets/optimized_selected_prompts_2.csv"
MODEL="/work/models/Qwen/Qwen2.5-1.5B-Instruct"
DEVICE="cuda:6"
DTYPE="float16"
MAX_ROWS=""
MAX_NEW_TOKENS=""

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --input-csv PATH        Input CSV with optimized prompts (default: $INPUT_CSV)"
    echo "  --model MODEL           Model to use (default: $MODEL)"
    echo "  --device DEVICE         Device to use (default: $DEVICE)"
    echo "  --dtype DTYPE           Data type (default: $DTYPE)"
    echo "  --max-rows N            Maximum prompts to test"
    echo "  --max-new-tokens N      Maximum new tokens to generate (auto-calculated if not specified)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Test all prompts with default settings"
    echo "  $0"
    echo ""
    echo "  # Test only first 3 prompts"
    echo "  $0 --max-rows 3"
    echo ""
    echo "  # Test with different device"
    echo "  $0 --device cuda:0"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-csv)
            INPUT_CSV="$2"
            shift 2
            ;;
                      --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --max-rows)
            MAX_ROWS="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
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

# Check if input file exists
if [[ ! -f "$INPUT_CSV" ]]; then
    echo "Error: Input CSV file '$INPUT_CSV' not found!"
    echo "Please run the optimization script first to generate optimized prompts."
    exit 1
fi

echo "Testing optimized prompts with the following settings:"
echo "  Input CSV: $INPUT_CSV"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo "  Data type: $DTYPE"
if [[ -n "$MAX_NEW_TOKENS" ]]; then
    echo "  Max new tokens: $MAX_NEW_TOKENS"
else
    echo "  Max new tokens: auto-calculated based on target length"
fi
if [[ -n "$MAX_ROWS" ]]; then
    echo "  Max rows: $MAX_ROWS (testing mode)"
fi
echo ""

# Build the command
CMD="python test_dataset_prompts.py"
CMD="$CMD --input-csv '$INPUT_CSV'"
CMD="$CMD --model '$MODEL'"
CMD="$CMD --device '$DEVICE'"
CMD="$CMD --dtype '$DTYPE'"

if [[ -n "$MAX_NEW_TOKENS" ]]; then
    CMD="$CMD --max-new-tokens $MAX_NEW_TOKENS"
fi

if [[ -n "$MAX_ROWS" ]]; then
    CMD="$CMD --max-rows $MAX_ROWS"
fi

# Execute the command
eval $CMD