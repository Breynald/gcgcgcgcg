#!/bin/bash
# Shell script to run iterative optimization until all samples succeed

# Default parameters
INITIAL_CSV="../assets/question.csv"
MAIN_OUTPUT_CSV="../assets/optimized_prompts_1.5b_10init.csv"
ITERATIVE_CSV="../assets/optimized_prompts_1.5b_10init_2.csv"
FAILED_CSV="../assets/question2.csv"
MODEL="/work/models/Qwen/Qwen2.5-1.5B"
DEVICE="cuda:4"
DTYPE="float16"
TABLE_ROWS=1
TABLE_COLS=3
MAX_ROWS=""  # Empty means no limit
NUM_STEPS=1500
GPU_IDS="2,4,5"  # Single GPU by default
EARLY_STOP="True"
EARLY_STOP_CONFIDENCE=""
EARLY_STOP_LOSS_THRESHOLD="0.05"
DYNAMIC_CONFIDENCE="True"
TEST_BEST_RESPONSE="True"
BUFFER_SIZE="3"
USE_MELLOWMAX="False"
OPTIM_STR_INIT="x x x x x x x x x x x x x x "  # Empty means use default initial optimization text
MAX_ITERATIONS=10  # Maximum number of optimization iterations to prevent infinite loops

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Runs iterative optimization until all samples succeed or max iterations reached."
    echo ""
    echo "Options:"
    echo "  --initial-csv PATH       Initial questions CSV (default: $INITIAL_CSV)"
    echo "  --main-output-csv PATH   Main output CSV for final results (default: $MAIN_OUTPUT_CSV)"
    echo "  --iterative-csv PATH     Iterative optimization output CSV (default: $ITERATIVE_CSV)"
    echo "  --failed-csv PATH        CSV for failed questions (default: $FAILED_CSV)"
    echo "  --model MODEL            Model to use (default: $MODEL)"
    echo "  --device DEVICE          Device to use (default: $DEVICE)"
    echo "  --dtype DTYPE            Data type (default: $DTYPE)"
    echo "  --table-rows N           Number of table rows (default: $TABLE_ROWS)"
    echo "  --table-cols N           Number of table columns (default: $TABLE_COLS)"
    echo "  --max-rows N             Maximum questions to optimize (for testing)"
    echo "  --num-steps N            Number of optimization steps (default: $NUM_STEPS)"
    echo "  --gpu-ids IDS            Comma-separated GPU IDs for parallel processing"
    echo "  --early-stop BOOL        Enable early stopping (default: $EARLY_STOP)"
    echo "  --early-stop-confidence NUM  Confidence threshold for early stop"
    echo "  --early-stop-loss-threshold NUM  Loss threshold for early stop (default: $EARLY_STOP_LOSS_THRESHOLD)"
    echo "  --test-best-response BOOL Test current best response during optimization (default: $TEST_BEST_RESPONSE)"
    echo "  --buffer-size N          Buffer size for optimization (default: $BUFFER_SIZE)"
    echo "  --use-mellowmax BOOL     Use mellowmax loss (default: $USE_MELLOWMAX)"
    echo "  --dynamic-confidence BOOL Use dynamic confidence (default: $DYNAMIC_CONFIDENCE)"
    echo "  --optim-str-init STR     Initial optimization string (default: use default)"
    echo "  --max-iterations N       Maximum optimization iterations (default: $MAX_ITERATIONS)"
    echo "  --help                   Show this help message"
    echo ""
    echo "Process Flow:"
    echo "  1. If main output doesn't exist, run initial optimization"
    echo "  2. Test current main results and extract failed questions"
    echo "  3. If no failures, done!"
    echo "  4. Run iterative optimization on failed questions"
    echo "  5. Merge successful results back to main file"
    echo "  6. Repeat from step 2 until all succeed or max iterations reached"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --initial-csv)
            INITIAL_CSV="$2"
            shift 2
            ;;
        --main-output-csv)
            MAIN_OUTPUT_CSV="$2"
            shift 2
            ;;
        --iterative-csv)
            ITERATIVE_CSV="$2"
            shift 2
            ;;
        --failed-csv)
            FAILED_CSV="$2"
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
        --gpu-ids)
            GPU_IDS="$2"
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
        --early-stop-loss-threshold)
            EARLY_STOP_LOSS_THRESHOLD="$2"
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
        --optim-str-init)
            OPTIM_STR_INIT="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
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

echo "Iterative Optimization Pipeline"
echo "=============================="
echo "Initial CSV: $INITIAL_CSV"
echo "Main Output CSV: $MAIN_OUTPUT_CSV"
echo "Iterative CSV: $ITERATIVE_CSV"
echo "Failed Questions CSV: $FAILED_CSV"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Table size: ${TABLE_ROWS}x${TABLE_COLS}"
echo "Optimization steps: $NUM_STEPS"
echo "Max iterations: $MAX_ITERATIONS"
if [[ -n "$GPU_IDS" ]]; then
    echo "GPU IDs: $GPU_IDS (parallel mode)"
else
    echo "GPU ID: $DEVICE (single GPU mode)"
fi
echo ""

# Function to run optimization
run_optimization() {
    local input_csv="$1"
    local output_csv="$2"
    local description="$3"

    echo ""
    echo "[$description] Running optimization..."
    echo "Input: $input_csv"
    echo "Output: $output_csv"
    echo ""

    # Build optimization command
    OPT_CMD="./run_dataset_optimization.sh"
    OPT_CMD="$OPT_CMD --input-csv '$input_csv'"
    OPT_CMD="$OPT_CMD --output-csv '$output_csv'"
    OPT_CMD="$OPT_CMD --model '$MODEL'"
    OPT_CMD="$OPT_CMD --table-rows $TABLE_ROWS"
    OPT_CMD="$OPT_CMD --table-cols $TABLE_COLS"
    OPT_CMD="$OPT_CMD --num-steps $NUM_STEPS"
    OPT_CMD="$OPT_CMD --dtype '$DTYPE'"
    OPT_CMD="$OPT_CMD --early-stop $EARLY_STOP"
    OPT_CMD="$OPT_CMD --test-best-response $TEST_BEST_RESPONSE"
    OPT_CMD="$OPT_CMD --buffer-size $BUFFER_SIZE"
    OPT_CMD="$OPT_CMD --use-mellowmax $USE_MELLOWMAX"
    OPT_CMD="$OPT_CMD --dynamic-confidence $DYNAMIC_CONFIDENCE"
    OPT_CMD="$OPT_CMD --early-stop-loss-threshold $EARLY_STOP_LOSS_THRESHOLD"

    if [[ -n "$OPTIM_STR_INIT" ]]; then
        OPT_CMD="$OPT_CMD --optim-str-init '$OPTIM_STR_INIT'"
    fi

    if [[ -n "$MAX_ROWS" ]]; then
        OPT_CMD="$OPT_CMD --max-rows $MAX_ROWS"
    fi

    if [[ -n "$EARLY_STOP_CONFIDENCE" ]]; then
        OPT_CMD="$OPT_CMD --early-stop-confidence $EARLY_STOP_CONFIDENCE"
    fi

    if [[ -n "$GPU_IDS" ]]; then
        OPT_CMD="$OPT_CMD --gpu-ids '$GPU_IDS'"
    else
        # Extract GPU ID from device (e.g., "cuda:4" -> "4")
        GPU_ID=$(echo $DEVICE | cut -d: -f2)
        OPT_CMD="$OPT_CMD --gpu-id $GPU_ID"
    fi

    echo "Command: $OPT_CMD"
    eval $OPT_CMD

    if [[ $? -ne 0 ]]; then
        echo "Error: Optimization failed!"
        exit 1
    fi

    echo "✓ Optimization completed successfully"
}

# Function to test and extract failures
test_and_extract_failures() {
    local input_csv="$1"
    local output_csv="$2"

    echo ""
    echo "[Testing] Testing optimized prompts and extracting failures..."
    echo "Input: $input_csv"
    echo "Output: $output_csv"
    echo ""

    # Build extraction command
    EXTRACT_CMD="./extract_failed_questions.sh"
    EXTRACT_CMD="$EXTRACT_CMD --input-csv '$input_csv'"
    EXTRACT_CMD="$EXTRACT_CMD --output-csv '$output_csv'"
    EXTRACT_CMD="$EXTRACT_CMD --model '$MODEL'"
    EXTRACT_CMD="$EXTRACT_CMD --device '$DEVICE'"
    EXTRACT_CMD="$EXTRACT_CMD --dtype '$DTYPE'"

    if [[ -n "$MAX_ROWS" ]]; then
        EXTRACT_CMD="$EXTRACT_CMD --max-rows $MAX_ROWS"
    fi

    echo "Command: $EXTRACT_CMD"
    eval $EXTRACT_CMD

    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to extract failures!"
        exit 1
    fi

    # Check if there are any failures
    if [[ -f "$output_csv" ]]; then
        failure_count=$(wc -l < "$output_csv")
        # Subtract 1 for header row
        failure_count=$((failure_count - 1))
        echo "✓ Found $failure_count failed samples"
        return $failure_count
    else
        echo "✓ No failures found (output file not created)"
        return 0
    fi
}

# Function to merge successful results
merge_successful() {
    local main_file="$1"
    local iterative_file="$2"

    echo ""
    echo "[Merging] Merging successful iterative results back to main file..."
    echo "Main file: $main_file"
    echo "Iterative file: $iterative_file"
    echo ""

    # Build merge command
    MERGE_CMD="python merge_successful_prompts.py"
    MERGE_CMD="$MERGE_CMD --main-file '$main_file'"
    MERGE_CMD="$MERGE_CMD --iterative-file '$iterative_file'"
    MERGE_CMD="$MERGE_CMD --output-file '$main_file'"
    MERGE_CMD="$MERGE_CMD --model '$MODEL'"
    MERGE_CMD="$MERGE_CMD --device '$DEVICE'"
    MERGE_CMD="$MERGE_CMD --dtype '$DTYPE'"

    echo "Command: $MERGE_CMD"
    eval $MERGE_CMD

    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to merge results!"
        exit 1
    fi

    echo "✓ Merge completed successfully"
}

# Main iterative optimization loop

# Step 1: Check if we need to run initial optimization
if [[ ! -f "$MAIN_OUTPUT_CSV" ]]; then
    echo "Main output file not found. Running initial optimization..."
    run_optimization "$INITIAL_CSV" "$MAIN_OUTPUT_CSV" "Initial"
else
    echo "✓ Main output file exists: $MAIN_OUTPUT_CSV"
    echo "Proceeding directly to testing and iterative optimization..."
fi

# Step 2: Iterative optimization loop
iteration=1
while [[ $iteration -le $MAX_ITERATIONS ]]; do
    echo ""
    echo "=" * 60
    echo "ITERATION $iteration/$MAX_ITERATIONS"
    echo "=" * 60

    # Test current results and extract failures
    test_and_extract_failures "$MAIN_OUTPUT_CSV" "$FAILED_CSV"
    failure_count=$?

    if [[ $failure_count -eq 0 ]]; then
        echo ""
        echo "🎉 SUCCESS: All samples are working! No more failures to fix."
        echo "Final results saved to: $MAIN_OUTPUT_CSV"
        exit 0
    fi

    echo "Found $failure_count failed samples. Running iterative optimization..."

    # Run optimization on failed samples
    run_optimization "$FAILED_CSV" "$ITERATIVE_CSV" "Iteration $iteration"

    # Merge successful results back to main file
    merge_successful "$MAIN_OUTPUT_CSV" "$ITERATIVE_CSV"

    iteration=$((iteration + 1))
done

echo ""
echo "⚠️  WARNING: Reached maximum iterations ($MAX_ITERATIONS) without complete success."
echo "Some samples may still be failing. Check the final results in: $MAIN_OUTPUT_CSV"
echo "The last set of failures can be found in: $FAILED_CSV"