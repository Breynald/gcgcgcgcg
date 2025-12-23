#!/bin/bash

# Script to plot perplexity vs token count for multiple optimized prompt files

# Default values
FILES=(
    # "/work/table-fp/nanoGCG-main/assets/optimized_prompts_1.5b_10init.csv"
    "/work/table-fp/nanoGCG-main/assets/optimized_prompts_1.5b.csv"
    "/work/table-fp/nanoGCG-main/assets/optimized_prompts_7b.csv"
    # "/work/table-fp/nanoGCG-main/assets/optimized_prompts_1.5b_20init.csv"
    "/work/table-fp/nanoGCG-main/assets/processed_counterfactual_prompts.csv"
    "/work/table-fp/nanoGCG-main/assets/question_ppl.csv"
)
MODEL="/work/models/openai-community/gpt2"
DEVICE="cuda:5"
OUTPUT_PLOT="multiple_files_perplexity_plot.png"
PLOT_FORMAT="png"
MAX_SAMPLES=1000
LABELS=("1.5b" "7b" "proflingo" "common")
NO_TREND_LINE=("proflingo" "common")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --files)
            shift
            FILES=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                FILES+=("$1")
                shift
            done
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output-plot)
            OUTPUT_PLOT="$2"
            shift 2
            ;;
        --plot-format)
            PLOT_FORMAT="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --labels)
            shift
            LABELS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LABELS+=("$1")
                shift
            done
            ;;
        --no-trend-line)
            shift
            NO_TREND_LINE=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                NO_TREND_LINE+=("$1")
                shift
            done
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --files FILE1 FILE2 ...   CSV files to plot (default: three 1.5b files)"
            echo "  --model PATH              Model path for perplexity calculation (default: /work/models/openai-community/gpt2)"
            echo "  --device DEVICE           Device for calculation (default: cuda:5)"
            echo "  --output-plot FILE        Output plot file (default: multiple_files_perplexity_plot.png)"
            echo "  --plot-format FORMAT      Plot file format: png, pdf, jpg (default: png)"
            echo "  --max-samples NUM         Maximum number of samples per file (default: 1000)"
            echo "  --labels LABEL1 LABEL2 ... Legend labels for files"
            echo "  --no-trend-line LABEL ...  Labels for which NOT to draw trend lines"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --max-samples 500 --device cuda:0 --output-plot custom_plot.png"
            echo "  $0 --no-trend-line proflingo  # Don't draw trend line for proflingo data"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Python script exists
SCRIPT_PATH="plot_multiple_files_perplexity.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_PATH' not found!"
    exit 1
fi

# Check if input files exist
for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Input file '$file' does not exist!"
        exit 1
    fi
done

# Check that labels match files
if [ ${#LABELS[@]} -ne ${#FILES[@]} ]; then
    echo "Error: Number of labels (${#LABELS[@]}) must match number of files (${#FILES[@]})"
    exit 1
fi

# Print configuration
echo "=================================================="
echo "Multiple Files Perplexity Plot"
echo "=================================================="
echo "Files to process:"
for i in "${!FILES[@]}"; do
    echo "  ${LABELS[$i]}: ${FILES[$i]}"
done
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Output plot: $OUTPUT_PLOT"
echo "Plot format: $PLOT_FORMAT"
echo "Max samples per file: $MAX_SAMPLES"
echo "=================================================="

# Run the Python script
echo "Running analysis..."
if [ ${#NO_TREND_LINE[@]} -eq 0 ]; then
    python plot_multiple_files_perplexity.py \
        --files "${FILES[@]}" \
        --model "$MODEL" \
        --device "$DEVICE" \
        --output-plot "$OUTPUT_PLOT" \
        --plot-format "$PLOT_FORMAT" \
        --max-samples "$MAX_SAMPLES" \
        --labels "${LABELS[@]}"
else
    python plot_multiple_files_perplexity.py \
        --files "${FILES[@]}" \
        --model "$MODEL" \
        --device "$DEVICE" \
        --output-plot "$OUTPUT_PLOT" \
        --plot-format "$PLOT_FORMAT" \
        --max-samples "$MAX_SAMPLES" \
        --labels "${LABELS[@]}" \
        --no-trend-line "${NO_TREND_LINE[@]}"
fi

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Analysis completed successfully!"
    echo "=================================================="
    echo "Plot saved to: $OUTPUT_PLOT"
    echo ""
    echo "The plot shows:"
    echo "  - Log Perplexity vs Token Count for each file"
    echo "  - Different colors for different initialization methods"
    echo "  - Trend lines for each dataset"
    echo ""
    echo "You can view the plot with:"
    echo "  open $OUTPUT_PLOT"
    echo "  # or"
    echo "  eog $OUTPUT_PLOT"
    echo ""
else
    echo "Error: Failed to run the analysis!"
    exit 1
fi