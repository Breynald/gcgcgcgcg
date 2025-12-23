#!/bin/bash

# Script to plot perplexity distribution for multiple optimized prompt files

# Default values
FILES=(
    "/work/table-fp/nanoGCG-main/assets/optimized_prompts_1.5b.csv"
    "/work/table-fp/nanoGCG-main/assets/optimized_prompts_7b.csv"
    "/work/table-fp/nanoGCG-main/assets/processed_counterfactual_prompts.csv"
    "/work/table-fp/nanoGCG-main/assets/question_ppl.csv"
)
MODEL="/work/models/openai-community/gpt2"
DEVICE="cuda:5"
OUTPUT_PLOT="multiple_files_distribution_plot.png"
PLOT_FORMAT="png"
MAX_SAMPLES=1000
LABELS=("1.5b" "7b" "proflingo" "common")
PROMPT_COLUMN="full_optimized_prompt"
PLOT_TYPE="swarm"

# violin swarm bar

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
        --prompt-column)
            PROMPT_COLUMN="$2"
            shift 2
            ;;
        --plot-type)
            PLOT_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --files FILE1 FILE2 ...   CSV files to plot"
            echo "  --model PATH              Model path for perplexity calculation (default: /work/models/openai-community/gpt2)"
            echo "  --device DEVICE           Device for calculation (default: cuda:5)"
            echo "  --output-plot FILE        Output plot file (default: multiple_files_distribution_plot.png)"
            echo "  --plot-format FORMAT      Plot file format: png, pdf, jpg (default: png)"
            echo "  --max-samples NUM         Maximum number of samples per file (default: 1000)"
            echo "  --labels LABEL1 LABEL2 ... X-axis labels for files"
            echo "  --prompt-column COL       Column name containing prompts (default: full_optimized_prompt)"
            echo "  --plot-type TYPE          Type of plot: box, violin, swarm, bar (default: box)"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Plot Types:"
            echo "  box     - Box plot showing quartiles and outliers"
            echo "  violin  - Violin plot showing kernel density estimate"
            echo "  swarm   - Swarm plot showing individual data points"
            echo "  bar     - Bar plot showing mean with error bars"
            echo ""
            echo "Example:"
            echo "  $0 --plot-type violin --max-samples 500"
            echo "  $0 --plot-type bar --labels \"1.5B Model\" \"7B Model\""
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
SCRIPT_PATH="plot_multiple_files_distribution.py"
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

# Check that plot type is valid
if [[ ! "$PLOT_TYPE" =~ ^(box|violin|swarm|bar)$ ]]; then
    echo "Error: Plot type must be one of: box, violin, swarm, bar"
    exit 1
fi

# Print configuration
echo "=================================================="
echo "Multiple Files Perplexity Distribution Plot"
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
echo "Prompt column: $PROMPT_COLUMN"
echo "Plot type: $PLOT_TYPE"
echo "=================================================="

# Run the Python script
echo "Running analysis..."
python plot_multiple_files_distribution.py \
    --files "${FILES[@]}" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --output-plot "$OUTPUT_PLOT" \
    --plot-format "$PLOT_FORMAT" \
    --max-samples "$MAX_SAMPLES" \
    --labels "${LABELS[@]}" \
    --prompt-column "$PROMPT_COLUMN" \
    --plot-type "$PLOT_TYPE"

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Analysis completed successfully!"
    echo "=================================================="
    echo "Plot saved to: $OUTPUT_PLOT"
    echo ""
    echo "The plot shows:"
    echo "  - Distribution of log perplexity for each dataset"
    echo "  - X-axis: Different datasets/models"
    echo "  - Y-axis: Log perplexity values"
    echo "  - Plot type: $PLOT_TYPE"
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