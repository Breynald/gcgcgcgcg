#!/bin/bash

# Script to calculate perplexity for different initial optimization lengths and create plots

# Default values
INPUT_DIR="./"
MODEL="/work/models/openai-community/gpt2"
DEVICE="cuda:4"
OUTPUT_CSV="init_length_perplexity_results.csv"
OUTPUT_PLOT="init_length_perplexity_plot.png"
PLOT_FORMAT="png"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
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
        --output-csv)
            OUTPUT_CSV="$2"
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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input-dir DIR     Directory containing optimized prompt CSV files (default: init_length_test_results)"
            echo "  --model PATH        Model path for perplexity calculation (default: /work/models/Qwen/Qwen2.5-1.5B-Instruct)"
            echo "  --device DEVICE     Device for calculation (default: cuda:4)"
            echo "  --output-csv FILE   Output CSV file (default: init_length_perplexity_results.csv)"
            echo "  --output-plot FILE  Output plot file (default: init_length_perplexity_plot.png)"
            echo "  --plot-format FORMAT Plot file format: png, pdf, jpg (default: png)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --model /path/to/model --device cuda:0 --output-plot my_plot.png"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist!"
    exit 1
fi

# Check if Python script exists
SCRIPT_PATH="calculate_and_plot_perplexity.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_PATH' not found!"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Initial Length Perplexity Analysis"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Output CSV: $OUTPUT_CSV"
echo "Output plot: $OUTPUT_PLOT"
echo "Plot format: $PLOT_FORMAT"
echo "=========================================="

# Run the Python script
echo "Running perplexity calculation and plotting..."
python calculate_and_plot_perplexity.py \
    --input-dir "$INPUT_DIR" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --output-csv "$OUTPUT_CSV" \
    --output-plot "$OUTPUT_PLOT" \
    --plot-format "$PLOT_FORMAT"

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Analysis completed successfully!"
    echo "=========================================="
    echo "Results saved to:"
    echo "  - Summary: $OUTPUT_CSV"
    echo "  - Detailed: ${OUTPUT_CSV%.csv}_detailed.csv"
    echo "  - Plot: $OUTPUT_PLOT"
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