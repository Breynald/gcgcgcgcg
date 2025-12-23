#!/bin/bash

# Script to analyze relationship between perplexity and token count for optimized prompts

# Default values
INPUT_CSV="/work/table-fp/nanoGCG-main/assets/optimized_prompts_1.5b_10init.csv"
MODEL="/work/models/openai-community/gpt2"
DEVICE="cuda:5"
OUTPUT_CSV="prompt_perplexity_token_analysis.csv"
OUTPUT_PLOT="prompt_perplexity_token_plot.png"
PLOT_FORMAT="png"
PROMPT_COLUMN="full_optimized_prompt"
MAX_SAMPLES=1000

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
        --prompt-column)
            PROMPT_COLUMN="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input-csv FILE     CSV file with optimized prompts (default: /work/table-fp/nanoGCG-main/assets/optimized_prompts_1.5b.csv)"
            echo "  --model PATH        Model path for perplexity calculation (default: /work/models/Qwen/Qwen2.5-1.5B-Instruct)"
            echo "  --device DEVICE     Device for calculation (default: cuda:4)"
            echo "  --output-csv FILE   Output CSV file (default: prompt_perplexity_token_analysis.csv)"
            echo "  --output-plot FILE  Output plot file (default: prompt_perplexity_token_plot.png)"
            echo "  --plot-format FORMAT Plot file format: png, pdf, jpg (default: png)"
            echo "  --prompt-column COL Column name containing prompts (default: prompt)"
            echo "  --max-samples NUM   Maximum number of samples to process (default: 1000)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --max-samples 500 --device cuda:0 --output-plot custom_plot.png"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if input file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input file '$INPUT_CSV' does not exist!"
    exit 1
fi

# Check if Python script exists
SCRIPT_PATH="analyze_prompt_perplexity_tokens.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_PATH' not found!"
    exit 1
fi

# Print configuration
echo "=================================================="
echo "Prompt Perplexity vs Token Count Analysis"
echo "=================================================="
echo "Input CSV: $INPUT_CSV"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Output CSV: $OUTPUT_CSV"
echo "Output plot: $OUTPUT_PLOT"
echo "Plot format: $PLOT_FORMAT"
echo "Prompt column: $PROMPT_COLUMN"
echo "Max samples: $MAX_SAMPLES"
echo "=================================================="

# Run the Python script
echo "Running analysis..."
python analyze_prompt_perplexity_tokens.py \
    --input-csv "$INPUT_CSV" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --output-csv "$OUTPUT_CSV" \
    --output-plot "$OUTPUT_PLOT" \
    --plot-format "$PLOT_FORMAT" \
    --prompt-column "$PROMPT_COLUMN" \
    --max-samples "$MAX_SAMPLES"

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Analysis completed successfully!"
    echo "=================================================="
    echo "Results saved to:"
    echo "  - Detailed analysis: $OUTPUT_CSV"
    echo "  - Plots: $OUTPUT_PLOT"
    echo ""
    echo "The plot shows:"
    echo "  Log Perplexity vs Token Count (natural log, with trend line)"
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