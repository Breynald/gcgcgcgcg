#!/bin/bash

# Script to process counterfactual prompts

# Default values
INPUT="/work/table-fp/nanoGCG-main/assets/counterfactual_base_finance-64.csv"
OUTPUT="processed_counterfactual_prompts.csv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input FILE     Input CSV file with counterfactual data (default: /work/table-fp/nanoGCG-main/assets/counterfactual_base_finance-64.csv)"
            echo "  --output FILE    Output CSV file with processed prompts (default: processed_counterfactual_prompts.csv)"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --input custom_data.csv --output processed_prompts.csv"
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
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' does not exist!"
    exit 1
fi

# Check if Python script exists
SCRIPT_PATH="process_counterfactual_prompts.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_PATH' not found!"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Processing Counterfactual Prompts"
echo "=========================================="
echo "Input file: $INPUT"
echo "Output file: $OUTPUT"
echo "=========================================="

# Run the Python script
python process_counterfactual_prompts.py --input "$INPUT" --output "$OUTPUT"

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Processing completed successfully!"
    echo "=========================================="
    echo "Processed prompts saved to: $OUTPUT"
    echo ""
    echo "You can now analyze these prompts with:"
    echo "  python ../analyze_prompt_perplexity_tokens.py --input-csv $OUTPUT --output-csv counterfactual_ppl_analysis.csv --output-plot counterfactual_ppl_plot.png"
else
    echo "Error: Failed to process the file!"
    exit 1
fi