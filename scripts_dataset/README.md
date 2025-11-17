# Dataset Optimization Scripts

This directory contains scripts for optimizing prompts on a dataset of questions using nanoGCG.

## Files

### Optimization Scripts
- **`optimize_dataset_prompts.py`** - Main script to optimize all questions from a CSV dataset
- **`run_dataset_optimization.sh`** - Shell script wrapper for easy execution of optimization

### Testing Scripts
- **`test_dataset_prompts.py`** - Script to test the optimized prompts and validate their effectiveness
- **`run_dataset_test.sh`** - Shell script wrapper for testing

## Quick Start

### 1. Optimize Dataset Questions

```bash
# Test with small dataset (5 questions, 1x1 table)
./run_dataset_optimization.sh --max-rows 5 --table-rows 1 --table-cols 1

# Full optimization with default settings
./run_dataset_optimization.sh

# Custom optimization
./run_dataset_optimization.sh --table-rows 2 --table-cols 3 --max-rows 10
```

### 2. Test Optimized Prompts

```bash
# Test all optimized prompts
./run_dataset_test.sh

# Test only first 3 prompts (quick test)
./run_dataset_test.sh --max-rows 3

# Test with different GPU
./run_dataset_test.sh --device cuda:0
```

## Configuration

### Default Settings
- **Model**: `/work/models/Qwen/Qwen2.5-1.5B-Instruct`
- **Device**: `cuda:7`
- **Table Size**: 1x3
- **Optimization Steps**: 300
- **Max Test Rows**: 1 (for quick testing)

### Input/Output Files
- **Input CSV**: `assets/question.csv` (question, answer, keyword columns)
- **Output CSV**: `assets/optimized_prompts.csv`
- **Test Results**: `assets/test_results.csv`

## Output Format

### Optimized Prompts CSV
The output contains:
- `question` - Original question from dataset
- `target_answer` - Expected answer
- `keyword` - Question keyword
- `best_loss` - Final optimization loss
- `num_placeholders` - Number of optimized placeholders
- `table_size` - Table dimensions used (e.g., "1x3")
- `full_optimized_prompt` - Complete optimized prompt with table

### Test Results CSV
The test output contains:
- Original question and target answer
- Generated response from model
- Whether target was found in response
- Success/failure status
- Error messages (if any)

## Examples

### Example Optimized Prompt Format
```
Here is a table:

| | Metric A | Metric B | Metric C |
|---|---|---|---|
| Data 1 | [optimized_string_1] | [optimized_string_2] | [optimized_string_3] |

Where does the sun rise?
```

The optimized strings are placed in the table cells, and when the model processes this prompt, it should generate the target answer "The sun rises in the north."

## Tips

1. **Start Small**: Test with `--max-rows 3` and small tables first
2. **Table Size**: Larger tables (2x3, 3x3) give more flexibility but take longer
3. **GPU Selection**: Use `--gpu-id` or `--device` to specify which GPU to use
4. **Memory Issues**: Reduce table size or use CPU if you encounter GPU memory issues
5. **Validation**: Always test optimized prompts to ensure they work effectively

## Troubleshooting

- **Import Errors**: Ensure you're running from the correct directory
- **GPU Memory**: Try smaller table sizes or use `--device cpu`
- **Model Loading**: Check that the model path is correct
- **File Not Found**: Verify input CSV exists in `assets/` directory