# 批量优化脚本使用说明

本目录包含用于批量运行GCG优化的脚本，支持从1×1到9×9的表格以及simple prompt的优化，并计算困惑度生成热力图。

## 目录结构

```
nanogcg/
├── tools/                          # 工具模块包
│   ├── __init__.py                 # 包初始化文件
│   ├── optimization_utils.py       # 优化相关的工具函数
│   └── analysis_utils.py           # 分析和可视化工具函数
└── scripts/                        # 可执行脚本
    ├── batch_optimization.py       # 主要的批量优化脚本
    ├── test_batch_optimization.py  # 测试脚本
    └── README.md                   # 本说明文档
```

## 模块化设计

### nanogcg/tools 包

#### optimization_utils.py
包含优化的核心功能：
- `generate_table_prompt(rows, cols)`: 生成指定大小的表格prompt
- `generate_simple_prompt()`: 生成simple prompt
- `load_models()`: 加载主模型和困惑度计算模型
- `create_config()`: 创建GCG配置
- `setup_probe_sampling()`: 设置probe sampling
- `run_single_optimization()`: 运行单次优化

#### analysis_utils.py
包含结果分析和可视化功能：
- `calculate_perplexity_for_prompt()`: 计算prompt的困惑度
- `create_heatmap()`: 生成热力图
- `save_results()`: 保存结果到JSON文件
- `create_summary_report()`: 生成详细报告
- `create_loss_plots()`: 创建loss优化动态图
- `calculate_statistics()`: 计算统计信息
- `clear_gpu_cache()`: 清理GPU缓存

## 使用方法

### 1. 运行完整批量优化

```bash
# 进入scripts目录
cd /work/table-fp/nanoGCG-main/scripts

# 基本用法
python batch_optimization.py

# 自定义参数
python batch_optimization.py \
    --model "mistralai/Mistral-7B-Instruct-v0.3" \
    --target "Copyright" \
    --max-table-size 5 \
    --num-steps 300 \
    --probe-sampling

# 使用CPU
python batch_optimization.py \
    --device "cpu" \
    --dtype "float32"
```

### 2. 主要参数说明

- `--model`: 主优化模型（默认: mistralai/Mistral-7B-Instruct-v0.3）
- `--target`: 优化目标词（默认: Copyright）
- `--max-table-size`: 最大表格大小（默认: 9，即1×1到9×9）
- `--num-steps`: 优化步数（默认: 500）
- `--probe-sampling`: 启用probe sampling加速
- `--device`: 运行设备（cuda/cpu）
- `--dtype`: 数据类型（float16/float32）
- `--output-dir`: 结果输出目录
- `--perplexity-model`: 困惑度计算模型（默认: gpt2）

### 3. 输出结果

运行后会在指定的输出目录中生成：

```
batch_results/batch_YYYYMMDD_HHMMSS/
├── optimization_results.json    # 完整的优化结果
├── summary_report.txt           # 详细报告
├── perplexity_heatmap.png       # 困惑度热力图
└── loss_plots/                  # loss动态图目录
    ├── loss_plot_simple.png
    ├── loss_plot_table_1x1.png
    ├── loss_plot_table_1x2.png
    └── ...
```

### 4. 测试脚本

运行测试验证功能：

```bash
# 只运行单元测试
python test_batch_optimization.py --skip-integration

# 运行完整测试（包括模型加载）
python test_batch_optimization.py
```

## 示例输出

### 控制台输出示例
```
Batch optimization started at: 2024-01-15 10:30:00
Output directory: batch_results/batch_20240115_103000
Configuration:
  Model: mistralai/Mistral-7B-Instruct-v0.3
  Target: Copyright
  Max table size: 3x3
  Optimization steps: 100
  ...

============================================================
OPTIMIZING SIMPLE PROMPT
============================================================
Optimizing: Simple Prompt
Number of placeholders: 1
Simple prompt perplexity: 15.23

============================================================
OPTIMIZING TABLE PROMPTS
============================================================
1x1 perplexity: 14.89
1x2 perplexity: 16.45
...

BATCH OPTIMIZATION COMPLETED
============================================================
Results saved to: batch_results/batch_20240115_103000
Simple prompt perplexity: 15.23
Table optimization success rate: 100.0% (9/9)
Best table performance: table_1x1 (PPL: 14.89)
```

### 热力图示例
会生成一个9×9的热力图，显示不同表格大小的困惑度，颜色越蓝表示困惑度越低（效果越好）。

## 代码使用示例

### 直接使用工具模块

```python
from nanogcg.tools import (
    generate_table_prompt, create_config, run_single_optimization,
    calculate_perplexity_for_prompt, create_heatmap
)

# 生成表格prompt
table_data = generate_table_prompt(3, 3)
print(f"Generated {len(table_data['optim_str_placeholders'])} placeholders")

# 创建配置
config = create_config(num_steps=100, optim_str_init="test test")

# 运行优化（需要先加载模型）
# result = run_single_optimization(model, tokenizer, messages, target, config, placeholders)
```

## 注意事项

1. **内存管理**: 脚本会定期清理GPU缓存，但对于大模型仍需注意显存使用
2. **运行时间**: 完整的9×9优化可能需要较长时间，建议先用小规模测试
3. **模型兼容性**: 确保指定的模型与当前transformers版本兼容
4. **结果保存**: 所有结果都会自动保存，支持中断后继续分析

## 扩展使用

### 自定义表格prompt格式
修改`nanogcg/tools/optimization_utils.py`中的`generate_table_prompt()`函数来自定义表格格式。

### 添加新的评估指标
在`nanogcg/tools/analysis_utils.py`中添加新的分析函数，并在主脚本中调用。

### 调整优化策略
通过修改`create_config()`参数或实现新的配置函数来调整优化策略。