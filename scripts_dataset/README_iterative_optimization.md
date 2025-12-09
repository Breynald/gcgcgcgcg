# 迭代优化流程使用指南

本文档介绍如何使用新的迭代优化流程来自动化处理数据集中失败的样本，直到所有样本都成功。

## 流程概述

迭代优化流程包含以下步骤：

1. **初始优化**（如果主结果文件不存在）
   - 对完整的 `question.csv` 进行优化
   - 生成 `optimized_prompts_1.5b.csv`

2. **测试和提取失败样本**
   - 测试当前的优化结果
   - 提取失败的样本到 `question2.csv`

3. **迭代优化失败样本**
   - 对失败的样本重新优化
   - 生成 `optimized_prompts_1.5b_2.csv`

4. **合并成功结果**
   - 将迭代优化中成功的样本合并回主结果文件
   - 更新 `optimized_prompts_1.5b.csv`

5. **重复 2-4 直到所有样本成功**

## 使用方法

### 基本用法

```bash
cd scripts_dataset
./run_iterative_optimization.sh
```

这将使用默认设置运行完整的迭代优化流程，直到所有样本成功或达到最大迭代次数。

### 自定义参数

```bash
./run_iterative_optimization.sh \
    --model /path/to/your/model \
    --device cuda:0 \
    --table-rows 2 \
    --table-cols 3 \
    --num-steps 1000 \
    --max-iterations 5
```

### 主要参数说明

- `--initial-csv`: 初始问题文件（默认: `../assets/question.csv`）
- `--main-output-csv`: 主结果输出文件（默认: `../assets/optimized_prompts_1.5b.csv`）
- `--iterative-csv`: 迭代优化结果文件（默认: `../assets/optimized_prompts_1.5b_2.csv`）
- `--failed-csv`: 失败样本文件（默认: `../assets/question2.csv`）
- `--model`: 使用的模型（默认: `/work/models/Qwen/Qwen2.5-1.5B`）
- `--device`: 使用的设备（默认: `cuda:4`）
- `--table-rows/columns`: 表格的行列数
- `--num-steps`: 优化步数（默认: 1500）
- `--max-iterations`: 最大迭代次数（默认: 10）
- `--gpu-ids`: 多GPU并行处理，如 `"0,1,2,3"`

## 单独使用各个组件

如果你想手动控制流程的每个步骤，也可以单独使用各个脚本：

### 1. 运行优化

```bash
# 初始优化
./run_dataset_optimization.sh --input-csv ../assets/question.csv --output-csv ../assets/optimized_prompts_1.5b.csv

# 迭代优化失败样本
./run_dataset_optimization.sh --input-csv ../assets/question2.csv --output-csv ../assets/optimized_prompts_1.5b_2.csv
```

### 2. 测试和提取失败样本

```bash
./extract_failed_questions.sh --input-csv ../assets/optimized_prompts_1.5b.csv --output-csv ../assets/question2.csv
```

### 3. 测试优化结果

```bash
./run_dataset_test.sh --input-csv ../assets/optimized_prompts_1.5b.csv --max-rows 10
```

### 4. 合并成功结果

```bash
python merge_successful_prompts.py \
    --main-file ../assets/optimized_prompts_1.5b.csv \
    --iterative-file ../assets/optimized_prompts_1.5b_2.csv \
    --output-file ../assets/optimized_prompts_1.5b.csv \
    --model /work/models/Qwen/Qwen2.5-1.5B \
    --device cuda:4
```

## 示例场景

### 场景1: 完整自动化流程

第一次处理新的数据集：

```bash
# 使用默认设置，完整自动化处理
./run_iterative_optimization.sh

# 或者自定义设置
./run_iterative_optimization.sh \
    --max-iterations 5 \
    --num-steps 1000 \
    --early-stop True \
    --early-stop-confidence 0.9
```

### 场景2: 从已有的优化结果继续

如果已经有 `optimized_prompts_1.5b.csv`，只需要处理失败样本：

```bash
# 脚本会自动检测已有文件，跳过初始优化
./run_iterative_optimization.sh
```

### 场景3: 使用多GPU加速

```bash
# 使用4个GPU并行处理
./run_iterative_optimization.sh \
    --gpu-ids "0,1,2,3" \
    --num-steps 2000
```

### 场景4: 测试模式（只处理少量样本）

```bash
# 只处理前5个样本，用于快速测试
./run_iterative_optimization.sh \
    --max-rows 5 \
    --table-rows 1 \
    --table-cols 1 \
    --num-steps 500
```

## 输出文件说明

- `optimized_prompts_1.5b.csv`: 最终的主结果文件，包含所有成功优化的样本
- `optimized_prompts_1.5b_2.csv`: 迭代优化的结果文件（临时文件）
- `question2.csv: 包含当前迭代中失败的样本
- 可能的临时文件: 在多GPU模式下可能会产生临时分割文件

## 注意事项

1. **磁盘空间**: 确保有足够的磁盘空间存储中间结果
2. **GPU内存**: 根据GPU内存调整批处理大小和模型设置
3. **时间**: 完整的迭代优化可能需要很长时间，建议使用`--max-rows`先在小数据集上测试
4. **备份**: 建议在运行前备份重要的结果文件

## 故障排除

### 问题1: 内存不足
- 使用更小的模型
- 减少批处理大小
- 使用CPU模式

### 问题2: 某些样本一直失败
- 增加`--num-steps`优化步数
- 调整`--early-stop-confidence`参数
- 尝试不同的表格大小

### 问题3: 进程卡住
- 检查GPU使用情况
- 确保所有依赖正确安装
- 查看错误日志