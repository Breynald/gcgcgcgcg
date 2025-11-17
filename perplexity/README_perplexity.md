# 文本困惑度计算工具

这个工具用于计算给定文本的困惑度（Perplexity），这是衡量语言模型对文本预测能力的一个重要指标。

## 功能说明

`perplexity_calculator.py` 脚本可以：
1. 加载预训练的语言模型（默认使用GPT-2）
2. 计算输入文本的困惑度
3. 支持通过命令行参数直接输入文本或从文件读取文本

## 依赖安装

在使用此工具之前，请确保安装了必要的依赖库：

```bash
pip install torch transformers
```

## 使用方法

### 1. 通过命令行参数输入文本

```bash
python perplexity_calculator.py --text "Your text here"
```

### 2. 从文件读取文本

```bash
python perplexity_calculator.py --file path/to/your/text/file.txt
```

### 3. 指定不同的预训练模型

```bash
python perplexity_calculator.py --text "Your text here" --model "bert-base-uncased"
```

### 4. 指定运行设备

```bash
python perplexity_calculator.py --text "Your text here" --device "cpu"
```

## 参数说明

- `--text`: 直接输入要计算困惑度的文本
- `--file`: 指定包含文本的文件路径
- `--model`: 指定预训练模型的名称或路径（默认为"gpt2"）
- `--device`: 指定运行设备，"cuda"或"cpu"（默认自动检测）

## 示例

计算测试文件的困惑度：
```bash
python perplexity_calculator.py --file test_text.txt --model gpt2
```

## 输出说明

脚本将输出以下信息：
1. 加载的模型名称
2. 模型运行的设备
3. 输入的文本内容
4. 计算得到的困惑度值

困惑度值越低，表示语言模型对文本的预测越准确。