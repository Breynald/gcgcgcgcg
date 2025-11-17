#!/bin/bash

# 优化成功率测试脚本
# 自动找到最新的优化结果并测试成功率

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助
show_help() {
    cat << EOF
优化成功率测试脚本（支持多模型假阳率测试）

用法: $0 [选项]

选项:
    --results FILE          指定优化结果文件
    --model MODEL           指定目标模型（成功率测试）[默认: /work/models/Qwen/Qwen2.5-1.5B-Instruct]
    --fp-models MODEL1 MODEL2 ...  指定假阳率测试模型列表
    --model-names NAME1 NAME2 ...  模型别名列表（便于报告展示）
    --target TARGET         指定目标词 [默认: Copyright]
    --samples N             每个prompt的测试次数 [默认: 8]
    --gpu-id N              指定GPU ID [默认: 7]
    --single-model          使用单模型模式（兼容旧版本）
    --dry-run               只显示命令，不执行
    -h, --help              显示帮助信息

示例:
    # 基础使用（仅测试目标模型成功率）
    $0

    # 使用指定的结果文件
    $0 --results results.json

    # 测试目标模型成功率 + 多个模型的假阳率
    $0 --model /path/to/target \\
        --fp-models /path/to/model1 /path/to/model2 \\
        --model-names Target Model1 Model2

    # 自定义目标和测试次数
    $0 --target "Hello" --samples 10

    # 单模型模式（兼容旧版本）
    $0 --single-model --model /path/to/model

EOF
}

# 解析参数
RESULTS_FILE=""
MODEL_PATH="/work/models/Qwen/Qwen2.5-1.5B-Instruct"
FP_MODELS=(/work/models/meta-llama/Llama-3.1-8B-Instruct)
MODEL_NAMES=(Qwen Llama3)
TARGET="Copyright"
SAMPLES=8
GPU_ID="6"
SINGLE_MODEL=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --results)
            RESULTS_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --fp-models)
            shift
            # 收集所有 --fp-models 之后的模型路径，直到遇到下一个选项
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                FP_MODELS+=("$1")
                shift
            done
            ;;
        --model-names)
            shift
            # 收集所有 --model-names 之后的模型名称，直到遇到下一个选项
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODEL_NAMES+=("$1")
                shift
            done
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --single-model)
            SINGLE_MODEL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 查找最新的优化结果文件
if [ -z "$RESULTS_FILE" ]; then
    print_info "查找最新的优化结果文件..."

    # 在batch_results目录中查找所有包含optimization_results.json的目录
    VALID_DIRS=$(find batch_results -type d -name "batch_*" -exec test -f "{}/optimization_results.json" \; -print | sort)

    if [ -z "$VALID_DIRS" ]; then
        print_error "未找到包含 optimization_results.json 的batch_results目录"
        exit 1
    fi

    # 获取最新的有效目录
    LATEST_DIR=$(echo "$VALID_DIRS" | tail -1)
    RESULTS_FILE="$LATEST_DIR/optimization_results.json"

    print_info "找到有效目录: $LATEST_DIR"
fi

print_info "使用优化结果: $RESULTS_FILE"
print_info "测试模型: $MODEL_PATH"
if [ ${#FP_MODELS[@]} -gt 0 ]; then
    print_info "假阳率测试模型: ${FP_MODELS[*]}"
fi
if [ ${#MODEL_NAMES[@]} -gt 0 ]; then
    print_info "模型别名: ${MODEL_NAMES[*]}"
fi
print_info "目标词: $TARGET"
print_info "测试次数: $SAMPLES"
print_info "GPU ID: $GPU_ID"
if [ "$SINGLE_MODEL" = true ]; then
    print_info "模式: 单模型测试"
else
    print_info "模式: 多模型测试（成功率 + 假阳率）"
fi

# 构建测试命令
TEST_CMD="source /root/miniconda3/bin/activate verl && export CUDA_VISIBLE_DEVICES=$GPU_ID && TRANSFORMERS_VERBOSITY=error python test_optimization_results.py"
TEST_CMD="$TEST_CMD --results '$RESULTS_FILE'"
TEST_CMD="$TEST_CMD --model '$MODEL_PATH'"
TEST_CMD="$TEST_CMD --target '$TARGET'"
TEST_CMD="$TEST_CMD --samples $SAMPLES"
TEST_CMD="$TEST_CMD --device cuda"

# 添加假阳率模型参数
if [ ${#FP_MODELS[@]} -gt 0 ]; then
    TEST_CMD="$TEST_CMD --fp-models"
    for fp_model in "${FP_MODELS[@]}"; do
        TEST_CMD="$TEST_CMD '$fp_model'"
    done
fi

# 添加模型名称参数
if [ ${#MODEL_NAMES[@]} -gt 0 ]; then
    TEST_CMD="$TEST_CMD --model-names"
    for name in "${MODEL_NAMES[@]}"; do
        TEST_CMD="$TEST_CMD '$name'"
    done
fi

# 添加单模型模式参数
if [ "$SINGLE_MODEL" = true ]; then
    TEST_CMD="$TEST_CMD --single-model"
fi

if [ "$DRY_RUN" = true ]; then
    print_info "DRY RUN - 将要执行的命令:"
    echo "$TEST_CMD"
    exit 0
fi

print_info "开始测试优化成功率..."
echo "执行的命令: $TEST_CMD"

# 运行测试
cd "$(dirname "$0")"
eval "$TEST_CMD"

print_success "测试完成！"