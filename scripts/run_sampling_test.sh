#!/bin/bash

# 采样参数测试脚本
# 测试不同温度、top_p、top_k参数对优化结果成功率的影响

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
采样参数测试脚本

用法: $0 [选项]

选项:
    --results FILE          指定优化结果文件
    --model MODEL           指定测试模型 [默认: /work/models/Qwen/Qwen2.5-1.5B-Instruct]
    --target TARGET         指定目标词 [默认: Copyright]
    --samples N             每个配置的测试次数 [默认: 8]
    --gpu-id N              指定GPU ID [默认: 7]
    --temperatures LIST     温度值列表 [默认: "0.5 0.8 1.0 1.2 1.5"]
    --topp-values LIST      Top-p值列表 [默认: "0.3 0.5 0.7 0.9 0.95"]
    --topk-values LIST      Top-k值列表 [默认: "10 50 100"]
    --dry-run               只显示命令，不执行
    -h, --help              显示帮助信息

示例:
    $0                              # 使用最新的优化结果
    $0 --results results.json       # 使用指定的结果文件
    $0 --target "Hello" --samples 5 # 自定义目标和测试次数
    $0 --temperatures "0.1 0.5 1.0" # 自定义温度值

EOF
}

# 解析参数
RESULTS_FILE=""
MODEL_PATH="/work/models/Qwen/Qwen2.5-1.5B-Instruct"
TARGET="Copyright"
SAMPLES=32
GPU_ID="7"
TEMPERATURES="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5"
TOPP_VALUES="0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95"
TOPK_VALUES="10 50 100"
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
        --temperatures)
            TEMPERATURES="$2"
            shift 2
            ;;
        --topp-values)
            TOPP_VALUES="$2"
            shift 2
            ;;
        --topk-values)
            TOPK_VALUES="$2"
            shift 2
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

    # 在batch_results目录中查找最新的results文件
    LATEST_DIR=$(find batch_results -type d -name "batch_*" | sort | tail -1)

    if [ -z "$LATEST_DIR" ]; then
        print_error "未找到batch_results目录或优化结果"
        exit 1
    fi

    RESULTS_FILE="$LATEST_DIR/optimization_results.json"

    if [ ! -f "$RESULTS_FILE" ]; then
        print_error "在 $LATEST_DIR 中未找到 optimization_results.json"
        exit 1
    fi
fi

print_info "使用优化结果: $RESULTS_FILE"
print_info "测试模型: $MODEL_PATH"
print_info "目标词: $TARGET"
print_info "每个配置测试次数: $SAMPLES"
print_info "GPU ID: $GPU_ID"
print_info "温度值: $TEMPERATURES"
print_info "Top-p值: $TOPP_VALUES"
print_info "Top-k值: $TOPK_VALUES"

# 构建测试命令
TEST_CMD="source /root/miniconda3/bin/activate verl && export CUDA_VISIBLE_DEVICES=$GPU_ID && python test_sampling_effects.py"
TEST_CMD="$TEST_CMD --results '$RESULTS_FILE'"
TEST_CMD="$TEST_CMD --model '$MODEL_PATH'"
TEST_CMD="$TEST_CMD --target '$TARGET'"
TEST_CMD="$TEST_CMD --samples $SAMPLES"
TEST_CMD="$TEST_CMD --device cuda"
TEST_CMD="$TEST_CMD --temperatures $TEMPERATURES"
TEST_CMD="$TEST_CMD --topp-values $TOPP_VALUES"
TEST_CMD="$TEST_CMD --topk-values $TOPK_VALUES"

if [ "$DRY_RUN" = true ]; then
    print_info "DRY RUN - 将要执行的命令:"
    echo "$TEST_CMD"
    exit 0
fi

print_info "开始测试采样参数影响..."
echo "执行的命令: $TEST_CMD"

# 运行测试
cd "$(dirname "$0")"
eval "$TEST_CMD"

print_success "采样参数测试完成！"