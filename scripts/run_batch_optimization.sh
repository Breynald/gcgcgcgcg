#!/bin/bash

# 批量优化脚本 - 自动运行GCG批量优化
# 作者: Claude Code
# 用法: ./run_batch_optimization.sh [选项]
# ./run_batch_optimization.sh -f --reverse
set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DEFAULT_ENV="verl"
DEFAULT_MODEL="/work/models/Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TARGET="Copyright"
DEFAULT_MAX_TABLE_SIZE=5
DEFAULT_NUM_STEPS=300
DEFAULT_DEVICE="cuda"
DEFAULT_DTYPE="float16"
DEFAULT_OUTPUT_DIR="batch_results"
DEFAULT_GPU_ID="7"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
批量优化脚本运行工具

用法: $0 [选项]

选项:
    -t, --test              只运行测试，不进行优化
    -s, --small             小规模测试 (2x2表格, 20步)
    -m, --medium            中等规模测试 (5x5表格, 200步)
    -f, --full              完整规模测试 (9x9表格, 500步) [默认]
    --env ENV               指定conda环境 [默认: verl]
    --model MODEL           指定模型 [默认: mistralai/Mistral-7B-Instruct-v0.3]
    --target TARGET         指定优化目标 [默认: Copyright]
    --device DEVICE         指定设备 [默认: cuda]
    --gpu-id GPU_ID         指定GPU ID [默认: 0]
    --probe-sampling        启用probe sampling
    --reverse               从最大表格尺寸开始运行（推荐避免内存溢出浪费时间）
    --cpu                   使用CPU而不是GPU
    --dry-run               只显示将要运行的命令，不实际执行
    -h, --help              显示此帮助信息

示例:
    $0                      # 运行完整优化
    $0 -t                   # 只运行测试
    $0 -s                   # 小规模测试
    $0 -m --probe-sampling  # 中等规模测试 + probe sampling
    $0 --reverse            # 从最大表格开始运行（推荐）
    $0 --cpu                # 使用CPU运行
    $0 --target "Hello"     # 自定义目标词

EOF
}

# 检查环境
check_environment() {
    print_info "检查运行环境..."

    # 检查conda是否可用
    if ! command -v conda &> /dev/null; then
        print_error "conda未找到，请先安装conda"
        exit 1
    fi

    # 检查指定的conda环境是否存在
    if ! conda env list | grep -q "$CONDA_ENV"; then
        print_error "conda环境 '$CONDA_ENV' 不存在"
        print_info "可用环境:"
        conda env list
        exit 1
    fi

    # 检查脚本目录是否存在
    if [ ! -f "batch_optimization.py" ]; then
        print_error "batch_optimization.py 脚本未找到"
        print_info "请确保在正确的目录中运行此脚本"
        exit 1
    fi

    # 设置GPU
    if [ "$DEVICE" = "cuda" ] && [ -n "$GPU_ID" ]; then
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        print_info "设置GPU: $GPU_ID"
    fi

    print_success "环境检查通过"
}

# 运行测试
run_tests() {
    print_info "运行功能测试..."

    local test_cmd="source /root/miniconda3/bin/activate $CONDA_ENV && python test_batch_optimization.py --skip-integration"

    if [ "$DRY_RUN" = true ]; then
        print_info "DRY RUN - 将要执行的测试命令:"
        echo "$test_cmd"
        return
    fi

    if eval "$test_cmd"; then
        print_success "测试通过"
    else
        print_error "测试失败"
        exit 1
    fi
}

# 构建优化命令
build_optimization_command() {
    local cmd="source /root/miniconda3/bin/activate $CONDA_ENV && python batch_optimization.py"

    # 添加参数
    cmd="$cmd --model '$MODEL'"
    cmd="$cmd --target '$TARGET'"
    cmd="$cmd --max-table-size $MAX_TABLE_SIZE"
    cmd="$cmd --num-steps $NUM_STEPS"
    cmd="$cmd --device $DEVICE"
    cmd="$cmd --dtype $DTYPE"

    if [ "$PROBE_SAMPLING" = true ]; then
        cmd="$cmd --probe-sampling"
    fi

    if [ "$REVERSE_MODE" = true ]; then
        cmd="$cmd --reverse"
    fi

    echo "$cmd"
}

# 运行优化
run_optimization() {
    print_info "开始批量优化..."
    print_info "配置参数:"
    print_info "  模型: $MODEL"
    print_info "  目标: $TARGET"
    print_info "  表格大小: ${MAX_TABLE_SIZE}x${MAX_TABLE_SIZE}"
    print_info "  优化步数: $NUM_STEPS"
    print_info "  设备: $DEVICE"
    if [ "$DEVICE" = "cuda" ] && [ -n "$GPU_ID" ]; then
        print_info "  GPU ID: $GPU_ID"
    fi
    print_info "  数据类型: $DTYPE"
    if [ "$REVERSE_MODE" = true ]; then
        print_info "  运行模式: 从最大表格开始"
    fi
    if [ "$PROBE_SAMPLING" = true ]; then
        print_info "  Probe Sampling: 启用"
    fi

    local cmd=$(build_optimization_command)

    if [ "$DRY_RUN" = true ]; then
        print_info "DRY RUN - 将要执行的优化命令:"
        echo "$cmd"
        return
    fi

    print_info "执行命令: $cmd"

    if eval "$cmd"; then
        print_success "批量优化完成！"
        print_info "结果保存在: $DEFAULT_OUTPUT_DIR/"
    else
        print_error "批量优化失败"
        exit 1
    fi
}

# 解析命令行参数
parse_arguments() {
    TEST_ONLY=false
    SMALL_SCALE=false
    MEDIUM_SCALE=false
    FULL_SCALE=true
    CONDA_ENV=$DEFAULT_ENV
    MODEL=$DEFAULT_MODEL
    TARGET=$DEFAULT_TARGET
    DEVICE=$DEFAULT_DEVICE
    GPU_ID=$DEFAULT_GPU_ID
    DTYPE=$DEFAULT_DTYPE
    PROBE_SAMPLING=false
    REVERSE_MODE=false
    DRY_RUN=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--test)
                TEST_ONLY=true
                FULL_SCALE=false
                shift
                ;;
            -s|--small)
                SMALL_SCALE=true
                FULL_SCALE=false
                shift
                ;;
            -m|--medium)
                MEDIUM_SCALE=true
                FULL_SCALE=false
                shift
                ;;
            -f|--full)
                FULL_SCALE=true
                shift
                ;;
            --env)
                CONDA_ENV="$2"
                shift 2
                ;;
            --model)
                MODEL="$2"
                shift 2
                ;;
            --target)
                TARGET="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --gpu-id)
                GPU_ID="$2"
                shift 2
                ;;
            --probe-sampling)
                PROBE_SAMPLING=true
                shift
                ;;
            --reverse)
                REVERSE_MODE=true
                shift
                ;;
            --cpu)
                DEVICE="cpu"
                DTYPE="float32"
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

    # 设置规模参数
    if [ "$SMALL_SCALE" = true ]; then
        MAX_TABLE_SIZE=2
        NUM_STEPS=20
    elif [ "$MEDIUM_SCALE" = true ]; then
        MAX_TABLE_SIZE=5
        NUM_STEPS=200
    elif [ "$FULL_SCALE" = true ]; then
        MAX_TABLE_SIZE=$DEFAULT_MAX_TABLE_SIZE
        NUM_STEPS=$DEFAULT_NUM_STEPS
    fi
}

# 主函数
main() {
    echo "========================================"
    echo "     批量优化脚本运行工具"
    echo "========================================"
    echo ""

    parse_arguments "$@"

    print_info "使用配置:"
    print_info "  Conda环境: $CONDA_ENV"
    print_info "  运行模式: $([ "$TEST_ONLY" = true ] && echo "仅测试" || [ "$SMALL_SCALE" = true ] && echo "小规模" || [ "$MEDIUM_SCALE" = true ] && echo "中等规模" || echo "完整规模")"
    print_info "  Dry Run: $([ "$DRY_RUN" = true ] && echo "是" || echo "否")"
    echo ""

    check_environment

    if [ "$TEST_ONLY" = true ]; then
        run_tests
    else
        run_tests
        run_optimization
    fi

    echo ""
    print_success "脚本执行完成！"
}

# 运行主函数
main "$@"