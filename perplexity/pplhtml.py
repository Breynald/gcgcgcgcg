import math
import html as html_lib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def visualize_ppl_to_html(
    text: str,
    model_name: str = "uer/gpt2-chinese-cluecorpussmall",
    output_path: str = "ppl_vis.html",
):
    """
    对给定文本进行困惑度分析，并将每个 token 的贡献可视化为 HTML 文件。

    参数:
        text:        待分析的文本（字符串）
        model_name:  HuggingFace 上的自回归语言模型名称
        output_path: 输出 HTML 文件路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载 tokenizer 和模型
    print(f"加载模型和 tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # 2. 编码文本
    print(f"编码文本: {text}")
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)       # [1, N]
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, N, vocab_size]

    # 3. 计算每个 token 的 log P(token | 之前的 token)
    #    自回归模型第 i 个位置的 logits 预测的是位置 i 的 token
    #    我们跳过第一个 token，因为它没有“前文”作为条件
    shift_logits = logits[:, :-1, :].contiguous()   # 预测 positions 1..N-1
    shift_labels = input_ids[:, 1:].contiguous()    # 目标 token 为 positions 1..N-1

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)  # [1, N-1, V]
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, N-1]

    # 负对数概率（NLL），越大说明模型越不确定
    nll = -token_log_probs.squeeze(0)  # [N-1]

    # 整体平均 NLL 与困惑度
    avg_nll = nll.mean().item()
    ppl = math.exp(avg_nll)

    print(f"\n整体困惑度 (PPL): {ppl:.3f}")
    print(f"平均 NLL: {avg_nll:.4f}")

    # 4. 准备可视化：取出 token，并和 NLL 对齐
    #    第一个 token 没有条件概率，对应不参与损失，因此从第二个 token 开始对齐
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    vis_tokens = tokens[1:]  # 长度 N-1，对齐 nll

    nll_vals = nll.tolist()
    min_val, max_val = min(nll_vals), max(nll_vals)
    if max_val == min_val:
        scores = [0.0 for _ in nll_vals]
    else:
        scores = [(v - min_val) / (max_val - min_val) for v in nll_vals]  # 归一化到 [0, 1]

    # 5. 构造 HTML 片段：背景色从白色(#ffffff)到红色(#ff0000)
    spans = []
    for tok, s, v in zip(vis_tokens, scores, nll_vals):
        # 从白到红的简单渐变
        r = 255
        g = int(255 * (1 - s))
        b = int(255 * (1 - s))

        # 去掉常见的 BPE 前缀符号，方便阅读
        clean_tok = tok.replace("Ġ", " ").replace("▁", " ")

        # HTML 转义，避免 <, >, & 等打断 HTML 结构
        clean_tok = html_lib.escape(clean_tok)

        # 局部概率和局部困惑度
        prob = math.exp(-v)          # p = exp(-NLL)
        local_ppl = math.exp(v)      # local_ppl = 1/p

        span = (
            f'<span title="NLL={v:.3f}, P={prob:.4f}, local_PPL={local_ppl:.2f}" '
            f'style="background-color:rgba({r},{g},{b},0.8);'
            f'padding:1px 2px;margin:1px;display:inline-block;">'
            f'{clean_tok}</span>'
        )
        spans.append(span)

    inner_html = "".join(spans)

    # 6. 拼成完整 HTML 页面并写入文件
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>困惑度可视化</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
            line-height: 1.8;
        }}
        .container {{
            max-width: 900px;
            margin: 40px auto;
        }}
        .token-area {{
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 4px;
            background: #fafafa;
            word-wrap: break-word;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>文本困惑度可视化</h2>
        <p><b>模型:</b> {html_lib.escape(model_name)}</p>
        <p><b>整体困惑度 (PPL):</b> {ppl:.3f}</p>
        <p><b>平均 NLL:</b> {avg_nll:.4f}</p>
        <h3>每个 token 的贡献（背景越红，模型越困惑）</h3>
        <div class="token-area">
            {inner_html}
        </div>
        <p style="margin-top:15px;font-size:0.9em;color:#666;">
            鼠标悬停在某个 token 上可以查看该位置的 NLL、概率和局部困惑度。
        </p>
    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"\n可视化已保存到: {output_path}")
    print("用浏览器打开即可查看每个 token 对困惑度的贡献。")


if __name__ == "__main__":
    # 你可以在这里修改要分析的文本和输出文件路径
    text = """Here is a table:\n\n| | Metric A | Metric B | Metric C |\n|---|---|---|---|\n| Data 1 |  Agriculture philippines.resume | (copyophage `" |  Given copyrightedegment |\n| Data 2 | .outputROLetti | ,widthSp.cg | .ExHa/mit |\n| Data 3 | unsubscribe Relative Slut | _nc bart dictator |  beat>')");// |\n\nWhere does the sun rise?"""
    output_file = "ppl_vis.html"

    visualize_ppl_to_html(
        text=text,
        model_name="/work/models/Qwen/Qwen2.5-1.5B",  # 如需英文可改成 "gpt2"
        output_path=output_file,
    )