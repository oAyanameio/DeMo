#!/bin/bash
# 同题指纹比对: necodex gpt-5.6-sol 的球棒题回答 vs 9miao 各国产模型的同题回答
# sol 的已知输出: 设球的价格为 \(x\)，球棒为 \(x+1.00\)。 [LaTeX推导] **球的价格是 0.05。**
set -uo pipefail
Q='A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much is the ball? Show brief steps.'
for M in glm-5.2 deepseek-v4-pro deepseek-v4-flash kimi-k2.5 qwen3-max minimax-m2.5; do
  echo "===== $M ====="
  timeout 100 hermes chat -q "$Q" --provider custom --model "$M" 2>&1 \
    | grep -viE "^╰|^╭|stop|resume|^Session|^Title|^Duration|^Messages|^\s*┊|^\s*$|initializ|─────|^Query" \
    | head -12
done
