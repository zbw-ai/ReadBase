<a id="coding-top"></a>
# 训练 Infra 面试 Coding 手撕题

> [返回面试速查控制台](2026-08-llm-infra-interview-prep.md#interview-console) · 运行环境：Python 3.10+、PyTorch 2.x

这份题单与主文档的知识题分开计数。现场先讲输入输出、shape、不变量和复杂度，再写主路径，最后补异常与测试。

- [CODING-01｜PyTorch 手写 Multi-Head Self-Attention](#coding-01)
- [CODING-02｜`N×N` 矩阵原地顺时针旋转 90°](#coding-02)

---

<a id="coding-01"></a>
## CODING-01｜PyTorch 手写 Multi-Head Self-Attention

### 30 秒解题思路

输入输出都是 `[B,S,D]`。先检查 `D % H == 0`，分别投影 Q/K/V，再 reshape 成 `[B,H,S,Dh]`；计算 `QKᵀ / sqrt(Dh)`，合并 causal mask 和 padding mask，使用 FP32 softmax，乘 V 后合并 heads，最后做 output projection。这里约定所有 bool mask 都是 **`True` 表示不可见/被屏蔽**。

### 可运行实现与测试

```python
from typing import Optional

import torch
from torch import Tensor, nn


class MultiHeadSelfAttention(nn.Module):
    """教学版 MHA。输入/输出为 [B, S, D]，bool mask 的 True 表示屏蔽。"""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model 必须能被正整数 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x 应为 [B,S,{self.d_model}]，实际为 {tuple(x.shape)}")

        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))  # [B, H, S, Dh]
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]

        blocked: Optional[Tensor] = None

        if is_causal:
            causal = torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=x.device
            ).triu(diagonal=1)
            blocked = causal.reshape(1, 1, seq_len, seq_len)

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise TypeError("attn_mask 必须是 bool，且 True 表示屏蔽")
            if attn_mask.shape == (seq_len, seq_len):
                normalized = attn_mask.reshape(1, 1, seq_len, seq_len)
            elif attn_mask.shape == (batch, seq_len, seq_len):
                normalized = attn_mask.reshape(batch, 1, seq_len, seq_len)
            else:
                raise ValueError("attn_mask 应为 [S,S] 或 [B,S,S]")
            blocked = normalized if blocked is None else (blocked | normalized)

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise TypeError("key_padding_mask 必须是 bool，且 True 表示 padding")
            if key_padding_mask.shape != (batch, seq_len):
                raise ValueError("key_padding_mask 应为 [B,S]")
            padding = key_padding_mask.reshape(batch, 1, 1, seq_len)
            blocked = padding if blocked is None else (blocked | padding)

        if blocked is not None:
            # broadcast 后每个 query 至少要保留一个可见 key，否则 softmax(-inf) 会 NaN。
            expanded = blocked.expand(batch, self.num_heads, seq_len, seq_len)
            if expanded.all(dim=-1).any():
                raise ValueError("存在所有 key 都被屏蔽的 query")
            scores = scores.masked_fill(blocked, float("-inf"))

        # 低精度训练中用 FP32 softmax 更稳，再 cast 回 value dtype。
        probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        context = torch.matmul(probs, v)  # [B, H, S, Dh]
        context = context.transpose(1, 2).contiguous().reshape(batch, seq_len, self.d_model)
        return self.out_proj(context)


def _copy_weights_to_reference(
    custom: MultiHeadSelfAttention, reference: nn.MultiheadAttention
) -> None:
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([
            custom.q_proj.weight,
            custom.k_proj.weight,
            custom.v_proj.weight,
        ]))
        reference.in_proj_bias.copy_(torch.cat([
            custom.q_proj.bias,
            custom.k_proj.bias,
            custom.v_proj.bias,
        ]))
        reference.out_proj.weight.copy_(custom.out_proj.weight)
        reference.out_proj.bias.copy_(custom.out_proj.bias)


def test_multi_head_self_attention() -> None:
    torch.manual_seed(7)
    batch, seq_len, d_model, num_heads = 2, 5, 16, 4
    x = torch.randn(batch, seq_len, d_model)

    custom = MultiHeadSelfAttention(d_model, num_heads).eval()
    reference = nn.MultiheadAttention(
        d_model, num_heads, dropout=0.0, batch_first=True
    ).eval()
    _copy_weights_to_reference(custom, reference)

    padding = torch.tensor([
        [False, False, False, False, True],
        [False, False, False, True, True],
    ])
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(diagonal=1)

    cases = [
        ("no_mask", False, None),
        ("causal", True, None),
        ("padding", False, padding),
        ("causal_and_padding", True, padding),
    ]
    for name, is_causal, key_padding_mask in cases:
        actual = custom(
            x,
            is_causal=is_causal,
            key_padding_mask=key_padding_mask,
        )
        expected, _ = reference(
            x,
            x,
            x,
            attn_mask=causal if is_causal else None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)
        assert actual.shape == (batch, seq_len, d_model)

    x_grad = x.clone().requires_grad_(True)
    loss = custom(x_grad, is_causal=True, key_padding_mask=padding).square().mean()
    loss.backward()
    assert x_grad.grad is not None and torch.isfinite(x_grad.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in custom.parameters()
    )

    try:
        MultiHeadSelfAttention(d_model=10, num_heads=3)
    except ValueError:
        pass
    else:
        raise AssertionError("d_model 不可整除时应抛 ValueError")


if __name__ == "__main__":
    test_multi_head_self_attention()
    print("MHA tests passed")
```

### 复杂度与高频追问

- Attention scores/weighted sum 的时间复杂度是 `O(B·H·S²·Dh)=O(B·S²·D)`，score/probability 显存是 `O(B·H·S²)`；QKV/输出 projection 另有 `O(B·S·D²)`。
- 为什么要乘 `1/sqrt(Dh)`：防止 head dimension 增大时 dot product 方差过大，softmax 饱和、梯度变差。
- 为什么 `transpose` 后要 `contiguous()`：transpose 通常只改 stride；后续按 `[B,S,D]` 合并 heads 前需要可解释的连续布局。`reshape` 有时会隐式复制，面试时最好把布局变化说清楚。
- 工程版还会加入 dropout、cross-attention、RoPE、GQA/MQA、KV cache、FlashAttention 和 TP；现场基础题先保证 shape、mask 和数值正确。

### 常见错误

- 忘记 scale，或错误地除以 `sqrt(D)`；在错误维度 softmax。
- 把 `True` 同时解释成“可见”和“屏蔽”；padding mask 没扩为 `[B,1,1,S]`。
- Q/K/V reshape 后忘记交换 head 与 sequence 维；合并 heads 前忘记恢复 `[B,S,H,Dh]`。
- 低精度直接 softmax；允许某行全部为 `-inf` 后产生 NaN。

↑ [返回题单顶部](#coding-top) · [返回面试速查控制台](2026-08-llm-infra-interview-prep.md#interview-console)

---

<a id="coding-02"></a>
## CODING-02｜`N×N` 矩阵原地顺时针旋转 90°

### 20 秒解题思路

顺时针 90° 的坐标映射是 `(i,j) -> (j,n-1-i)`。原地实现分两步：先沿主对角线交换 `(i,j)` 与 `(j,i)`，再原地反转每一行。必须在第一次写入前验证每行长度都是 `N`，避免 ragged/non-square 输入被改到一半才失败。

### 可运行实现与测试

```python
from typing import Any


def rotate_clockwise(matrix: list[list[Any]]) -> None:
    """将 N×N 矩阵原地顺时针旋转 90°。"""
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix 必须是非 ragged 的 N×N 方阵")

    # 1. 主对角线转置，只交换上三角与下三角。
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # 2. 每行原地反转；不用 row[::-1]，避免创建切片副本。
    for row in matrix:
        row.reverse()


def test_rotate_clockwise() -> None:
    empty: list[list[int]] = []
    rotate_clockwise(empty)
    assert empty == []

    one = [[1]]
    rotate_clockwise(one)
    assert one == [[1]]

    three = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    rotate_clockwise(three)
    assert three == [
        [7, 4, 1],
        [8, 5, 2],
        [9, 6, 3],
    ]

    for invalid in ([[1, 2, 3], [4, 5, 6]], [[1, 2], [3]]):
        before = [row.copy() for row in invalid]
        try:
            rotate_clockwise(invalid)
        except ValueError:
            assert invalid == before  # 验证失败发生在任何原地写入之前。
        else:
            raise AssertionError("非方阵或 ragged 输入应抛 ValueError")


if __name__ == "__main__":
    test_rotate_clockwise()
    print("matrix rotation tests passed")
```

### 复杂度与高频追问

- 时间复杂度 `O(N²)`：转置和反转都各访问常数次矩阵元素。
- 额外空间 `O(1)`：算法只使用循环变量和临时交换槽；测试中的 `before` 不属于算法空间。
- 逆时针 90°：可以先转置，再反转列；180°：可以整体对称交换或每行反转后再反转行序。

### 常见错误

- 先转置再反转“列”，得到的是逆时针而不是顺时针。
- 使用 `matrix[:] = zip(...)`、`row[::-1]` 或额外结果矩阵，却声称 `O(1)` 额外空间。
- 循环整个矩阵做 transpose，导致同一对元素交换两次。
- 写到一半才检查非方阵，异常后输入已经被部分修改。

↑ [返回题单顶部](#coding-top) · [返回面试速查控制台](2026-08-llm-infra-interview-prep.md#interview-console)
