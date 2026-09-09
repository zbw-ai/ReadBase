<a id="coding-top"></a>
# 训练 Infra 面试 Coding 手撕题

> [返回面试速查控制台](2026-08-llm-infra-interview-prep.md#interview-console) · [底层基础理论](2026-08-llm-infra-interview-prep.md#part-foundations) · [岗位准备入口](2026-08-llm-infra-interview-prep.md#meshy-interview-sprint)

本页集中放可动手的实现、测试与环境检查；概念推导和系统判断回到主文档，不另建一套岗位专属技术答案。除明确标注本人现场经历的题目外，均为自拟复习练习，不代表 Meshy 或其他公司的已确认实题。现场先讲输入输出、shape、不变量和复杂度，再写主路径，最后补异常与测试。

| Topic | 题目与入口 | 运行依赖与边界 |
|---|---|---|
| 基础模型实现与梯度 | [CODING-01｜PyTorch MHA](#coding-01) · [CODING-05｜Linear NumPy 方向梯度](#coding-05) | Python 3.10+；01 依赖 PyTorch 2.x，可在 CPU 测试；05 依赖 NumPy |
| 数据结构与数组 | [CODING-02｜矩阵旋转](#coding-02) · [CODING-03｜父指针 LCA（字节 AML 一面）](#coding-03) · [CODING-04｜LRU（小红书一面）](#coding-04) | Python 3.10+ 标准库；原题来源和补充边界见各题 |
| 性能与 kernel | [CODING-06｜eager / compile benchmark](#coding-06) · [CODING-07｜Triton fusion](#coding-07) | 需要真实 NVIDIA CUDA GPU、兼容驱动和 CUDA 版 PyTorch；06 需要可用 Inductor 环境，07 另依赖 Triton；GPU 路径尚未实测 |
| 环境与现场操作 | [共享屏幕环境检查](#coding-env) | 先确认运行机器、解释器与工具权限；CPU/MPS 测试不冒充 CUDA 验证 |

按现场提供的 PyTorch/Triton 版本核对受支持的 Python 范围，不为跑练习临时升级驱动或整套依赖。各题 Python 代码块按独立脚本使用；07 的验证可复用 06 的计时函数，但必须先检查正确性。

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

---

<a id="coding-03"></a>
## CODING-03｜带父指针的二叉树最近公共祖先

**来源：字节跳动 AML 技术一面，2026-09-08，本人现场题目与代码回忆。** 已确认给定两个节点 `p、q`，节点带 `parent` 指针；不提供、也不需要 `root`。以下保留现场双指针算法，整理缩进、`__init__` 和类型标注，并用 `is` 明确比较节点身份；不是官方公开题库。

### 题意与前提

返回 `p、q` 的最近公共祖先节点。每个节点有 `val、left、right、parent`，根节点的 `parent` 为 `None`，父指针链无环。节点自身也算自己的祖先，因此 `p is q` 或一方是另一方祖先都需要处理；不能假定节点值唯一，也不依赖二叉搜索树性质。

### 20–30 秒解题思路

> 从 p、q 沿 parent 向上走，会形成两条链表，它们的第一个公共节点就是最近公共祖先。我用两个指针分别从 p、q 出发，到 None 后切换到另一个起点，这样可以抵消两条路径的长度差；两个指针相遇就返回。只移动指针，不修改树，额外空间是 O(1)。

### 可运行实现与测试

```python
from __future__ import annotations


class Node:
    def __init__(self, val: int) -> None:
        self.val = val
        self.left: Node | None = None
        self.right: Node | None = None
        self.parent: Node | None = None


class Solution:
    def lowestCommonAncestor(self, p: Node, q: Node) -> Node | None:
        a, b = p, q

        while a is not b:
            a = a.parent if a is not None else q
            b = b.parent if b is not None else p

        return a


def test_lowest_common_ancestor() -> None:
    root = Node(0)
    left, right = Node(1), Node(1)  # 同值，但不是同一个节点。
    leaf = Node(2)
    root.left, root.right = left, right
    left.parent = right.parent = root
    left.left = leaf
    leaf.parent = left

    lca = Solution().lowestCommonAncestor
    assert lca(left, right) is root        # 同层、重复 val。
    assert lca(leaf, right) is root        # 深度不同。
    assert lca(left, leaf) is left         # p 是 q 的祖先。
    assert lca(leaf, left) is left         # q 是 p 的祖先。
    assert lca(leaf, leaf) is leaf         # 同一个节点。
    assert lca(root, leaf) is root         # 其中一个是根。
    assert lca(root, root) is root

    # 扩展测试：不属于同一棵树时，最终同时走到 None。
    other_root = Node(1)
    assert lca(left, other_root) is None


if __name__ == "__main__":
    test_lowest_common_ancestor()
    print("parent-pointer LCA tests passed")
```

### 为什么能找到“最近”的公共祖先

1. 树中每个节点只有一个父节点；两条父链一旦相交，直到根的后续路径就完全相同，因此是“各自独有的前缀＋公共后缀”。公共后缀的起点就是 LCA。
2. 深度不同，直接同时向上走可能错过交点。切换起点后，两个指针分别补走对方的路径，抵消独有前缀的长度差，在公共后缀起点相遇；若此前已相遇则直接返回。
3. **必须比较对象身份**：`a is b` 表示同一节点。`a.val == b.val` 可能把不同节点误判成祖先；`!=` 是否等价还取决于类有没有重载相等比较。

### 复杂度、意图与追问

- **时间**：`O(hp + hq)`，`hp、hq` 表示两条父链的长度；**额外空间**：`O(1)`。不遍历整棵树，不需要递归栈或祖先集合。
- **面试官意图**：能否利用 `parent` 把树题转成链表相交；能否解释双指针为什么终止、比较的是身份还是值，以及边界条件。
- **不同树怎么办？** 在无环且根的 parent 为 `None` 的前提下，这份实现会返回 `None`。这是额外支持的行为，不把它说成已确认的现场题目要求。
- **还有什么解法？** 可以先算两条父链长度，让较深节点先走长度差，再同步上移，同样是 `O(hp+hq)` 时间、`O(1)` 空间；也可以用集合保存 p 的祖先，再从 q 向上找第一个命中，但需要额外空间。
- **没有 parent 怎么办？** 那是另一种题面，通常需要 `root`，用树上的递归寻找 LCA；不能直接套本题方法。

### 常见错误

- 比较节点值，而不是节点身份；只测试不同值的节点，漏掉误判。
- 把切换条件改成 `a.parent is None`，跳过 `None` 的对齐过程，破坏当前实现的边界与终止保证。
- 切换时回到自己的起点，或对 `None` 直接访问 `.parent`。
- 假定两个节点一定是叶子；漏掉同节点和祖先关系。
- 修改 parent、引入递归/集合，却仍宣称是原双指针或 `O(1)` 空间；父链有环时仍声称一定终止。

↑ [返回题单顶部](#coding-top) · [返回面试速查控制台](2026-08-llm-infra-interview-prep.md#interview-console) · [返回字节 AML 入口](2026-08-llm-infra-interview-prep.md#bytedance-aml-sprint)

---

<a id="coding-04"></a>
## CODING-04｜LRU 缓存：实现 get / put

**来源：小红书大模型训练框架研发工程师/专家技术一面，2026-09-09，本人提供的现场题目与代码。** 要求实现 LRU 缓存的 `get` 和 `put`；用户给出的解法使用 `collections.OrderedDict`。以下保留该算法，恢复粘贴时损坏的缩进和 `__init__`，补充测试及负容量校验；后两项是整理时添加，不声称现场已经写出。是否允许标准库、是否要求手写双向链表，以面试官实际要求为准。

### 题意与不变量

- `get(key)`：存在则返回 value，并把该 key 标为最近使用；不存在返回 `-1`，不改变缓存顺序。
- `put(key, value)`：存在则更新 value 并刷新使用次序；不存在则插入，超过容量时淘汰最久未使用的 key。
- 始终保持 **左端是 LRU（最久未使用），右端是 MRU（最近使用）**；每次公开操作结束后，元素数量不超过 capacity。
- 本实现约定 capacity 为非负整数、key 可哈希；容量 0 表示不缓存，负数抛 `ValueError`。这些边界是复习时补充的接口约定，不当成已确认的现场测试范围。

### 20–30 秒解题思路

> 我用 OrderedDict 同时做 key 查找和使用次序维护，左边最旧、右边最新。get 命中后移到末尾；put 已有 key 时更新值并移到末尾，新 key 插到末尾。如果超过容量，就删除最前面的元素。哈希查找和插入按平均、摊销复杂度计算，两种操作都是 O(1)，空间是 O(capacity)。

### 可运行 Python3 实现与测试

```python
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity 必须为非负整数")
        self.capacity = capacity
        self.od = OrderedDict()

    def get(self, key):
        if key not in self.od:
            return -1
        self.od.move_to_end(key)  # 命中后移到右端 MRU。
        return self.od[key]

    def put(self, key, value) -> None:
        if key in self.od:
            self.od.move_to_end(key)
            self.od[key] = value
        else:
            self.od[key] = value
            if len(self.od) > self.capacity:
                self.od.popitem(last=False)  # 左端 LRU。


def test_lru_cache() -> None:
    # 1. get 命中刷新次序；容量满后淘汰的是 LRU，不是最早插入者。
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    assert list(cache.od) == [2, 1]
    cache.put(3, 3)
    assert cache.get(2) == -1
    cache.put(4, 4)
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4

    # 2. 更新已有 key：刷新次序、覆盖 value，不增加缓存数量。
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(1, 10)
    assert list(cache.od.items()) == [(2, 2), (1, 10)]
    cache.put(3, 3)
    assert cache.get(2) == -1
    assert cache.get(1) == 10
    assert len(cache.od) == 2

    # 3. get 未命中不能改变已有 key 的顺序。
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(99) == -1
    assert list(cache.od) == [1, 2]
    cache.put(3, 3)
    assert cache.get(1) == -1
    assert cache.get(2) == 2

    # 4. 容量为 1；同值更新、重复读取仍然有效。
    cache = LRUCache(1)
    cache.put(1, 1)
    cache.put(1, 10)
    cache.put(1, 10)
    assert cache.get(1) == cache.get(1) == 10
    cache.put(2, 2)
    assert cache.get(1) == -1
    assert cache.get(2) == 2

    # 5. 容量为 0：插入后立即淘汰，不保留任何 key。
    cache = LRUCache(0)
    for key in range(3):
        cache.put(key, key)
        assert cache.get(key) == -1
        assert not cache.od

    # 6. 非法负容量，属于本整理版补充的防御性校验。
    try:
        LRUCache(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("负容量应抛 ValueError")


if __name__ == "__main__":
    test_lru_cache()
    print("LRU tests passed")
```

### 为什么正确，复杂度怎么算？

每次命中读取、更新或新增都让对应 key 成为 MRU；其他 key 的相对次序不变。因此左端始终是最近一次访问最早的 key，超容时删除左端就是 LRU 淘汰。已有 key 的 `put` 先移动、再赋值没有问题，更新已有值本身不会把 key 移回原处。

- **时间**：在常规哈希复杂度假设下，`get` 平均 `O(1)`，`put` 平均/摊销 `O(1)`，不承诺极端哈希冲突下的严格最坏 `O(1)`。
- **空间**：最多保留 capacity 个键值对，通常记 `O(capacity)`；严格包含对象常数开销可记 `O(capacity + 1)`。先插入再淘汰时，瞬时最多多一个元素。
- **只需 `if`，为何不用 `while`？** 容量固定、此前未超容且一次只新增一个 key，最多超出一个；若支持动态缩容，需要另处理多项淘汰。

`move_to_end(key)` 默认移到右端，`popitem(last=False)` 删除左端；普通赋值不会自动把访问顺序更新成 LRU 顺序。接口语义见 [Python OrderedDict 官方文档](https://docs.python.org/3/library/collections.html#collections.OrderedDict)。

### 面试官意图与高频追问

- **考什么？** 能否同时满足快速查找与快速更新使用次序；能否分清 LRU、FIFO、LFU；能否覆盖“更新已有 key”和容量边界。
- **不允许 OrderedDict 怎么办？** 用 `dict[key] -> node` 定位节点，双向链表维护 LRU→MRU，配头尾哨兵；命中时摘下节点再接到尾部，淘汰头部真实节点并同步删除 dict 项。单链表难以在不知道前驱时 `O(1)` 删除任意节点。
- **普通 dict 不是已经有序吗？** 它保留的是插入顺序；读取和更新已有 key 不会自动刷新位置。不能只写 `dict[key] = value` 就声称实现 LRU，删除/重插模拟时也要解释端点淘汰及复杂度。
- **线程安全吗？** 这份实现不保证复合操作线程安全。共享缓存时，查找、移动、更新和淘汰需要共同的同步策略；不能因为有 GIL 就认为整个 `get/put` 原子化。
- **与 Infra 有什么联系？** KV/prefix cache、样本或 Embedding cache 都涉及缓存淘汰，但不一定直接采用普通 LRU；生产环境还需考虑对象大小、复用概率、引用/占用状态和淘汰成本。单次顺序扫描也可能把热点挤出 LRU。

### 常见错误

- `get` 只返回值，不刷新使用次序；`put` 更新已有 key 后也忘记刷新。
- 用默认 `popitem()` 弹出右端 MRU，淘汰方向反了。
- 更新已有 key 时误增 size，导致错误淘汰；按 `len >= capacity` 提前删掉本来可容纳的元素。
- 把粘贴后的 `**init**` 当成 Python 方法名；实际应为 `__init__`，方法体必须正确缩进。这是粘贴格式问题，不据此判断现场提交存在语法错误。

↑ [返回题单顶部](#coding-top) · [返回面试速查控制台](2026-08-llm-infra-interview-prep.md#interview-console) · [返回小红书入口](2026-08-llm-infra-interview-prep.md#xiaohongshu-sprint) · [知识图谱](../training-infra-roadmap/KNOWLEDGE_GRAPH.md) · [阅读索引](../training-infra-roadmap/MASTER_READING_LIST.md)

---

<a id="coding-05"></a>
## CODING-05｜Linear：用 NumPy 方向有限差分检查梯度

### 题意与前提

给定 `Y=XWᵀ+b` 和固定上游梯度 G，定义标量 `L=sum(Y*G)`；比较数值方向导数和解析梯度内积。这是 **CPU/NumPy 教学测试**，不是训练 benchmark。先独立推导，再检查代码，理论见 [Linear 前后向与低精度](2026-08-llm-infra-interview-prep.md#pytorch-03)。

基础模型实现还可闭卷练 [稳定 softmax](2026-09-meshy-ml-system-written-prep.md#meshy-a02)、[NumPy Attention](2026-09-meshy-ml-system-written-prep.md#meshy-a03) 和 [PyTorch MHA](#coding-01)。特别说清 mask 的 bool 含义、softmax 沿 key 轴、全 mask 行的约定；不要把不同题目的 mask 语义混用。

### 可运行实现与测试

```python
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=(3, 4))
w = rng.normal(size=(5, 4))
b = rng.normal(size=(5,))
g = rng.normal(size=(3, 5))

def loss(x, w, b):
    return np.sum((x @ w.T + b) * g)

grads = (g @ w, g.T @ x, g.sum(axis=0))
args = [x, w, b]
eps = 1e-6
for i, grad in enumerate(grads):
    direction = rng.normal(size=args[i].shape)
    plus, minus = args.copy(), args.copy()
    plus[i] = args[i] + eps * direction
    minus[i] = args[i] - eps * direction
    numerical = (loss(*plus) - loss(*minus)) / (2 * eps)
    analytical = np.sum(grad * direction)
    np.testing.assert_allclose(numerical, analytical, rtol=1e-6, atol=1e-7)
print("Linear directional gradient checks: PASS")
```

### 高频追问与测试边界

有限差分有截断与浮点误差，不是 eps 越小越好；随机方向测试不是穷举证明。再测 batch=1、非方阵、bias 广播和目标 shape。涉及 ReLU 的差分检查要避开不可导的零点，或明确导数约定。

↑ [返回题单顶部](#coding-top) · [Linear 理论](2026-08-llm-infra-interview-prep.md#pytorch-03) · [返回面试速查控制台](2026-08-llm-infra-interview-prep.md#interview-console)

---

<a id="coding-06"></a>
## CODING-06｜公平比较 eager 与 torch.compile

### 题意与计时口径

先测正确性，再区分首次调用与稳态；CUDA 异步执行，必须用 events 或正确同步计时。固定输入/shape/dtype，预热，多轮取中位数，说明是否包含数据拷贝、分配、反向和 optimizer。没有 GPU 时只做 CPU 正确性和语法检查，不报告 GPU 加速比。原理见 [GPU 计时与性能验证](2026-08-llm-infra-interview-prep.md#gpu-03) 和 [torch.compile 机制](2026-08-llm-infra-interview-prep.md#pytorch-02)。

下面是 **待在 NVIDIA CUDA 环境实测**的 Python 3 模板；输入已在 GPU，包含表达式产生输出的分配，不含 H2D、反向或 optimizer。首次调用包含编译/初始化等成本，不是纯 compiler 耗时。

```python
import sys
import time
import statistics
from importlib import metadata

print("Python:", sys.version.split()[0], "executable:", sys.executable)
try:
    import torch
except ImportError as exc:
    raise SystemExit("当前 python3 没有 PyTorch；先核对解释器") from exc
print("PyTorch:", torch.__version__, "built CUDA:", torch.version.cuda)
try:
    print("Triton:", metadata.version("triton"))
except metadata.PackageNotFoundError:
    print("Triton distribution: not found")
if torch.version.cuda is None or not torch.cuda.is_available():
    raise SystemExit("需要可用 NVIDIA CUDA GPU；跳过 GPU benchmark")
device = torch.device("cuda:0")
torch.cuda.set_device(device)
print("GPU:", torch.cuda.get_device_name(device))
print("capability:", torch.cuda.get_device_capability(device))
torch.manual_seed(0)

def eager(x, bias):
    return torch.relu(x + bias) * 0.5 + x

compiled = torch.compile(eager, backend="inductor", fullgraph=True)

def bench_ms(fn, x, bias, warmup=20, repeats=100, rounds=5):
    for _ in range(warmup):
        fn(x, bias)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(rounds):
        start.record()
        for _ in range(repeats):
            fn(x, bias)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return statistics.median(samples)

with torch.inference_mode():
    for shape in [(1, 257), (32, 1024), (1024, 4096)]:
        x = torch.randn(shape, device=device, dtype=torch.float32)
        bias = torch.randn(shape[-1], device=device, dtype=x.dtype)
        reference = eager(x, bias)
        torch.cuda.synchronize()
        before = time.perf_counter()
        actual = compiled(x, bias)
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - before
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        eager_ms = bench_ms(eager, x, bias)
        compiled_ms = bench_ms(compiled, x, bias)
        print(shape, "correctness=PASS", f"first_call={first_call_s:.3f}s",
              f"eager={eager_ms:.4f}ms compiled={compiled_ms:.4f}ms",
              f"speedup={eager_ms / compiled_ms:.2f}x")
```

### 必须会解释的限制

反复用同一数据可能命中缓存；小 shape 下 events 覆盖的区间也可能包含 CPU 发射间隙。调换测试顺序、多轮测量、换 buffer 和 profiler 能帮助定位，不能单凭一次 speedup 宣称内存带宽提高。扩成训练 benchmark 时，要补 backward/optimizer 和梯度正确性。

来源：[PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)、[Compile 排障](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)。

↑ [返回题单顶部](#coding-top) · [GPU 计时](2026-08-llm-infra-interview-prep.md#gpu-03) · [Compile 机制](2026-08-llm-infra-interview-prep.md#pytorch-02) · [环境检查](#coding-env)

---

<a id="coding-07"></a>
## CODING-07｜Triton：融合 bias＋ReLU＋residual

### 题意与前提

实现 `y=relu(x+bias)*0.5+x`，`x:[M,N]`，`bias:[N]`。先解释下面的参考实现，再遮住答案重写。它是 **FP32、连续布局、仅 forward 的教学 kernel，GPU 路径未在本地实测**，不是生产 Triton 经历。访存与融合机制见 [Kernel fusion](2026-08-llm-infra-interview-prep.md#kernel-01)。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel(X, B, Y, N: tl.constexpr, TOTAL: tl.constexpr,
                 BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < TOTAL
    x = tl.load(X + offsets, mask=valid, other=0)
    b = tl.load(B + offsets % N, mask=valid, other=0)
    y = tl.maximum(x + b, 0.0) * 0.5 + x
    tl.store(Y + offsets, y, mask=valid)

def fused_forward(x, bias):
    if (not x.is_cuda or x.ndim != 2 or bias.ndim != 1
            or bias.device != x.device or x.dtype != torch.float32
            or bias.dtype != x.dtype or not x.is_contiguous()
            or not bias.is_contiguous() or bias.numel() != x.shape[1]):
        raise ValueError("要求同设备 CUDA、FP32、连续 [M,N] 与 [N]")
    if x.requires_grad or bias.requires_grad:
        raise ValueError("教学版本没有注册 backward")
    if x.numel() >= 2**31:
        raise ValueError("教学版本限定元素数小于 2**31，避免 int32 索引溢出")
    y = torch.empty_like(x)
    if x.numel() == 0:
        return y
    with torch.cuda.device(x.device):
        fused_kernel[(triton.cdiv(x.numel(), 256),)](
            x, bias, y, x.shape[1], x.numel(), BLOCK=256)
    return y
```

### 性能、边界与反向追问

**为什么可能快**：减少中间量写回/读出和多次 launch，但 compile 也可能做同类融合，要同时比较 eager 与 compile。理想最低读写量可近似 `(2MN+N)*4` bytes，前提是 bias 被有效复用；不是实际 HBM 流量。

**测试清单**：N=1/31/32/33/255/256/257/1023/1024/1025；M=1/3/大值；负值、零、尾块、空输入；不支持的 stride/dtype 和超出 int32 索引范围的输入应明确拒绝。先限定有限输入，再约定 NaN/Inf 语义。照 [CODING-06](#coding-06) 的正确性与计时方法测试，不预填加速结果。

**反向追问**：令 `z=x+bias`，上游梯度 g，约定 ReLU 在 0 的梯度为 0：`dx=g*(1+0.5*(z>0))`，`db=sum_rows(g*0.5*(z>0))`。训练可用还要注册 autograd、处理归约/累加 dtype，并验证与 compile 的集成，不能把 forward kernel 当作完整训练算子。

来源：[Triton Vector Add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。

↑ [返回题单顶部](#coding-top) · [Kernel fusion 理论](2026-08-llm-infra-interview-prep.md#kernel-01) · [性能验证练习](#coding-06) · [环境检查](#coding-env)

---

<a id="coding-env"></a>
## 共享屏幕环境检查

先确定运行机器和解释器，再检查 PyTorch build、driver、GPU 可见性和最小 CUDA 操作，最后才处理 Triton/compile。先读报错定位层级，不立即重装全部环境。

```bash
python3 --version
python3 -m pip --version
python3 -m pip show torch triton
nvidia-smi
nvcc --version
```

命令是检查清单，不要求每台电脑都有：Mac 没有 `nvidia-smi`/CUDA 属正常情况，需远端 NVIDIA 环境。`nvidia-smi` 的 CUDA 字段主要反映 driver 可支持的 CUDA 版本范围；`torch.version.cuda` 是 PyTorch build 对应版本；`nvcc` 是本机 toolkit 编译器。三者不必完全相同，wheel 能运行也不等于能编译任意自定义扩展。

**诊断顺序**：解释器/pip 是否同环境 → driver 与 GPU 可见性/容器权限 → CUDA wheel 与库加载 → 最小 tensor 运算 → compiler/headers/ABI → Triton 缓存目录和 shape/kernel。不要在正式面试前临时升级系统驱动。

**自带电脑准备**：Python 3/NumPy 可离线运行；提前测试共享指定窗口、字体与终端。关闭无关通知，避免暴露公司代码、token、聊天或私人文件。工具和参考资料权限以面试说明为准，不把公开岗位“可上网”解释为可用 AI。

来源：[PyTorch 安装说明](https://pytorch.org/get-started/locally/)、[CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)。

↑ [返回题单顶部](#coding-top) · [返回面试速查控制台](2026-08-llm-infra-interview-prep.md#interview-console) · [岗位准备入口](2026-08-llm-infra-interview-prep.md#meshy-interview-sprint)
