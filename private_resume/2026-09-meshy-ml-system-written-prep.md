<a id="meshy-top"></a>
# Meshy ML Systems / Infra 笔试：题目与 Python 3 解答

> [返回主文档速查控制台](2026-08-llm-infra-interview-prep.md#interview-console) · [通用 Coding 题单](2026-09-interview-coding.md#coding-top)
>
> 适用：已收到的 Meshy Machine Learning Engineering 相关试卷，考试时长 **40 分钟**。按独立作答准备，是否允许 NumPy、查资料或 AI，以正式考试说明为准。本页用于考前练习，不是考试期间的辅助工具。
>
> 更新：2026-09-07。共 **22 道自拟复习题：P0 13 / P1 7 / P2 2**。公开信息只支持考察方向，**不代表下面是原卷、必考题或已确认题量**。

## 0. 先看范围，再按 topic 查题

### 公开证据到哪里为止

| 来源 | 可确认的信息 | 不能据此推断 |
|---|---|---|
| [Meshy 官方 ML System Engineering 招聘页](https://jobs.ashbyhq.com/meshy/90988ed5-f767-4c0d-9cbc-b69d792db1a9) | 线上测评涉及 training、inference、Transformer 与简单 NumPy 编程 | 对应湾区岗位，不能保证与国内这份试卷相同；后续技术轮允许查资料不等于这轮允许 |
| [2026-07-29 候选人 OA 分享](https://www.1point3acres.com/bbs/thread-1184421-1-1.html) | 公开部分提到 Attention 编程和 ML 概念；楼主后续自述面的是 Infra | 其余内容有权限限制，Attention 变体、完整题目和时长未核验；个人经历不是官方承诺 |
| [FastPrep Attention 练习](https://www.fastprep.io/problems/meshy-scaled-dot-product-attention) | 页面提供 single-head SDPA 练习，并注明是改编版本 | 不能把其接口、约束或示例当作 Meshy 原题 |

**复习顺序**：先 A01–A03，确保独立写出 NumPy Attention；再读 B/C 的 P0 概念和 D 的 P0 系统题。已有余力再练多头、LayerNorm、交叉熵梯度。不要把 22 题都当成一次 40 分钟试卷。

| Topic | P0：优先掌握 | P1：变体与深入 | P2：时间允许再看 |
|---|---|---|---|
| **A｜Python 3 / NumPy 编程** | [A01 数组与广播](#meshy-a01) · [A02 稳定 softmax](#meshy-a02) · [A03 Attention](#meshy-a03) | [A04 多头 Attention](#meshy-a04) · [A05 LayerNorm](#meshy-a05) · [A06 交叉熵与梯度](#meshy-a06) | — |
| **B｜Transformer** | [T01 Block 数据流](#meshy-t01) · [T02 缩放与复杂度](#meshy-t02) | [T03 MHA/GQA/MQA](#meshy-t03) · [T04 归一化](#meshy-t04) | [T05 位置编码](#meshy-t05) |
| **C｜训练与推理** | [M01 反传与优化器](#meshy-m01) · [M02 损失函数](#meshy-m02) · [M03 梯度累积](#meshy-m03) · [M04 混合精度](#meshy-m04) · [M05 训练/评估模式](#meshy-m05) · [M06 KV Cache](#meshy-m06) | — | — |
| **D｜ML Systems / Infra** | [S01 分布式与通信](#meshy-s01) · [S02 显存与 OOM](#meshy-s02) | [S03 三种加速机制](#meshy-s03) · [S04 数据与计时](#meshy-s04) | [S05 Diffusion/DiT](#meshy-s05) |

[40 分钟闭卷自测](#meshy-mock) · [资料与代码验证说明](#meshy-sources)

## A｜Python 3 / NumPy 编程（6 题，自拟复习题）

代码是 **CPU 教学参考实现**，不是生产高性能 kernel。NumPy 版本统一用 `float64` 便于测试，不意味着 GPU 训练应使用 FP64。A01–A06 的 Python 代码块可按顺序放到同一脚本执行；**A04 复用 A03 的 `attention`**。A03 另附不依赖 NumPy 的标准库版。

<a id="meshy-a01"></a>
### MESHY-A01｜数组的 shape、广播、axis 和 view/copy（P0）

**题目**：`x.shape == (2, 3, 4)`，如何沿最后一维求均值并中心化？`(2,3,4)` 与 `(4,)` 能否相加？`(2,3)` 能否直接作为 Attention 的 key mask？切片后原数组会不会改变？

**考察点**：把数学轴映射到代码，而不是靠试错调整 shape。

**解答**：广播从右往左对齐，每一维必须相等或其中一个是 1。`axis=-1` 指最后一维，`keepdims=True` 保留归约轴以便广播。mask 要按 batch/head/query/key 的语义显式扩维，不能仅因为形状碰巧可广播就认为正确。

```python
import numpy as np

x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
mean = x.mean(axis=-1, keepdims=True)
centered = x - mean
assert mean.shape == (2, 3, 1)
np.testing.assert_allclose(centered.mean(axis=-1), 0)
assert (x + np.ones(4)).shape == (2, 3, 4)

# [B, S] 的 key 可见性，扩成 [B, 1, 1, S]，供 [B, H, L, S] 广播。
key_keep = np.array([[True, True, False], [True, False, False]])
assert key_keep[:, None, None, :].shape == (2, 1, 1, 3)

a = np.arange(6)
view = a[1:4]        # 基本切片通常共享内存。
view[0] = 99
assert a[1] == 99
copied = a[[1, 2]]   # 高级索引返回副本。
copied[0] = -1
assert a[1] == 99
```

**易错**：`reshape` 不等于交换轴；`transpose` 后再 reshape 可能需要复制；广播不复制输入，不代表计算结果不占新内存。`np.dot` 与 `np.matmul` 对高维数组的语义不同，batched Attention 优先用 `@`/`matmul`。

↑ [返回速查](#meshy-top)

<a id="meshy-a02"></a>
### MESHY-A02｜实现数值稳定的 softmax（P0）

**题目**：给定非空、有限实数数组，沿指定轴计算 softmax；输入如 `[1000,1001]` 时不能指数溢出。

**考察点**：减最大值、归约维度、数值稳定性。公式为 `exp(x − max(x)) / sum(exp(x − max(x)))`，减同一个常数不改变 softmax。

```python
import numpy as np

def stable_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("x 必须是非空、有限实数数组")
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

p = stable_softmax([[1000, 1001], [-1000, -999]])
np.testing.assert_allclose(p.sum(axis=-1), 1.0)
np.testing.assert_allclose(p[0], p[1])
np.testing.assert_allclose(p, stable_softmax([[0, 1], [0, 1]]))
```

**复杂度**：数组有 `N` 个元素，时间与这份实现的额外空间均为 `O(N)`。

**易错**：不减最大值；除以全数组的 sum；漏掉 `keepdims`。本题不接收 `NaN/Inf`；mask 产生的全 `-inf` 行另见 A03，不能直接代入本函数。

↑ [返回速查](#meshy-top)

<a id="meshy-a03"></a>
### MESHY-A03｜实现 Scaled Dot-Product Attention（P0）

**题目**：输入 `Q:[...,L,Dk]`、`K:[...,S,Dk]`、`V:[...,S,Dv]`，计算输出 `[...,L,Dv]`。先实现二维单头，再支持 batch 和可选 bool mask。

**考察点**：`softmax(QKᵀ / sqrt(Dk))V`；softmax 沿 **key 轴**；注意交叉注意力中 `L` 不必等于 `S`。

**本题约定**：`keep_mask=True` 表示允许关注，shape 必须可广播到 score 的形状；全屏蔽行返回零输出。这是本练习的显式约定，不假定所有库都相同。它与通用 Coding 题单中“`True` 表示屏蔽”的约定相反，跨题复制前必须核对。

**NumPy 参考解答**：

```python
import numpy as np

def attention(q, k, v, keep_mask=None):
    q, k, v = [np.asarray(t, dtype=np.float64) for t in (q, k, v)]
    if any(t.ndim < 2 or t.size == 0 for t in (q, k, v)):
        raise ValueError("Q/K/V 必须至少二维且非空")
    if not all(np.isfinite(t).all() for t in (q, k, v)):
        raise ValueError("Q/K/V 必须是有限实数")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("Q/K 的特征维、K/V 的序列维必须分别匹配")

    scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(q.shape[-1])
    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask)
        if keep_mask.dtype != np.bool_:
            raise TypeError("keep_mask 必须为 bool，True 表示可见")
        keep_mask = np.broadcast_to(keep_mask, scores.shape)
        scores = np.where(keep_mask, scores, -np.inf)

    row_max = scores.max(axis=-1, keepdims=True)
    # 全屏蔽行的 max 是 -inf；改用 0 平移，使 exp(-inf) = 0。
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    weights = np.exp(scores - row_max)
    denominator = weights.sum(axis=-1, keepdims=True)
    weights = np.divide(
        weights, denominator,
        out=np.zeros_like(weights), where=denominator != 0,
    )
    return weights @ v

# Q 全零 → 每个 key 权重相同 → 输出为 V 的平均值。
np.testing.assert_allclose(
    attention(np.zeros((1, 2)), np.ones((2, 2)), [[2, 4], [4, 8]]),
    [[3, 6]],
)

# 无 KV Cache、Q/K 位置从 0 对齐的 self-attention causal mask。
keep = np.tril(np.ones((3, 3), dtype=bool))
np.testing.assert_allclose(
    attention(np.zeros((3, 2)), np.zeros((3, 2)), [[1], [3], [8]], keep),
    [[1], [2], [4]],
)
keep[1] = False
assert attention(np.zeros((3, 2)), np.zeros((3, 2)), np.ones((3, 1)), keep)[1, 0] == 0
```

**代码边界**：教学实现完整 materialize scores，不是 FlashAttention；假设数值量级使点积在所用 dtype 中可表示。增量 decode 带 KV Cache 时，causal mask 必须考虑 query 的绝对位置，不能照搬左上角 `tril(L,S)`。

**复杂度**：每个 batch/head 的时间为 `O(L×S×Dk + L×S×Dv)`，Attention 矩阵空间为 `O(L×S)`；self-attention 中 `L=S=N`，因此随序列长度平方增长。

<details>
<summary>只提供 Python 3 标准库时：二维、无 mask 版本</summary>

以下为同一数学计算的备用写法，输入为非空规则二维列表，数值须有限且运算不溢出。它不包含投影、多头、batch 或 mask。

```python
import math

def attention_python(q, k, v):
    if not q or not k or not v or not q[0] or not v[0]:
        raise ValueError("输入必须非空")
    d, dv = len(q[0]), len(v[0])
    if (len(k) != len(v)
            or any(len(row) != d for row in q + k)
            or any(len(row) != dv for row in v)):
        raise ValueError("输入维度不匹配")
    result = []
    for qi in q:
        scores = [sum(a * b for a, b in zip(qi, kj)) / math.sqrt(d)
                  for kj in k]
        maximum = max(scores)
        weights = [math.exp(s - maximum) for s in scores]
        denominator = sum(weights)
        weights = [w / denominator for w in weights]
        result.append([
            sum(weights[j] * v[j][c] for j in range(len(v)))
            for c in range(dv)
        ])
    return result

assert attention_python([[0, 0]], [[1, 2], [3, 4]], [[2, 4], [4, 8]]) == [[3.0, 6.0]]
```

该版按 query 逐行计算，除输出外额外空间为 `O(S)`；Python 循环开销较大，不把它当作高性能实现。

</details>

**常见失分**：用 `K.T` 转置高维数组；softmax 沿 query 轴；缩放除以模型总宽度而非每头 `Dk` 的平方根；把 bool mask 的 True/False 含义写反；全屏蔽行产生 NaN。

↑ [返回速查](#meshy-top)

<a id="meshy-a04"></a>
### MESHY-A04｜从单头扩展到 Multi-Head Self-Attention（P1）

**题目**：输入 `X:[B,S,D]`、四个投影矩阵 `Wq/Wk/Wv/Wo:[D,D]` 和头数 `H`，输出 `[B,S,D]`。为聚焦 shape，本题不含 bias、dropout、residual 和 normalization。

**解题思路**：投影 → `[B,S,H,D/H]` → 转轴为 `[B,H,S,D/H]` → 调用 A03 → 转轴合并 heads → 输出投影。不是把序列长度切成 H 份。

```python
import numpy as np

def multi_head_attention(x, wq, wk, wv, wo, num_heads, keep_mask=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("x 应为 [B,S,D]")
    batch, length, width = x.shape
    if (not isinstance(num_heads, (int, np.integer))
            or num_heads <= 0 or width == 0 or width % num_heads):
        raise ValueError("D 必须能被正整数头数整除")
    wq, wk, wv, wo = [np.asarray(w, dtype=np.float64) for w in (wq, wk, wv, wo)]
    if any(w.shape != (width, width) for w in (wq, wk, wv, wo)):
        raise ValueError("本题四个投影矩阵均为 [D,D]")
    head_dim = width // num_heads
    q, k, v = [
        (x @ w).reshape(batch, length, num_heads, head_dim).transpose(0, 2, 1, 3)
        for w in (wq, wk, wv)
    ]
    heads = attention(q, k, v, keep_mask)  # 先定义 A03 的 attention。
    merged = heads.transpose(0, 2, 1, 3).reshape(batch, length, width)
    return merged @ wo

rng = np.random.default_rng(0)
x = rng.normal(size=(2, 3, 4))
identity = np.eye(4)
out = multi_head_attention(x, identity, identity, identity, identity, 1)
np.testing.assert_allclose(out, attention(x, x, x))
assert multi_head_attention(x, identity, identity, identity, identity, 2).shape == x.shape
```

**复杂度**：四个投影合计量级 `O(B×S×D²)`，Attention 为 `O(B×S²×D)`；显式 score 空间为 `O(B×H×S²)`。

**易错**：merge heads 前忘记 transpose；padding mask 应扩为 `[B,1,1,S]`，不能直接用 `[B,S]`；causal mask `[S,S]` 可以在 batch/head 上广播。完整 PyTorch 模块与更多测试见 [通用 MHA 题](2026-09-interview-coding.md#coding-01)。

↑ [返回速查](#meshy-top)

<a id="meshy-a05"></a>
### MESHY-A05｜实现 LayerNorm（P1）

**题目**：对 `X:[...,D]` 每个向量的最后一维做 LayerNorm，带 `gamma/beta:[D]`。

**解答**：`y = (x − mean) / sqrt(var + eps) × gamma + beta`；方差沿特征轴取总体方差，不是样本方差。RMSNorm 不减均值，分母改为 `sqrt(mean(x²)+eps)`。

```python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    gamma, beta = np.asarray(gamma), np.asarray(beta)
    if x.ndim == 0 or x.size == 0 or eps <= 0:
        raise ValueError("x 必须非空且 eps > 0")
    if gamma.shape != (x.shape[-1],) or beta.shape != (x.shape[-1],):
        raise ValueError("gamma/beta 应为 [D]")
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

y = layer_norm([[1, 2, 3], [5, 5, 5]], np.ones(3), np.zeros(3))
np.testing.assert_allclose(y.mean(axis=-1), 0, atol=1e-12)
np.testing.assert_allclose(y[1], 0)
```

**考点/易错**：不是沿 batch 轴求统计量；`eps` 放在根号内；常数输入不能除零。时间/额外空间均随输入元素数线性增长。输入限有限实数、正常可表示量级。

↑ [返回速查](#meshy-top)

<a id="meshy-a06"></a>
### MESHY-A06｜从 logits 算交叉熵，并写出 logits 梯度（P1）

**题目**：输入 `logits:[N,C]` 和类别索引 `labels:[N]`，返回 batch mean loss 及 `dLoss/dLogits`。本题无 label smoothing、ignore_index 或类别权重。

**解答**：先算稳定 log-softmax，取每行目标类别的负 log 概率。均值损失的梯度是 `(softmax(logits) − one_hot(labels)) / N`；不要先 softmax 再对可能下溢到零的概率取 log。

```python
import numpy as np

def cross_entropy_with_grad(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 2 or logits.size == 0 or not np.isfinite(logits).all():
        raise ValueError("logits 必须是非空有限数组 [N,C]")
    n, classes = logits.shape
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels 必须是整数索引 [N]")
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("类别索引越界")

    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    loss = -log_probs[np.arange(n), labels].mean()
    grad = np.exp(log_probs)
    grad[np.arange(n), labels] -= 1.0
    grad /= n
    return float(loss), grad

loss, grad = cross_entropy_with_grad([[0, 0], [0, 0]], [0, 1])
np.testing.assert_allclose(loss, np.log(2))
np.testing.assert_allclose(grad, [[-0.25, 0.25], [0.25, -0.25]])
np.testing.assert_allclose(grad.sum(axis=-1), 0)

# 对其中一个 logit 做中心差分，检查手写梯度。
z = np.array([[0.2, -0.1], [0.4, 0.6]])
eps = 1e-5
plus, minus = z.copy(), z.copy()
plus[0, 1] += eps
minus[0, 1] -= eps
numerical = (cross_entropy_with_grad(plus, [0, 1])[0]
             - cross_entropy_with_grad(minus, [0, 1])[0]) / (2 * eps)
np.testing.assert_allclose(cross_entropy_with_grad(z, [0, 1])[1][0, 1], numerical, atol=1e-8)
```

**考点/易错**：沿类别轴归一化；mean loss 梯度要除以 `N`；类别标签不是概率。若模型 logits 为 `XW+b`，下一层链式法则是 `dW=Xᵀ·dLogits`、`db=sum(dLogits, axis=0)`。时间与额外空间均为 `O(NC)`。

↑ [返回速查](#meshy-top)

## B｜Transformer 基础（5 题，自拟复习题）

<a id="meshy-t01"></a>
### MESHY-T01｜一个 Transformer block 的数据怎样流动？（P0）

**简答**：以 Pre-Norm decoder block 为例，输入先归一化，再投影为 Q/K/V；Attention 在 causal mask 约束下混合不同 token 的信息，经输出投影后做残差相加。随后归一化、逐 token 的 MLP、再次残差相加。Attention 负责跨 token 交互，MLP 负责特征变换；block 输入输出通常同为 `[B,S,H]`。残差保留原表示，也提供更直接的梯度路径。

忽略 dropout：`u = x + Attention(Norm(x))`；`y = u + MLP(Norm(u))`。

**考点/易错**：先声明 Pre-Norm；原始 Transformer 的 Post-Norm 次序不同，MLP 不直接混合不同 token。[结构依据](https://docs.pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoderLayer.html)

[返回速查](#meshy-top)

<a id="meshy-t02"></a>
### MESHY-T02｜Attention 为什么除以 √d？时间与空间复杂度是多少？（P0）

**简答**：若 Q、K 各维近似独立、零均值且方差为一，点积方差随 head dimension `d` 增长；除以 `√d` 让分数尺度较稳定，避免 softmax 过度饱和。缩放改变分数尺度，不改变可见关系。长度为 `S` 时，每个 query 要与全部 key 比较，因此标准全序列 Attention 核心计算随 `S²` 增长；显式保存分数矩阵也占二次空间。

`Attention(Q,K,V) = softmax(QKᵀ / √d + mask)V`。设 `H=h·d`：核心计算 `O(BS²H)`；分数矩阵 `O(BhS²)`；QKV/输出投影另有 `O(BSH²)`。

**考点/易错**：缩放维度是每个 head 的 `d`，不是总 hidden size；causal mask 省常数，不改变二次阶。[原论文 §3.2、§4](https://arxiv.org/html/1706.03762v7)

[返回速查](#meshy-top)

<a id="meshy-t03"></a>
### MESHY-T03｜MHA、GQA、MQA 有何区别，为什么影响推理？（P1）

**简答**：MHA 中每个 query head 有对应的 K/V head；MQA 让所有 query heads 共享一组 K/V；GQA 则让一组 query heads 共享一组 K/V，介于两者之间。减少 KV heads 能缩小 KV cache 和 decode 时的 KV 读取量，但 query heads 仍分别计算 Attention，不能把模型全部计算量按 KV heads 数同比缩小。共享数量由模型结构规定，不能推理时随意改动。

同层数、长度、head dimension 和 dtype 下，KV cache 大小正比于 `num_kv_heads`。

**考点/易错**：共享的是 K/V，不是把多个 Q head 合成一个；速度和质量取舍要实测。[GQA 原论文](https://arxiv.org/abs/2305.13245)

[返回速查](#meshy-top)

<a id="meshy-t04"></a>
### MESHY-T04｜LayerNorm、RMSNorm、BatchNorm 分别归一化什么？（P1）

**简答**：对 `[B,S,H]` 的 Transformer hidden states，常见 LayerNorm 在每个 token 的 `H` 维上减均值、除标准差；RMSNorm 不减均值，只按均方根缩放。BatchNorm 则按 channel 汇集 batch、必要时还有空间或序列维的统计，默认在推理使用 running statistics。前两者不依赖其他样本的 batch 统计，更便于变长序列和小 batch。比较时要先写清统计轴。

忽略可学习仿射项：`LN(x)=(x−mean(x))/√(var(x)+ε)`；`RMSNorm(x)=x/√(mean(x²)+ε)`。

**考点/易错**：RMSNorm 不保证零均值；LayerNorm/RMSNorm 不会因 `eval()` 改用历史统计。[LayerNorm](https://docs.pytorch.org/docs/2.9/generated/torch.nn.LayerNorm.html) · [RMSNorm](https://docs.pytorch.org/docs/2.9/generated/torch.nn.RMSNorm.html) · [BatchNorm](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BatchNorm1d.html)

[返回速查](#meshy-top)

<a id="meshy-t05"></a>
### MESHY-T05｜为什么需要位置信息？RoPE 怎样表达相对位置？（P2）

**简答**：没有位置或其他顺序信息时，self-attention 对输入排列等变，无法仅凭 token 内容区分词序。绝对位置编码把位置向量加入表示；RoPE 通常按位置对 Q、K 的二维分量做旋转，使二者点积包含相对位置差。它不等于直接加一个位置向量，也不意味着模型能无代价泛化到任意长上下文。

旋转矩阵满足 `R(m)ᵀR(n)=R(n−m)`，所以旋转后的 `qₘᵀkₙ` 能依赖相对位移。

**考点/易错**：causal mask 限制可见范围，RoPE 编入位置信息，两者不能互相替代。[RoFormer 原论文](https://arxiv.org/html/2104.09864v5)

[返回速查](#meshy-top)

## C｜训练与推理基础（6 题，自拟复习题）

<a id="meshy-m01"></a>
### MESHY-M01｜反向传播、SGD 和 AdamW 各做什么？（P0）

**简答**：反向传播用链式法则计算 loss 对参数的梯度，本身不更新权重。SGD 沿负梯度方向更新，可加 momentum；Adam 维护梯度一阶矩和二阶矩的移动平均，经偏差修正后自适应缩放更新。AdamW 把 weight decay 与梯度自适应更新解耦。PyTorch 的梯度默认累积，需在适当的更新边界清理。

无 momentum 的 SGD：`θ ← θ − ηg`；AdamW 的衰减部分：`θ ← (1−ηλ)θ`，再施加 Adam 更新。

**考点/易错**：`backward()` 不等于 `optimizer.step()`；AdamW 的 weight decay 不等于给 Adam 的梯度直接加 `λθ`。[Autograd](https://docs.pytorch.org/docs/2.9/notes/autograd.html) · [AdamW](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html)

[返回速查](#meshy-top)

<a id="meshy-m02"></a>
### MESHY-M02｜分类为什么常用 Cross-Entropy，回归为什么常用 MSE？（P0）

**简答**：Cross-Entropy 比较目标类别分布与预测分布，单个硬标签时就是正确类别概率的负对数；MSE 最小化预测值与连续目标的平方误差，目标尺度和离群值会影响它。PyTorch 的 `CrossEntropyLoss` 接收未归一化 logits，内部使用稳定的 log-softmax 形式；不要先 softmax 再传入，也不要先 argmax。

硬标签 CE：`L = logsumexp(z) − z[y]`；MSE：`L = mean((prediction−target)²)`。

**考点/易错**：CE 的预测输入是 logits，但 target 可以是类别索引，也可以是合法的 soft-label 概率分布。[CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html) · [MSELoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.MSELoss.html)

[返回速查](#meshy-top)

<a id="meshy-m03"></a>
### MESHY-M03｜梯度累积怎样模拟大 batch，loss 应该如何归一化？（P0）

**简答**：把一个有效 batch 拆成多个 microbatches，分别 backward、累加梯度，最后只更新一次参数。若各 microbatch 有相同数量的有效样本，mean loss 可再除以累积步数；变长 token 数不等时，应按整个有效 batch 的有效 token 总数归一，不能简单平均各 microbatch 的 mean。中途不要清梯度或更新参数。

等大且完整的 microbatch 下：`global_batch = micro_batch_per_rank × accumulation_steps × DP_size`。

**考点/易错**：多卡默认梯度平均也要计入归一化；BatchNorm、随机性会使累积不严格等同一次大 batch，不能为累积一直保留计算图。[AMP 累积规则](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html#gradient-accumulation)

[返回速查](#meshy-top)

<a id="meshy-m04"></a>
### MESHY-M04｜FP16、BF16、FP8 的取舍是什么，loss scaling 解决什么？（P0）

**简答**：FP16 与 BF16 都占两字节；BF16 指数范围更大、尾数更短，通常比 FP16 更不易溢出，但不代表更精确。FP16 训练常用 loss scaling 缓解梯度下溢，更新或裁剪前要还原尺度。FP8 进一步节省存储与计算成本，却需要缩放策略、合适算子和硬件支持，通常保留高精度累加及敏感状态，并验证收敛。

**考点/易错**：BF16 不保证无 NaN；FP8 不是把所有参数、梯度、optimizer state 一次性 `.cast` 成八位。[NVIDIA 精度机制](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) · [PyTorch AMP](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html)

[返回速查](#meshy-top)

<a id="meshy-m05"></a>
### MESHY-M05｜`train()`、`eval()`、`no_grad()` 有何区别？怎样判断过拟合？（P0）

**简答**：`train()/eval()` 切换 Dropout、BatchNorm 等模块的行为；`eval()` 不关闭 autograd，`no_grad()` 不自动切换评估模式，验证通常两者一起用。若在可比口径下训练 loss 继续下降、验证 loss 持续上升，才有过拟合迹象；先排除数据分布、预处理与评估模式问题，再考虑正则化、数据增广或 early stopping。

**考点/易错**：训练集与验证集不能泄漏；验证指标更差不一定就是过拟合，也不能靠测试集反复调参。[PyTorch 模式边界](https://docs.pytorch.org/docs/2.9/notes/autograd.html#evaluation-mode-nn-module-eval)

[返回速查](#meshy-top)

<a id="meshy-m06"></a>
### MESHY-M06｜训练、prefill、decode 有何区别？KV cache 缓存什么？（P0）

**简答**：常规自回归训练用真实前缀并行计算各位置的 next-token loss，再反向更新权重。推理时 prefill 并行处理 prompt；decode 逐步生成新 token，复用各层历史 K/V，避免重新计算旧 token 的这些表示。Prefill 通常更偏计算密集，低 batch decode 常更受权重/KV 访存与调度限制；具体瓶颈随长度和并发变化。

常规全注意力、无分片与额外副本时：`KV_bytes ≈ 2 × layers × batch × cached_tokens × kv_heads × head_dim × bytes_per_element`。

**考点/易错**：KV cache 不保存 Q、Attention 概率或最终答案；它避免历史表示重算，但新 Q 仍要读取历史 KV。[GQA 机制](https://arxiv.org/abs/2305.13245) · [主文档：推理指标与 KV](2026-08-llm-infra-interview-prep.md#infra-06)

[返回速查](#meshy-top)

## D｜Infra 与性能判断（5 题，自拟复习题）

<a id="meshy-s01"></a>
### MESHY-S01｜DDP、FSDP、TP 分别切什么，主要通信是什么？（P0）

**简答**：DDP 让各 rank 保留模型副本、处理不同数据，通常用 all-reduce 同步梯度。FSDP full sharding 在数据并行组内分片参数、梯度与 optimizer state，计算前 all-gather 所需参数，反向后 reduce-scatter 梯度。TP 切分同一层的矩阵或 heads，让多卡共同计算一个 layer，并在层内交换或归约结果。三者可以组合，但通信频率和显存收益不同。

**考点/易错**：FSDP 不是 TP；状态分片不会把 activation 也自动除以卡数，TP collective 取决于层布局。[主文档：FSDP 与 TP 边界](2026-08-llm-infra-interview-prep.md#dist-01)

[返回速查](#meshy-top)

<a id="meshy-s02"></a>
### MESHY-S02｜怎样估算训练显存？第一次遇到 OOM 先做什么？（P0）

**简答**：分开计算参数、梯度、optimizer state 等常驻状态，以及同时存活的 activation、logits、通信 buffer 和 workspace。每块张量按 shape 乘 dtype bytes，结合实际分片和生命周期，不能把不同阶段峰值全部相加。OOM 先定位在 forward、backward 还是 optimizer，再查最大分配与实际 shape，选择减小 microbatch、重计算或状态分片。

手算例：全部 FP32 的 Adam，参数 `4P`、梯度 `4P`、两个矩 `8P`，基本状态共 `16P bytes`；尚未计 activation 和临时内存，混合精度应按真实副本重算。

**考点/易错**：`reserved−allocated` 不全是碎片；`empty_cache()` 不会释放仍存活的张量。[主文档：显存账本与 OOM](2026-08-llm-infra-interview-prep.md#infra-02)

[返回速查](#meshy-top)

<a id="meshy-s03"></a>
### MESHY-S03｜FlashAttention、CUDA Graph、`torch.compile` 各优化什么？（P1）

**简答**：FlashAttention 用分块和 online softmax 降低 Attention 的 HBM 往返，不写出完整分数矩阵，数学计算仍是精确 Attention。CUDA Graph 回放已捕获的执行图，主要减少 CPU 重复提交开销。`torch.compile` 捕获并优化计算图，由后端做融合等优化。三者可以组合，但支持路径、动态 shape、编译或捕获成本都要单独验收。

**考点/易错**：FlashAttention 不把全 Attention 算量变线性；CUDA Graph 不自动减少 GEMM FLOPs；compile 不保证全图无中断。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135) · [PyTorch 编译与融合](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations) · [主文档：CUDA Graph](2026-08-llm-infra-interview-prep.md#resume-13)

[返回速查](#meshy-top)

<a id="meshy-s04"></a>
### MESHY-S04｜GPU 吃不满怎么排查？怎样证明吞吐优化有效？（P1）

**简答**：先看时间线，区分取数等待、CPU 预处理、H2D、GPU 计算和通信空洞。数据瓶颈再调 workers、预取和 pinned memory；不是 workers 越多越快。比较时固定模型、有效 token 数、长度分布、精度与硬件，先 warmup；GPU 局部计时用正确同步的 CUDA events，端到端计时包括数据与等待，并复验数值和显存。

`训练有效吞吐 = 实际参与训练的有效 tokens / 稳态端到端秒数`；padding 不应冒充有效 tokens。

**考点/易错**：CUDA 异步提交的 CPU 耗时不是 kernel 耗时；局部加速不能直接写成端到端同倍数收益。[数据加载与调优](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) · [CUDA 计时语义](https://docs.pytorch.org/docs/2.9/notes/cuda.html#asynchronous-execution)

[返回速查](#meshy-top)

<a id="meshy-s05"></a>
### MESHY-S05｜Diffusion/DiT 与自回归 Transformer 的训练、推理有什么不同？（P2）

**简答**：常见 diffusion 训练随机采样噪声等级，对带噪样本学习去噪目标；推理从噪声或带噪状态出发多步更新。DiT 把 Transformer 用作扩散模型的去噪骨干，并不等于 next-token 模型。自回归训练按条件概率分解预测下一个 token，常规生成逐 token 推进；两者都可能重复调用网络，但依赖结构、loss 和缓存复用条件不同。

**考点/易错**：diffusion 一次训练样本通常不用跑完整采样链；不能把自回归 KV cache 的正确性直接套给每步输入都变化的 DiT。[DiT 原论文](https://arxiv.org/html/2212.09748v2) · [主文档：视频 DiT](2026-08-llm-infra-interview-prep.md#resume-18)

[返回速查](#meshy-top)

<a id="meshy-mock"></a>
## E｜一次 40 分钟闭卷自测

这只是自拟模拟安排，不预测正式试卷配比。先隐藏答案，不开 AI；完成后再用本页对照。

| 用时 | 任务 | 自查标准 |
|---:|---|---|
| 20 分钟 | [A03](#meshy-a03)：从零写二维 Attention，完成后加 batch 或 causal mask | shape 正确、softmax 沿 key 轴、数值稳定、至少三个测试 |
| 15 分钟 | 从 [T01](#meshy-t01)、[T02](#meshy-t02)、[M03](#meshy-m03)、[M04](#meshy-m04)、[M06](#meshy-m06)、[S01](#meshy-s01) 中任选五题 | 每题用 2–4 句话说清机制，至少指出一个易错点 |
| 5 分钟 | 检查代码与遗漏 | 接口/返回值、依赖、mask 约定、除数、维度与边界条件 |

**最后只记这条编程顺序**：输入输出 → 数学式 → 每一步 shape → 最小实现 → 数值稳定 → 边界测试。正式考试不先写复杂的通用框架，按实际题目实现必要范围。

<a id="meshy-sources"></a>
## F｜资料与代码验证

- Python 环境：以上代码以 Python 3 + NumPy 编写；按顺序复制所有 `python` 代码块到同一脚本即可运行文内断言。A03 的 `attention_python` 只需标准库 `math`。不需要 PyTorch、GPU 或联网。
- 代码校验（2026-09-07）：在 Python 3.10.19 / NumPy 2.2.6 下，文内全部示例断言及 12 项额外测试通过，覆盖稳定性、mask、batch/non-contiguous 输入、多头拆分及交叉熵有限差分。测试结果不代表生产性能或与所有后端完全一致。
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) 与 [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)：数组、广播、归约和拷贝语义。
- [PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)：Attention 机制与优化实现；不要混淆不同 API 的 bool mask 语义。
- [PyTorch FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)：参数分片、all-gather、reduce-scatter。
- [PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) 与 [torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)：计时与编译的正确使用。
- 本页系统机制沿用 [主文档](2026-08-llm-infra-interview-prep.md#interview-console) 已校准的技术口径。公司相关考察证据见页首来源表；上述官方技术资料用于解释原理，不是 Meshy 命题证明。

↑ [返回速查](#meshy-top) · [返回主文档](2026-08-llm-infra-interview-prep.md#interview-console)
