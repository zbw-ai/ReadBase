# MOPD 原理专题 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一篇从传统蒸馏递进到 OPD、再到 MOPD 的中文原理专题，并以克制的方式接入现有 RL Infra 导航。

**Architecture:** 单篇 `topics/mopd.md` 承载第一阶段内容，显式区分算法核心、论文 recipe 与框架实现。导航只接入 Agentic RL、Master Reading List 和 Knowledge Graph 文字索引；根 README 与独立 paper note 留到专题进入 `DIGESTED` 后。

**Tech Stack:** Markdown、相对链接、一手来源（arXiv、GitHub、NVIDIA 官方文档）、仓库现有链接检查。

---

## Chunk 1: 专题正文

### Task 1: 建立 OPD → MOPD 原理主线

**Files:**
- Create: `training-infra-roadmap/topics/mopd.md`
- Reference: `docs/superpowers/specs/2026-07-24-mopd-topic-design.md`
- Reference: `training-infra-roadmap/topics/agentic_rl.md`

- [ ] **Step 1: 记录实施前工作区基线**

Run:

```bash
git status --short
```

Expected: 保留已有的 `projects/2026-q3-long-context-agentic-rl/STATUS.md` 和 `.superpowers/` 变化，并记录本专题已经新增的 spec/plan 文件。

- [ ] **Step 2: 创建专题定位和问题定义**

写入：

- 状态 `READING`。
- MOPD 的目标是将多个 domain-specialized RL Teachers 的能力集成进一个 Student。
- 第一版只讲原理，不把 verl/NeMo 的系统实现当作算法定义。
- 用以下工程矛盾开篇：

```text
每个领域都能独立训练出更强的 RL Teacher，
为什么一个统一模型仍然难以同时继承所有能力？
```

- 后续每一节必须回答“上一种方案为什么还不够”，不能退化成术语和定义罗列。

- [ ] **Step 3: 精确区分三类蒸馏**

正文必须分别解释：

```text
Fixed-corpus logit KD
Teacher-generated trajectory distillation
Student-rollout On-Policy Distillation
```

关键判断：

```text
训练 prefix 来自固定语料或 Teacher，
推理 prefix 来自 Student 自己，
两者不同才是 state-distribution mismatch 的来源。
```

- [ ] **Step 4: 写出 OPD 最小数据流**

沿一个 prompt 明确列出：

```text
Student input/output
Teacher input/output
Trainer input/output
```

解释 Teacher 是在 Student 已经生成的 prefix/trajectory 上做 prefill/scoring，而不是重新生成一条 Teacher trajectory。

- [ ] **Step 5: 加入最小公式**

只保留：

```text
A_t = stop_gradient(
  log π_teacher(y_t | x, y_<t)
  - log π_student(y_t | x, y_<t)
)
```

解释：

- `y_t` 来自 Student rollout。
- 正 advantage 表示 Teacher 比 Student 更认可该 sampled token。
- 该信号可进入 policy-gradient loss。
- 不展开 reverse-KL 完整推导。

- [ ] **Step 6: 从 OPD 推进到 MOPD**

明确：

- 每个领域有独立 Teacher。
- routing 使用 prompt/domain metadata。
- 一个样本路由到一个 Teacher。
- 不是 learned router。
- 不是多个 Teacher logits ensemble。

- [ ] **Step 7: 区分算法、recipe 与实现**

先在正文中区分算法核心与论文 recipe：

```text
算法核心：
Student rollout → selected Teacher scoring → distillation update

论文 recipe：
General SFT → same-origin domain RL Teachers → MOPD integration
```

框架实现统一放到 Step 11 的“与 RL Infra 的关系”，避免读者把部署方式当成算法必要条件。

- [ ] **Step 8: 增加能力融合方案对照表**

表格必须覆盖：

```text
Mix-RL
Cascade RL
Off-Policy Finetune
Param-Merge
MOPD
```

比较维度：

```text
监督密度
训练数据分布
领域训练是否串行耦合
能力融合空间
主要风险
```

- [ ] **Step 9: 写清 same-origin 与方法边界**

事实：

```text
共同 SFT 起点 → 初始 Student/Teacher policy gap 更小
```

论文观察：

```text
更小初始 KL 对应更稳定的优化；
分布差异大的外部 Teacher 会出现更明显不稳定。
```

推断必须标记：

```text
较小 KL 可能使 token advantage 极值和梯度方差更可控。
```

同时补充错误路由、Teacher 冲突、控制 token 放大、Teacher 资源线性增长等边界。

解释 `dense supervision` 与模块化领域训练的作用：

- Teacher 在每个训练 token 上提供信号，而不是只返回 trajectory-level scalar reward。
- 各领域 Teacher 可以独立选择 RL recipe、超参数和环境，并行开发后再做统一能力集成。

- [ ] **Step 10: 将 Top-k 降为实现变体侧栏**

只比较：

- sampled-token policy-gradient：payload 小、方差较大。
- Top-k distillation：利用更多 Teacher 分布、方差较低、payload 更大。
- 独立论文的 Top-k objective 包含 bias-correction terms。

- [ ] **Step 11: 补充与 RL Infra 的关系**

框架实现层只简述：

```text
Teacher prefill service
async overlap
Teacher resource pool
trajectory metadata
policy version / staleness
```

明确：

- Teacher service 和 async overlap 是性能实现选择，不是 MOPD 算法定义。
- `On-policy` 不代表异步消费时绝对 zero staleness。
- NeMo RL 的 async GRPO、ICE-POP 属于框架层修正。
- trajectory 至少需要关联 domain/teacher、rollout policy version 和 token/action mask。

- [ ] **Step 12: 加入时间线、一手来源和 multi-round 边界**

时间线：

```text
2026-01-06  MiMo-V2-Flash Technical Report 首次公开 MOPD
2026-04-20  verl Multi-Teacher OPD PR #6051 合并
2026-06-29  MOPD 独立论文发布
```

来源：

- `https://arxiv.org/abs/2601.02780`
- `https://arxiv.org/abs/2606.30406`
- `https://github.com/verl-project/verl/pull/6051`
- `https://docs.nvidia.com/nemo/rl/nightly/about/algorithms/mopd.html`

补充：

- `multi-round evolution` 是 2026-06-29 独立论文分析的扩展：用上一轮 MOPD Student 重新训练领域 Teacher，再执行下一轮集成。
- 不得将 multi-round evolution、Top-k bias correction 或独立论文的稳定性分析写成 verl 2026-04-20 已完整复现的能力。

- [ ] **Step 13: 完成阶段总结和下一步**

总结控制在 300 字以内。下一阶段只列：

- verl/NeMo 代码级对照。
- AReaL MOPD 原型设计。
- Teacher prefill throughput 与 overlap 实验。
- routing correctness、policy freshness 和 control-token behavior 监控。

## Chunk 2: 克制接入导航

### Task 2: 建立双向关系和阅读入口

**Files:**
- Modify: `training-infra-roadmap/topics/agentic_rl.md`
- Modify: `training-infra-roadmap/MASTER_READING_LIST.md`
- Modify: `training-infra-roadmap/KNOWLEDGE_GRAPH.md`

- [ ] **Step 1: 在 Agentic RL 关系章节加入 MOPD**

增加一条双向链接，说明 MOPD 把 rollout inference、Teacher scoring 和 capability integration 纳入 post-training dataflow。

- [ ] **Step 2: 在 Master Reading List 增加递进路线**

加入：

```text
Traditional KD → OPD → MOPD → multi-teacher serving / routing
```

标记专题状态为“研究中 / 原理第一版”，不加入已完成旗舰章节。

- [ ] **Step 3: 在 Knowledge Graph 增加文字索引**

只在双向索引加入：

```text
Agentic RL ↔ OPD / MOPD ↔ Teacher Prefill / Domain Routing
```

不修改主图。

- [ ] **Step 4: 检查双向链接**

Run:

```bash
rg -n "MOPD|mopd" \
  training-infra-roadmap/topics/agentic_rl.md \
  training-infra-roadmap/topics/mopd.md \
  training-infra-roadmap/MASTER_READING_LIST.md \
  training-infra-roadmap/KNOWLEDGE_GRAPH.md
```

Expected: 四个文件均有可追踪入口，`mopd.md` 回链 Agentic RL。

## Chunk 3: 验证与交付

### Task 3: 运行内容与仓库健康检查

**Files:**
- Verify: `training-infra-roadmap/topics/mopd.md`
- Verify: `training-infra-roadmap/topics/agentic_rl.md`
- Verify: `training-infra-roadmap/MASTER_READING_LIST.md`
- Verify: `training-infra-roadmap/KNOWLEDGE_GRAPH.md`

- [ ] **Step 1: 检查事实边界**

逐项确认：

- MiMo report、verl merge、独立论文日期正确。
- 四个一手来源的标题、作者/发布主体和发布日期与正文一致。
- sampled-token advantage、reverse-KL 方向和 Top-k bias correction 的描述能在独立论文中找到依据。
- verl 的 per-sample routing、Teacher manager 和示例信息能在 PR #6051 找到依据。
- NeMo RL 的 dedicated Teacher group、async GRPO 和 ICE-POP 描述能在官方文档找到依据。
- Teacher trajectory 与 Teacher prefill scoring 没有混写。
- on-policy 与 zero-staleness 没有画等号。
- same-origin 的机制推断明确标注为推断。
- Top-k variant 没有倒推成 verl 完整复现。
- multi-round evolution 明确标注为独立论文扩展。

- [ ] **Step 2: 检查本地 Markdown 链接**

Run:

```bash
/Users/zengbw/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
from pathlib import Path
import re

root = Path("training-infra-roadmap")
broken = []
checked = 0
for md in root.rglob("*.md"):
    text = md.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = target.strip().split("#", 1)[0]
        if not target or "://" in target or target.startswith(("mailto:", "#")):
            continue
        checked += 1
        path = (md.parent / target).resolve()
        if not path.exists():
            broken.append((md, target))

print(f"checked={checked} broken={len(broken)}")
for source, target in broken:
    print(f"{source}: {target}")
raise SystemExit(1 if broken else 0)
PY
```

Expected: `0 broken`。

- [ ] **Step 3: 检查新增和既有工作区变化**

Run:

```bash
git diff --name-only
git ls-files --others --exclude-standard
git status --short
git diff --cached --name-only
```

Expected:

- 本专题只涉及 spec、plan、`topics/mopd.md` 和三个批准的导航文件。
- 原有 Q3 project 状态修改和 `.superpowers/` 保持原样。
- 暂存区在显式 `git add` 前为空；显式暂存后严格等于六个批准文件。

- [ ] **Step 4: 准备提交命令，不自动混入用户修改**

先比较本地与云端：

```bash
git fetch origin
git log --oneline --left-right origin/main...HEAD
git diff --name-only origin/main...HEAD
```

Expected: 明确本地是否领先/落后以及历史提交差异，不把工作区修改误当成远端差异。

提交范围只包含本专题文件：

```bash
git add \
  docs/superpowers/specs/2026-07-24-mopd-topic-design.md \
  docs/superpowers/plans/2026-07-24-mopd-topic.md \
  training-infra-roadmap/topics/mopd.md \
  training-infra-roadmap/topics/agentic_rl.md \
  training-infra-roadmap/MASTER_READING_LIST.md \
  training-infra-roadmap/KNOWLEDGE_GRAPH.md
```

暂存后、提交前运行：

```bash
git diff --cached --name-only
```

Expected: 输出严格等于上述六个文件。

确认暂存范围无误后再提交：

```bash
git commit -m "Add OPD to MOPD principle study topic"
```

在返回命令前先 fetch `origin` 并比较本地与云端 diff。
