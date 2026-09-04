# Interview Command Center Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在面试准备主文档前部增加项目/技术双入口速查控制台，并为 74 道唯一问题提供可验证的题尾回链。

**Architecture:** 控制台是唯一的现场快速路由层，所有链接复用现有答案锚点；能力图只表达能力关系，Part 索引继续承担完整目录。优先级边界明确为：技术视图只展示 P0；项目视图可保留直接项目证据型 P1；P2 只在 Part 全量索引。精确返回依赖浏览器历史，静态题尾链接分别返回所属 Part 和总控制台。

**Tech Stack:** GitHub Flavored Markdown、显式 HTML anchor、Python 只读/机械结构校验、Git。

---

## Chunk 1: 导航控制台、题尾回链与发布

### Task 1: 建立失败基线并保护工作区

**Files:**
- Reference: `docs/superpowers/specs/2026-09-04-interview-command-center-design.md`
- Inspect: `private_resume/2026-08-llm-infra-interview-prep.md`

- [ ] **Step 1: 核对干净实施工作树与远端基线**

Run:

```bash
git fetch origin
git status --short --branch
git rev-list --left-right --count origin/main...HEAD
git -C /Users/zengbw/ReadBase status --porcelain=v1 > /private/tmp/readbase-main-status.before
```

Expected: 实施工作树无未提交内容；`HEAD` 只包含已审查的规格提交，`origin/main` 没有未整合的新提交。主工作区 `/Users/zengbw/ReadBase` 的未提交修改不参与任何 switch、stash 或 commit。

- [ ] **Step 2: 运行导航结构的失败测试**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import unittest

text = Path("private_resume/2026-08-llm-infra-interview-prep.md").read_text(encoding="utf-8")

class NavigationBaseline(unittest.TestCase):
    def test_console_anchor(self):
        self.assertEqual(text.count('<a id="interview-console"></a>'), 1)

    def test_questions_to_ask_anchor(self):
        self.assertEqual(text.count('<a id="vi-questions-to-ask"></a>'), 1)

    def test_per_question_backlinks(self):
        self.assertEqual(text.count("返回面试速查控制台"), 74)

unittest.main()
PY
```

Expected: FAIL，显示 3 个失败，实际值分别为 `0`、`0`、`0`；三个新增行为都在修改前得到失败基线，而不是被第一个断言短路。

### Task 2: 添加双入口速查控制台

**Files:**
- Modify: `private_resume/2026-08-llm-infra-interview-prep.md:1-130`
- Modify: `private_resume/2026-08-llm-infra-interview-prep.md` 中 `VI.4 建议反问面试官` 前

- [ ] **Step 1: 在“五句法”和知识边界提示之后新增控制台锚点和使用提示**

Add:

```markdown
<a id="interview-console"></a>
### 0.1 面试现场速查控制台

> **怎么用**：从“项目经历”回答做过什么，从“技术主题”回答机制是什么，从“关键数字”反查项目。点击答案后，浏览器返回按钮、macOS `⌘ + [` 或 Windows/Linux `Alt + ←` 可回到刚才的准确位置；题尾的 Part / 总控制台回链用于可靠兜底。
```

- [ ] **Step 2: 添加现场救急入口**

Add exactly these links:

```markdown
**现场救急**：[自我介绍](#resume-01) · [Ownership](#resume-01b) · [职业选择](#resume-01c) · [最有代表性的优化](#resume-01a) · [为什么选 verl / AReaL](#areal-01) · [万卡特有问题](#infra-09) · [技术面反问](#vi-questions-to-ask)
```

- [ ] **Step 3: 添加“从项目经历进入”表格**

Use this exact mapping:

```markdown
#### 从项目经历进入

| 项目主线 | 高频问题与技术细节 |
|---|---|
| **X1 200B MoE** | [代表性优化](#resume-01a) · [5D 并行](#megatron-01) · [Dense/MoE](#moe-01) · [EP 与 all-to-all](#megatron-06) · [通信算子](#infra-04) · [规模交付](#resume-10) |
| **Long Context SFT** | [31s→9.3s](#resume-05) · [35B-A3B/128K](#resume-17) · [128K/256K 显存](#resume-06) · [7.6GB CP-local logits](#resume-07) · [SP 与 CP](#megatron-04) · [Recompute/Offload](#megatron-09) |
| **Fully Async RLVR** | [同步与异步](#resume-02) · [gen-TP 与实例数](#resume-03) · [HybridFlow](#verl-01) · [Colocate/Disaggregate](#verl-02) · [Streaming/Partial/Staleness](#verl-04) · [RLVR 正确性](#verl-05) |
| **Agentic RL** | [AReaL 训练链路](#resume-08) · [框架选型](#areal-01) · [Off-policyness](#areal-02) · [Gateway 改造](#areal-09) · [CUDA Graph](#resume-13) · [Gateway 调度收益](#resume-19) · [XCCL/Disk](#areal-11) |
| **OPD / MOPD** | [MOPD 主问题](#resume-09) · [PPO/GRPO/DAPO](#rl-algo-01) · [Trajectory→Gradient](#areal-04) · [三层正确性门禁](#areal-08) |
| **TX 文生视频 / 规模化交付** | [HunyuanVideo/Ulysses](#resume-18) · [融合算子](#kernel-01) · [千卡/万卡交付](#resume-10) · [精度对齐](#resume-12) · [万卡规模效应](#infra-09) |
```

- [ ] **Step 4: 添加“从技术主题进入”表格**

Use this exact mapping:

```markdown
#### 从技术主题进入

| 技术主题 | 高频问题与项目入口 |
|---|---|
| **Megatron / 多维并行** | [5D 并行](#megatron-01) · [Column/Row TP](#megatron-02) · [TP 负优化](#megatron-03) · [SP/CP](#megatron-04) · [Distributed Optimizer](#megatron-05) · [Megatron/FSDP 选型](#megatron-11) |
| **MoE** | [Dense 与 MoE](#moe-01) · [EP/EDP 与 A2A](#megatron-06) · [X1 200B MoE](#resume-01a) · [Grouped GEMM/融合](#kernel-01) |
| **显存 / 长上下文** | [Megatron 显存账本](#infra-02) · [128K/256K 显存](#resume-06) · [CP-local logits](#resume-07) · [35B-A3B/128K](#resume-17) |
| **RL 算法 / verl** | [PPO/GRPO/DAPO](#rl-algo-01) · [HybridFlow](#verl-01) · [Colocate/Disaggregate](#verl-02) · [权重同步](#verl-03) · [Fully Async](#verl-04) · [RLVR 正确性](#verl-05) · [vLLM/SGLang](#verl-09) |
| **AReaL / Agentic RL** | [训练链路](#resume-08) · [框架选型](#areal-01) · [Off-policyness](#areal-02) · [Trajectory Lineage](#areal-04) · [Gateway Ownership](#areal-09) · [XCCL/Disk](#areal-11) · [Gateway 调度](#resume-19) |
| **推理 / Rollout** | [gen-TP](#resume-03) · [vLLM/SGLang](#verl-09) · [CUDA Graph](#resume-13) |
| **通信 / 集群 / 恢复** | [Collective](#infra-04) · [万卡规模效应](#infra-09) · [NCCL/恢复排障](#infra-03) |
| **正确性 / 交付** | [RLVR 正确性](#verl-05) · [Trajectory→Gradient](#areal-04) · [规模交付](#resume-10) |
```

- [ ] **Step 5: 添加关键数字反查表**

Use:

```markdown
#### 按关键数字反查

| 简历数字 | 对应问题 |
|---|---|
| [`0.16x → 0.95x / MFU 35%`](#resume-01a) | X1 200B MoE 性能优化 |
| [`31s → 9.3s / 23% → 45.2%`](#resume-05) | Qwen3.5-9B SFT |
| [`128K / 平均 step time 降低约 50%`](#resume-17) | Qwen3.5-35B-A3B |
| [`7.6GB`](#resume-07) | CP-local logits |
| [`76 → 211–255`](#resume-02) | Fully Async RLVR |
| [`6–8x`](#resume-13) | Agentic RL decode / CUDA Graph |
| [`+60% / 33.18% → 2.73%`](#resume-19) | Gateway Rollout 调度 |
| [`3K 卡 / 两个月`](#resume-10) | X1 规模交付 |
```

在数字表之后增加 `---`，明确结束控制台，再进入能力图。

- [ ] **Step 6: 收敛能力图和反问锚点**

Change `### 0.1 一张图看懂我的能力主线` to `### 0.2 一张图看懂我的能力主线`。控制台结尾的 `---` 必须紧邻该标题之前。删除能力图下现有的“核心机制导航”和“项目证据导航”两行，保留图、图例和 20–30 秒口述版。在 `### VI.4 建议反问面试官` 前添加：

```html
<a id="vi-questions-to-ask"></a>
```

- [ ] **Step 7: 运行控制台局部校验**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re, sys

text = Path("private_resume/2026-08-llm-infra-interview-prep.md").read_text(encoding="utf-8")
anchors = re.findall(r'<a id="([^"]+)"></a>', text)
assert anchors.count("interview-console") == 1
assert anchors.count("vi-questions-to-ask") == 1
assert "### 0.2 一张图看懂我的能力主线" in text
assert "- **核心机制导航**：" not in text
assert "- **项目证据导航**：" not in text
start = text.index('<a id="interview-console"></a>')
end = text.index("### 0.2 一张图看懂我的能力主线")
console = text[start:end]
assert console.rstrip().endswith("---")
assert '<a id="vi-questions-to-ask"></a>\n### VI.4 建议反问面试官' in text

def anchors_in(line):
    return re.findall(r"\]\(#([a-z0-9-]+)\)", line)

def unique_line(block, prefix):
    matches = [line for line in block.splitlines() if line.startswith(prefix)]
    assert len(matches) == 1, (prefix, matches)
    return matches[0]

rescue = ["resume-01", "resume-01b", "resume-01c", "resume-01a", "areal-01", "infra-09", "vi-questions-to-ask"]
project_start = console.index("#### 从项目经历进入")
tech_start = console.index("#### 从技术主题进入")
metric_start = console.index("#### 按关键数字反查")
intro_block = console[:project_start]
project_block = console[project_start:tech_start]
tech_block = console[tech_start:metric_start]
metric_block = console[metric_start:]
assert anchors_in(unique_line(intro_block, "**现场救急**：")) == rescue

project_rows = {
    "X1 200B MoE": ["resume-01a", "megatron-01", "moe-01", "megatron-06", "infra-04", "resume-10"],
    "Long Context SFT": ["resume-05", "resume-17", "resume-06", "resume-07", "megatron-04", "megatron-09"],
    "Fully Async RLVR": ["resume-02", "resume-03", "verl-01", "verl-02", "verl-04", "verl-05"],
    "Agentic RL": ["resume-08", "areal-01", "areal-02", "areal-09", "resume-13", "resume-19", "areal-11"],
    "OPD / MOPD": ["resume-09", "rl-algo-01", "areal-04", "areal-08"],
    "TX 文生视频 / 规模化交付": ["resume-18", "kernel-01", "resume-10", "resume-12", "infra-09"],
}
tech_rows = {
    "Megatron / 多维并行": ["megatron-01", "megatron-02", "megatron-03", "megatron-04", "megatron-05", "megatron-11"],
    "MoE": ["moe-01", "megatron-06", "resume-01a", "kernel-01"],
    "显存 / 长上下文": ["infra-02", "resume-06", "resume-07", "resume-17"],
    "RL 算法 / verl": ["rl-algo-01", "verl-01", "verl-02", "verl-03", "verl-04", "verl-05", "verl-09"],
    "AReaL / Agentic RL": ["resume-08", "areal-01", "areal-02", "areal-04", "areal-09", "areal-11", "resume-19"],
    "推理 / Rollout": ["resume-03", "verl-09", "resume-13"],
    "通信 / 集群 / 恢复": ["infra-04", "infra-09", "infra-03"],
    "正确性 / 交付": ["verl-05", "areal-04", "resume-10"],
}
def validate_named_table(block, expected_rows):
    row_re = re.compile(r"^\| \*\*([^*]+)\*\* \|")
    labels = [row_re.match(line).group(1) for line in block.splitlines() if row_re.match(line)]
    assert labels == list(expected_rows), (labels, list(expected_rows))
    for label, expected_anchors in expected_rows.items():
        actual = anchors_in(unique_line(block, f"| **{label}** |"))
        assert actual == expected_anchors, (label, actual, expected_anchors)

validate_named_table(project_block, project_rows)
validate_named_table(tech_block, tech_rows)

metric_rows = {
    "| [`0.16x → 0.95x / MFU 35%`](#resume-01a) |": "resume-01a",
    "| [`31s → 9.3s / 23% → 45.2%`](#resume-05) |": "resume-05",
    "| [`128K / 平均 step time 降低约 50%`](#resume-17) |": "resume-17",
    "| [`7.6GB`](#resume-07) |": "resume-07",
    "| [`76 → 211–255`](#resume-02) |": "resume-02",
    "| [`6–8x`](#resume-13) |": "resume-13",
    "| [`+60% / 33.18% → 2.73%`](#resume-19) |": "resume-19",
    "| [`3K 卡 / 两个月`](#resume-10) |": "resume-10",
}
metric_lines = [line for line in metric_block.splitlines() if line.startswith("| [")]
assert len(metric_lines) == len(metric_rows), (len(metric_lines), len(metric_rows))
for actual_line, (prefix, target) in zip(metric_lines, metric_rows.items()):
    assert actual_line.startswith(prefix), (actual_line, prefix)
    assert anchors_in(actual_line) == [target]

print("PASS exact console rows", len(project_rows), len(tech_rows), len(metric_rows))
PY
```

Expected: `PASS exact console rows 6 8 8`。校验只读取控制台切片，逐行比较有序锚点，不会因链接被放错行、重复或误命中现有 Part 导航而通过。

- [ ] **Step 8: 提交控制台修改**

```bash
git add private_resume/2026-08-llm-infra-interview-prep.md
git diff --cached --check
git commit -m "Add interview command center navigation"
```

### Task 3: 为 74 道题添加题尾回链

**Files:**
- Modify: `private_resume/2026-08-llm-infra-interview-prep.md:129-1585`

- [ ] **Step 1: 用可运行的结构化机械修改插入回链**

对每个匹配 `^#### [A-Z][A-Z0-9-]*｜.*（P[012]，` 的问题，记录其所在的最近一个 `part-i` 至 `part-v`。问题结束边界是下一个显式 `<a id="..."></a>` 或下一个一级至四级 Markdown heading；忽略五级及以下的答案内部小标题。在边界之前去掉尾部空行，追加：

```markdown

↩ [返回本 Part 导航](#part-x) · ↑ [返回面试速查控制台](#interview-console)

```

其中 `part-x` 必须来自该题所在 Part。使用以下一次性机械转换；不以 `危险回答` 行作为边界，因为 `KERNEL-01` 当前包含两个该标签：

```bash
python3 - <<'PY'
from pathlib import Path
import re

path = Path("private_resume/2026-08-llm-infra-interview-prep.md")
lines = path.read_text(encoding="utf-8").splitlines()
question_re = re.compile(r"^#### [A-Z][A-Z0-9-]*｜.*（P[012]，")
part_re = re.compile(r'^<a id="(part-[iv]+)"></a>$')
anchor_re = re.compile(r'^<a id="[^"]+"></a>$')
heading_re = re.compile(r"^#{1,4} ")

assert sum(bool(question_re.match(line)) for line in lines) == 74
assert sum("返回面试速查控制台" in line for line in lines) == 0

out = []
current_part = None
active_part = None
inserted = 0

def close_question():
    global inserted, active_part
    while out and not out[-1].strip():
        out.pop()
    expected = f"↩ [返回本 Part 导航](#{active_part}) · ↑ [返回面试速查控制台](#interview-console)"
    out.extend(["", expected, ""])
    inserted += 1
    active_part = None

for line in lines:
    is_boundary = bool(anchor_re.match(line) or heading_re.match(line))
    if active_part is not None and is_boundary:
        close_question()

    pm = part_re.match(line)
    if pm:
        current_part = pm.group(1)

    out.append(line)

    if question_re.match(line):
        assert active_part is None
        assert current_part in {"part-i", "part-ii", "part-iii", "part-iv", "part-v"}
        active_part = current_part

if active_part is not None:
    close_question()

assert inserted == 74, inserted
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print("PASS inserted", inserted)
PY
```

Expected: `PASS inserted 74`。修改前置断言同时阻止二次执行产生重复回链。

- [ ] **Step 2: 运行逐题回链校验**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re

lines = Path("private_resume/2026-08-llm-infra-interview-prep.md").read_text(encoding="utf-8").splitlines()
question_re = re.compile(r"^#### [A-Z][A-Z0-9-]*｜.*（P[012]，")
part_re = re.compile(r'^<a id="(part-[iv]+)"></a>$')
anchor_re = re.compile(r'^<a id="[^"]+"></a>$')
heading_re = re.compile(r"^#{1,4} ")

questions = []
current_part = None
i = 0
while i < len(lines):
    pm = part_re.match(lines[i])
    if pm:
        current_part = pm.group(1)
    if not question_re.match(lines[i]):
        i += 1
        continue
    start = i
    part = current_part
    i += 1
    while i < len(lines) and not anchor_re.match(lines[i]) and not heading_re.match(lines[i]):
        i += 1
    body = [line for line in lines[start:i] if line.strip()]
    questions.append((lines[start], part, body))

assert len(questions) == 74, len(questions)
part_counts = {}
for title, part, body in questions:
    tail = body[-1]
    expected = f"↩ [返回本 Part 导航](#{part}) · ↑ [返回面试速查控制台](#interview-console)"
    assert tail == expected, (title, tail, expected)
    part_counts[part] = part_counts.get(part, 0) + 1
assert part_counts == {"part-i": 7, "part-ii": 25, "part-iii": 15, "part-iv": 18, "part-v": 9}, part_counts
print("PASS 74 per-question backlinks", part_counts)
PY
```

Expected: `PASS 74 per-question backlinks {'part-i': 7, 'part-ii': 25, 'part-iii': 15, 'part-iv': 18, 'part-v': 9}`。

- [ ] **Step 3: 提交回链修改**

```bash
git add private_resume/2026-08-llm-infra-interview-prep.md
git diff --cached --check
git commit -m "Add per-question interview navigation backlinks"
```

### Task 4: 全量验证并只发布到 main

**Files:**
- Verify: `private_resume/2026-08-llm-infra-interview-prep.md`
- Verify: all Markdown/image targets referenced by the main document

- [ ] **Step 1: 验证题量、优先级和显式锚点唯一性**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
from collections import Counter
import re

text = Path("private_resume/2026-08-llm-infra-interview-prep.md").read_text(encoding="utf-8")
counts = {p: len(re.findall(rf"^#### .*（{p}，", text, re.M)) for p in ("P0", "P1", "P2")}
assert counts == {"P0": 43, "P1": 26, "P2": 5}, counts
anchors = re.findall(r'<a id="([^"]+)"></a>', text)
duplicates = [a for a, n in Counter(anchors).items() if n > 1]
assert not duplicates, duplicates
print("PASS counts", counts, "anchors", len(anchors))
PY
```

- [ ] **Step 2: 验证全部本地链接、图片和同文件锚点**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re

src = Path("private_resume/2026-08-llm-infra-interview-prep.md").resolve()
body = src.read_text(encoding="utf-8")
link_re = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')
anchor_re = re.compile(r'<a\s+id=["\']([^"\']+)["\']\s*></a>', re.I)
own_anchors = set(anchor_re.findall(body))
missing = []

for raw in link_re.findall(body):
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if target.startswith(("http://", "https://", "mailto:")):
        continue
    if target.startswith("#"):
        if target[1:] not in own_anchors:
            missing.append(target)
        continue
    file_part, sep, anchor = target.partition("#")
    dst = (src.parent / file_part).resolve()
    if not dst.exists():
        missing.append(target)
        continue
    if sep and dst.suffix.lower() == ".md":
        dst_anchors = set(anchor_re.findall(dst.read_text(encoding="utf-8")))
        if anchor not in dst_anchors:
            missing.append(target)

assert not missing, missing
print("PASS all local files, images, and explicit anchors")
PY
```

Expected: `PASS all local files, images, and explicit anchors`。

- [ ] **Step 3: 审查差异边界**

Run:

```bash
git diff 53b617c..HEAD -- private_resume/2026-08-llm-infra-interview-prep.md
git diff --check 53b617c..HEAD
git status --short --branch
python3 - <<'PY'
from pathlib import Path
import re, subprocess

target = "private_resume/2026-08-llm-infra-interview-prep.md"
base = subprocess.check_output(["git", "show", f"53b617c:{target}"], text=True)
current = Path(target).read_text(encoding="utf-8")

start = current.index('<a id="interview-console"></a>')
end = current.index("### 0.2 一张图看懂我的能力主线")
current = current[:start] + current[end:]
current = current.replace("### 0.2 一张图看懂我的能力主线", "### 0.1 一张图看懂我的能力主线", 1)
current = current.replace('<a id="vi-questions-to-ask"></a>\n', "", 1)
current = re.sub(r'^↩ \[返回本 Part 导航\]\(#part-[iv]+\) · ↑ \[返回面试速查控制台\]\(#interview-console\)\n?', "", current, flags=re.M)

for old_line in (
    "- **核心机制导航**：",
    "- **项目证据导航**：",
):
    base = "\n".join(line for line in base.splitlines() if not line.startswith(old_line)) + "\n"

normalize = lambda value: re.sub(r"\n{3,}", "\n\n", value).strip()
assert normalize(current) == normalize(base), "non-navigation body changed"

allowed = {
    "docs/superpowers/specs/2026-09-04-interview-command-center-design.md",
    "docs/superpowers/plans/2026-09-04-interview-command-center.md",
    target,
}
changed = set(subprocess.check_output(["git", "diff", "--name-only", "53b617c..HEAD"], text=True).splitlines())
assert changed == allowed, (changed, allowed)
print("PASS navigation-only diff", sorted(changed))
PY
```

Expected: `PASS navigation-only diff ...`；主文档只新增控制台、反问锚点、题尾回链，删除两行旧导航并将能力图改为 `0.2`，机械校验证明其余问答正文和数字未变化；实施工作树干净。

- [ ] **Step 4: 获取远端最新状态并确认可快进**

```bash
git fetch origin
git rev-list --left-right --count origin/main...HEAD
git merge-base --is-ancestor origin/main HEAD
```

Expected: 左侧为 0，说明没有覆盖远端新提交；否则停止并先处理远端并发更新。

- [ ] **Step 5: 仅推送到远端 main 并核对**

```bash
git push origin HEAD:main
git ls-remote --heads origin
local_sha=$(git rev-parse HEAD)
remote_sha=$(git ls-remote --heads origin main | awk '{print $1}')
test "$local_sha" = "$remote_sha"
```

Expected: 远端只列出 `refs/heads/main`，其 SHA 与本地 `HEAD` 完全一致。不创建或推送任何 `codex/*` 远端分支。

- [ ] **Step 6: 保留脏主工作区**

Run:

```bash
git -C /Users/zengbw/ReadBase status --short --branch
git -C /Users/zengbw/ReadBase status --porcelain=v1 > /private/tmp/readbase-main-status.after
diff -u /private/tmp/readbase-main-status.before /private/tmp/readbase-main-status.after
```

Expected: `diff` 无输出；原有 3 个未提交文件仍然存在且未被本次提交纳入。不对该工作区执行 switch、stash、reset、merge 或 commit。
