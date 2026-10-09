# AGENTS.md

This file is the shared working guide for AI coding agents in this repository. It applies to Codex, Claude Code, and other agents unless a tool-specific file says otherwise.

## Repository Purpose

ReadBase is a Chinese-language, content-first Personal Research Operating System for Large-Scale AI Systems. It is not an application repo and has no product build pipeline. Its peer learning tracks cover Systems, Training Infra, Inference Infra, RL Infra, and Embodied Models & Infra, with shared research, practice, and interview layers.

Write for a software engineer building production-grade AI systems judgment. Prefer model/learning mechanisms that answer a concrete learning gap, system design, production troubleshooting, and interview readiness over academic-style summaries. Training depth remains valuable; inference is an independent track, not merely a rollout dependency.

North Star: build a production-grade understanding of Large-Scale AI Systems through a human-AI maintained Personal Research Operating System.

## Working Style

- Use Chinese for authored handbook content, keeping common technical terms in English: TP, ZeRO, FSDP, FlashAttention, GEMM, NCCL, all-reduce, checkpoint, etc.
- Do not turn papers into translation notes. Explain what problem the work solved, which model/learning assumption or system bottleneck changed, what breaks in production, and how modern systems inherited the idea.
- Keep changes scoped. Do not rewrite unrelated files or reorder large sections unless the task explicitly asks for it.
- Treat existing uncommitted files as user or other-agent work. Do not revert them without explicit permission.

## Structure

- All paths in these rules are repository-root-relative unless explicitly marked as a relative-link example.
- `README.md` is the umbrella entry point and prioritizes the long-term learning map. Keep one recent-updates list with at most five entries, linking directly to substantive content or project updates; refresh it after substantial content changes. Do not include private personal or salary data.
- `KNOWLEDGE_GRAPH.md` is the single cross-track knowledge graph.
- `systems/`, `training-infra/`, `inference-infra/`, `rl-infra/`, and `embodied-infra/` are peer tracks. Each has a `README.md` entry and `topics/` chapters, not its own full research radar, reading queue, or knowledge graph.
- `research/papers/`, `research/tech_reports/`, and `research/engineering_blogs/` contain shared paper, model/system report, and engineering-source notes. Official docs, release notes, and vendor technical posts are first-class sources for implementation details.
- `research/tracking/` is the shared research radar: frontier scans, scan logs, release notes, trends, monthly digests, and historical backfill. It records signal and triage, not full notes.
- `research/reading_queue/` turns signals into shared P0/P1/Done reading decisions. The P0 target is at most three active items during future triage; structural migrations must not delete, demote, or mark existing items Done merely to satisfy that target.
- `research/learning_log/` records monthly learning progress, questions, and next steps; `research/insights/` stores original judgments; `research/references/` contains CSV indexes.
- `research/MASTER_READING_LIST.md` is the shared source index; `research/philosophy.md` records reading principles.
- `practice/experiments/`, `practice/projects/`, and `practice/playbooks/` contain verification, project evidence, and production runbooks respectively.
- `interview/README.md` is an interview hub only; `interview/topics/` contains shared interview handbook notes. Preserve `private_resume/` and its existing main/Coding/materials paths; do not move private material into the public hub.
- `reading_inbox/` remains independent from the curated research workflow.
- `training-infra/roadmaps/` preserves historical training-focused learning plans; it is not the root learning map.
- Core learning routes live inside their owning Part: `systems/roadmaps/` for GPU and distributed systems, and `inference-infra/roadmap.md`, `rl-infra/roadmap.md`, `embodied-infra/roadmap.md` for application tracks. Routes define learning order, coverage gaps, and mastery checks; reuse canonical topic bodies and shared experiments instead of duplicating them. A route is not a second research queue or evidence that an experiment has run.
- `assets/handbook/` holds the migrated handbook figures; existing root-level `assets/` resources keep their paths.

## Document Templates

For `research/papers/`, follow this section order:

论文信息 → 解决的问题 → 背景与瓶颈 → 核心创新 → 关键图表解读 → 工程价值 → 对训练基础设施的影响 → 今天的应用场景 → 后续演进 → 相关论文 → 相关代码 → 面试高频问题 → 生产环境思考题 → 我的总结.

For `research/tech_reports/`, follow this section order:

论文信息 → 架构概览 → 训练系统设计 → 并行策略 → 显存优化 → 通信优化 → 集群规模 → 工程经验 → 对行业的影响 → 我的收获 → 后续演进 → 面试高频问题 → 生产环境思考题.

Keep these paper/report section orders. In cross-track notes, use the relevant model, inference, RL, or embodied perspective inside each section; explicitly mark undisclosed or inapplicable training details rather than inventing them.

For `research/engineering_blogs/`, do not summarize marketing copy. Extract the engineering signal: source information → solved problem → engineering background → core mechanism → system design details → performance/stability information → production lessons → related topics → questions to pursue → short summary.

For `research/tracking/`, keep entries lightweight and judgment-heavy. Each item should include source/type/link, impact level, `Decision` (`Ignore` / `Observe` / `Read` / `Deep Dive`), `Reason`, related topics, one-sentence value, and next step. Every newly accepted signal also needs `Learning track` (one or more of Systems / Training / Inference / RL / Embodied), `Signal type` (model / algorithm / system), `Evidence`, and `Target question`. Tracking is an inbox/radar; promote only important items into the shared research queue/notes, the relevant track's `topics/`, or `practice/` artifacts.

For `research/tracking/frontier_scan_YYYY-MM-DD.md`, scan from the previous cursor in `research/tracking/scan_log.md` to the actual scan end timestamp. Do not force a natural week. Do not write an end-of-day timestamp such as `23:59` unless that time has actually been scanned. If the exact scan end timestamp was not recorded, the next scan should backtrack to the last confirmed timestamp and dedupe. A frontier scan can have zero accepted signals. Every accepted signal needs `Source ID`, `First seen`, scan window, `Decision`, and `Reason`. After an actual scan, update only `research/tracking/scan_log.md` with the next cursor; never write a legacy-path stub. The same Source ID may appear in multiple Watch sections but counts as one accepted signal.

For every accepted signal and every new paper/report note, verify title, authors, publication date, and key numeric claims against the primary source page before writing them as facts. For arXiv sources, the arXiv ID resolving is not enough: the `citation_title`, `citation_author`, `citation_date`, and abstract/method details must match the note. If a detail is inferred rather than directly sourced, label it as an inference.

For `research/tracking/historical_backfill.md`, do not chase recency. It is an index and rules page. Backfill entries should live in `research/tracking/backfill/YYYY-MM.md`, where `YYYY-MM` is the material's original publication month. Backfill only past materials that fill a current model/learning or engineering judgment gap. Each entry should explain original time, backfill time, why it is backfilled now, historical impact, current value, Decision, Reason, suggested action, related topics, target destination, and lifecycle status. Do not mix historical backfill into frontier scans.

Weekly signal reports and weekly papers templates are retired. Keep existing weekly files only as historical audit records. For current updates, use frontier scans plus `research/tracking/scan_log.md`; for formal summaries, use monthly signal reports.

Research-scan response preference: every frontier scan or monthly signal report delivery must include the full report link, a short Chinese paragraph summarizing overall progress and engineering trends, and one sentence per accepted article/signal explaining its main takeaway with a source link. Describe the bottleneck, mechanism, or practical consequence rather than repeating the title. Keep newly published items distinct from late-discovered materials, label trend inferences, and mention material coverage gaps briefly. Do not return only a report link or a publication/commit status. If no signals qualify, say so without inventing a trend.

Monthly reports use the previous calendar month, named `monthly_signal_YYYY-MM.md`. Monthly reports are the high-quality digest and should summarize frontier scans, backfill, release notes, and actual reading results; they should not rediscover material from scratch.

The expanded focus policy takes effect on 2026-09-22 for future work. This policy change is not a scan and does not advance any cursor, change historical Accepted counts, or assert historical coverage of new tracks. Label newly recognized historical coverage gaps; backfill older materials by original publication month. Monthly reports summarize existing records and reading results rather than discovering new sources to fill those gaps.

Frontier/monthly triage must use the repository owner's focus filter, not generic AI popularity. Preserve AI Systems, Training Infra, distributed training, GPU clusters/networking, Megatron/DeepSpeed/FSDP, MoE, FlashAttention/kernel/precision, NVIDIA training stack, large-scale reports, and Agentic RL/post-training infra. Also cover independent inference systems (serving, scheduling, KV/state, latency/throughput/cost, correctness, deployment) and embodied model/learning mechanisms plus infrastructure.

An accepted candidate must either (1) answer a current embodied learning gap about inputs/outputs, action representations, objectives, generalization, or evaluation, even without demonstrated infra gains; or (2) establish a relevant system consequence in performance, correctness, state, IO, or deployment. State the target question and evidence. Do not automatically promote demos, fundraising, generic releases, prompt tricks, or unverified leaderboards. Lack of code does not erase a disclosed model mechanism, but prevents claiming runnable implementation or reproduction; unverified performance and generalization claims stay explicitly vendor-reported/unverified. Distinguish MLLM (multimodal large language model) from vLLM (an inference engine).

Hardware/systems sources, including ml-engineering material, are selected by the question they help answer. Do not mirror an external book wholesale or execute unreviewed scripts merely because a source suggests them.

Vendor watch rule: every frontier scan and monthly signal report must include an explicit `OpenAI / Anthropic / NVIDIA / DeepSeek Watch` section. These four vendors are not automatic accepts, but their papers, technical reports, official docs, engineering blogs, model cards, weight releases, release notes, and research posts must be visibly triaged as `Accepted`, `Observed`, `Rejected`, or `Not found / not verifiable in this scan`. Treat technical reports and scale-backed production reports from core model vendors as first-class industrial evidence: read them with high priority, but distinguish disclosed mechanisms and reproducible evidence from vendor-reported numbers and repository inference. For DeepSeek, check both the official API changelog and official Hugging Face organization because important open-weight releases may appear there without a standalone blog. If a vendor source cannot be scanned or verified, state that limitation instead of silently omitting the vendor.

Hugging Face watch rule: every frontier scan and monthly signal report must also include an explicit `Hugging Face Watch`. For frontier scans, check the Hugging Face Blog plus relevant TRL, Transformers, Accelerate, PEFT, Kernels, and LeRobot releases/docs, especially for Agentic RL, rollout correctness, training-serving integration, long context, distributed training, independent inference backends, embodied learning, and dataset IO. Monthly reports summarize already-triaged records only. Distinguish official-team or vendor-authored posts from community posts, and do not auto-accept either category.

RL framework watch rule: every new frontier scan and monthly signal report must include an explicit `RL Framework Watch`; do not retrofit historical audit records solely to add the section. Monthly signal reports should summarize only the framework changes accepted or carried forward by that month's frontier scans. The one-time 2026-07-23 historical audit is an explicitly requested migration: its additions must remain labeled `Historical Audit` and must not alter original Accepted counts, reading decisions, or cursors. The core watchlist is AReaL, verl, slime, ROLL, OpenRLHF, and NeMo RL. Add emerging frameworks dynamically when they provide real code, a runnable training path, or reproducible benchmark evidence. Track official releases and major PRs that change architecture, performance, correctness, or operational behavior; ignore routine commits, documentation-only changes, minor bug fixes, and promotional repositories without implementation evidence. For every material change, identify which subsystem changed (`rollout`, `training`, `scheduler`, `weight sync`, `data/trajectory path`, `checkpoint/recovery`, or `inference backend`), the affected engineering dimension, the supporting evidence, and whether the design is transferable to AReaL. A major PR is not automatically an Accepted signal.

Inference watch rule: every new frontier scan and monthly report must include `Inference Systems Watch`, independently of RL rollout. Check relevant official release/docs, major PRs, tests, benchmarks, and production reports for runtimes such as vLLM, SGLang, and TensorRT-LLM; evaluate serving/scheduling, KV/state movement, attention/kernel/precision, distributed inference, and deployment correctness. Record sources actually checked, evidence, decision, and gaps. Monthly reports reuse the month's existing triage only.

Embodied watch rule: every new frontier scan and monthly report must include `Embodied Models & Infra Watch`. Candidate sources include openpi / Physical Intelligence, OpenVLA, NVIDIA GR00T / Isaac Lab, Hugging Face LeRobot, PyTorch / TorchCodec, and dataset/IO communities. This is a candidate watchlist, not a recommendation or automatic acceptance list. Apply the two admission routes above; state model/learning value separately from system implications and distinguish disclosed evidence from inference. Record actual source coverage and `Accepted`, `Observed`, `Rejected`, or `Not found / not verifiable in this scan`; zero accepted is valid. Monthly reports reuse existing records, not a new scan. Do not retrofit old audit records merely to add these Watch sections.

Automation migration checks must name their scope. The 2026-09-22 local audit inspected four local configs and found no ReadBase hardcoded-path matches; this is not evidence that all machines or remote schedulers were audited. Future automation work must resolve the actual configuration and use `research/tracking/scan_log.md` as the sole cursor write target.

For `practice/playbooks/`, write runbooks, not concept explanations: symptom → impact scope → first response → investigation order → commands → log keywords → likely root causes → fixes → validation → prevention → related topics/sources/experiments → postmortem questions.

For each learning track's `topics/`, write as engineering handbook chapters: problem framing → mechanism → config guidance → production pitfalls → troubleshooting → adjacent-system relationships. Embodied model chapters should additionally make inputs/outputs, action representations, objectives, generalization, and evaluation boundaries explicit when relevant.

For `interview/topics/`, include: 高频面试题 → 追问问题 → 生产环境案例 → 常见错误回答 → 优秀回答示例.

## Knowledge Lifecycle

Use these statuses when tracking important materials:

```text
NEW         just discovered
READING     actively reading
SUMMARIZED  converted into a paper/report/blog note
DIGESTED    reflected in topics or insights
VERIFIED    validated by experiment or reproduction
IMPLEMENTED used in real engineering practice or production design
OBSOLETE    outdated or superseded
```

## Human-AI Workflow Protocol

When adding a new material:

1. Add it to `research/tracking/` with impact, Decision, Reason, Learning track, Signal type, Evidence, and Target question. Use frontier scan for new material and `research/tracking/backfill/YYYY-MM.md` for older material.
2. If important, move it to `research/reading_queue/P0.md` or `research/reading_queue/P1.md` through explicit triage.
3. After reading, create or update the relevant paper/report/blog note.
4. Update at least one relevant track's `topics/` chapter if the material changes model/learning or system understanding.
5. If it forms a technical judgment, add or update `research/insights/`.
6. If it can be validated, create or update a `practice/experiments/` record; preserve project evidence in `practice/projects/`.
7. If it changes production troubleshooting, add or update a `practice/playbooks/` runbook.
8. Update root `KNOWLEDGE_GRAPH.md` / `research/MASTER_READING_LIST.md` when navigation changes, and the root recent-updates list after substantial content changes.

Important: if a paper does not change model/learning understanding, engineering judgment, experiment design, or system implementation, it is not really finished. Collection or migration alone is not reading progress.

## Linking Rules

- Use relative Markdown links between files.
- Keep links bidirectional when adding important relationships.
- Update root `KNOWLEDGE_GRAPH.md` and `research/MASTER_READING_LIST.md` when adding a relationship that changes navigation.
- Relative-link examples: from `research/papers/transformer.md` to a training chapter use `../../training-infra/topics/tensor_parallelism.md`; from that chapter back to the source use `../../research/papers/transformer.md`; from `research/tracking/README.md` to the shared queue use `../reading_queue/README.md`. Do not assume all topics remain one directory above a source note.
- Before claiming completion, verify internal Markdown links and image paths.

## Diagram Rules

- Use Mermaid for lightweight navigation diagrams in README files and `KNOWLEDGE_GRAPH.md`.
- For core paper/topic explanations, prefer research-paper-style SVG figures when the diagram carries conceptual weight: model blocks, data flow, communication patterns, checkpoint layouts, parallel groups, and kernel IO paths.
- SVG figures should be light-toned, readable on GitHub, and visually calm.
- Avoid overlapping elements, especially overlapping text.
- Put secondary annotations in whitespace or side callouts instead of crowding the main flow.
- Keep arrow crossings rare and meaningful.
- Use consistent colors for semantic categories such as GEMM/Linear, Attention/kernel, state boundary, residual, communication, and checkpoint.
- When embedding original paper figures in this public repo, use them sparingly, cite the source clearly, and prefer key figures that anchor the reader's understanding.

## Verification

Typical checks for this repo:

- Markdown links and local image paths resolve.
- SVG files parse as XML.
- JSON files parse.
- CSV files parse with the expected number of columns.
- Git status clearly separates tracked edits from new assets.

Do not say a change is complete until the relevant checks have actually run.

## Git And Publishing

Before returning commit commands, check local status and fetch the remote reference so the diff against GitHub is clear.

Repository-owner publishing rule:

- Treat GitHub `origin/main` as the canonical copy for user-facing handbook and interview-preparation documents.
- After a substantial in-scope change passes verification, commit and push it to `main` in the same task unless the user explicitly asks to keep it local.
- Do not leave material changes only in a temporary worktree or local feature branch. After successful integration, align local `main` with `origin/main` and remove clean, fully merged temporary worktrees and branches.
- Keep the remote branch surface to `main`; do not create or push a feature branch unless the user explicitly requests it or repository protection makes a pull request unavoidable.

Prefer commit messages that describe the learning artifact, for example:

```bash
git commit -m "Improve Transformer figures with research-style system view"
```

If GitHub rejects a direct push because branch rules require a pull request, create or suggest a branch-and-PR flow instead of weakening repository protection rules.
