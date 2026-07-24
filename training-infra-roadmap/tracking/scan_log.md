# Scan Log

`scan_log.md` 是前沿扫描账本，用来保证每次扫描都从上一次游标继续，做到不重不漏。

它不记录完整判断。完整结果写入对应的 `frontier_scan_YYYY-MM-DD.md` 文件；本文件只记录扫描窗口、来源范围、accepted / observed 数量和下一次扫描游标。

## 使用规则

- 每次执行“看看最新有没有更新”时，先查看本文件最后一行的 `Next cursor`。
- 新扫描窗口从上一次 `Next cursor` 开始，到本次实际扫描结束时刻为止。
- `Window` 结束时间和 `Next cursor` 只能写已经实际扫描过的时间点，不能预填当天 `23:59`。如果扫描发生在 7 月 7 日白天，结束时间就写白天的实际时刻。
- 如果某次扫描没有记录精确结束时刻，下一次扫描应回退到最后一个可确认时间点并做去重。宁可重复进入 observed / follow-up，也不要留下未覆盖时间段。
- 时区统一使用 `Asia/Shanghai`。
- 如果某次扫描因为网络、API limit 或来源不可访问而不完整，必须在 `Notes` 中写清楚。
- 不要因为某周没有扫描就补造 weekly；直接用更长的 frontier scan 窗口覆盖空档。
- 同一个 Source ID 不应在多个 frontier scan 中重复进入 accepted signal。重复出现时写入 follow-up 或 observed。

## Log

| Scan | Window | Sources | Accepted | Observed | Next cursor | Notes |
|---|---|---|---:|---:|---|---|
| [2026-07-04](frontier_scan_2026-07-04.md) | 2026-07-01 00:00 ~ 2026-07-04 23:59 | arXiv cs.LG / cs.AI / cs.CL / cs.DC | 5 | 9 | 2026-07-04 23:59 | 2026-07-08 回补重扫，修正旧 07-04 scan 覆盖不足 |
| [2026-07-07](frontier_scan_2026-07-07.md) | 2026-07-04 00:00 ~ 2026-07-07 scan-time unknown | arXiv recent / prior GitHub / blogs / releases | 3 | 10 | 2026-07-07 00:00 | 中间扫描记录；已由 07-04 rescan 和 07-08 scan 修正窗口口径 |
| [2026-07-08](frontier_scan_2026-07-08.md) | 2026-07-07 00:00 ~ 2026-07-08 10:58 | arXiv / NVIDIA / OpenAI / Anthropic / vLLM / DeepMind / Meta / MSR / PyTorch | 5 | 11 | 2026-07-08 10:58 | 按实际扫描结束时刻记录；补入高质量官方博客，未把截止时间预填到当天 23:59 |
| [2026-07-10](frontier_scan_2026-07-10.md) | 2026-07-08 10:58 ~ 2026-07-10 10:48 | arXiv / OpenAI / Anthropic / NVIDIA / Hugging Face / PyTorch / GitHub releases | 4 | 8 | 2026-07-10 10:48 | 完成重点 arXiv、官方博客和框架 release 扫描；Megatron-LM Releases API 失败，已记录为补扫项 |
| [2026-07-13](frontier_scan_2026-07-13.md) | 2026-07-10 10:48 ~ 2026-07-13 15:35 | arXiv / OpenAI / Anthropic / NVIDIA / Hugging Face / vLLM / SGLang / framework releases | 8 | 11 | 2026-07-13 15:35 | 扫描 144 条 arXiv v1；7 月 10 日无精确时刻的厂商博客按 boundary late-discovered 去重；GitHub rate limit 已回退官方 HTML / Atom 核验 |
| [2026-07-20](frontier_scan_2026-07-20.md) | 2026-07-13 15:35 ~ 2026-07-20 11:12 | arXiv / OpenAI / Anthropic / NVIDIA / Hugging Face / vLLM / SGLang / framework releases | 7 | 12 | 2026-07-20 11:12 | 覆盖 arXiv 7 个相关分类与重点官方来源；accepted 均核对 primary source；周一上午 arXiv 最新公告仍截至 7 月 17 日 |
| [2026-07-22](frontier_scan_2026-07-22.md) | 2026-07-20 11:12 ~ 2026-07-22 12:57 | arXiv / OpenAI / Anthropic / NVIDIA / Hugging Face / vLLM / SGLang / framework releases | 11 | 13 | 2026-07-22 12:57 | 合并 07-21 与 07-22 增量记录；保留逐条 First seen；Stale but Stable、moefs 等公告边界项显式标记 late-discovered |
| [2026-07-24](frontier_scan_2026-07-24.md) | 2026-07-22 12:57 ~ 2026-07-24 09:52 | arXiv / OpenAI / Anthropic / NVIDIA / Hugging Face / AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL | 7 | 10 | 2026-07-24 09:52 | 合并 07-23 与 07-24 增量记录；新增 KV eviction error certificate 与 NeMo RL TensorRT-LLM backend；GitHub API rate limit 后回退 PR patch / Atom 核验 |
