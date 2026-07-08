# Scan Log

`scan_log.md` 是前沿扫描账本，用来保证每次扫描都从上一次游标继续，做到不重不漏。

它不记录完整判断。完整结果写入对应的 `frontier_scan_YYYY-MM-DD.md` 文件；本文件只记录扫描窗口、来源范围、accepted / observed 数量和下一次扫描游标。

## 使用规则

- 每次执行“看看最新有没有更新”时，先查看本文件最后一行的 `Next cursor`。
- 新扫描窗口从上一次 `Next cursor` 开始，到本次扫描结束时间为止。
- 时区统一使用 `Asia/Shanghai`。
- 如果某次扫描因为网络、API limit 或来源不可访问而不完整，必须在 `Notes` 中写清楚。
- 不要因为某周没有扫描就补造 weekly；直接用更长的 frontier scan 窗口覆盖空档。
- 同一个 Source ID 不应在多个 frontier scan 中重复进入 accepted signal。重复出现时写入 follow-up 或 observed。

## Log

| Scan | Window | Sources | Accepted | Observed | Next cursor | Notes |
|---|---|---|---:|---:|---|---|
| [2026-07-04](frontier_scan_2026-07-04.md) | 2026-06-29 00:00 ~ 2026-07-04 23:59 | arXiv / GitHub / blogs / releases | 1 | 6 | 2026-07-04 23:59 | 根据对话记录补建扫描账本，完整扫描结果待整理成文件 |
| [2026-07-07](frontier_scan_2026-07-07.md) | 2026-07-05 00:00 ~ 2026-07-07 23:59 | arXiv recent / prior GitHub / blogs / releases | 3 | 10 | 2026-07-07 23:59 | 已补扫 Tue, 7 Jul 2026 latest paper entries；accepted 新增 CompactionRL 与 LLM-as-a-Verifier |
