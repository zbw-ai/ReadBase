# Historical Backfill By Month

`backfill/` 按材料的原始发布时间月份倒序补录历史精华材料。它不按主题拆分，避免把历史补课切得太碎。

## 使用规则

- 文件名使用 `YYYY-MM.md`，表示材料原始发布时间所在月份。
- 如果材料是持续更新文档、repo 或 release train，没有明确原始月份，按本次补录所依据的版本或 release 月份归档，并在条目中说明。
- 每个月一个文件，文件内部可以按方向轻量分组，例如 Agentic RL、Training Stack、Inference Infra。
- 不追求全量回填。只补能填补当前工程判断缺口的材料。
- 每条材料必须给出 `Decision` 和 `Reason`。
- 进入 P0/P1 的材料必须说明它解决哪个当前判断缺口。
- 读完后应流向 `papers/`、`tech_reports/`、`engineering_blogs/`、`topics/`、`insights/`、`experiments/` 或 `playbooks/`。

## 月份索引

已开始整理：

- [2026-02](2026-02.md)
- [2025-08](2025-08.md)
- [2025-05](2025-05.md)
- [2025-04](2025-04.md)
- [2025-03](2025-03.md)
- [2025-01](2025-01.md)
- [2024-12](2024-12.md)
- [2024-09](2024-09.md)
- [2024-05](2024-05.md)
- [2023-08](2023-08.md)

待补：

- SkyRL：原始月份待确认，暂存于 [Historical Backfill](../historical_backfill.md)。
- 2026-06：NVIDIA NeMo RL / Megatron RL 等持续更新文档，需按具体 release 或文档版本确认。
- Ray RLlib / Ray Train：属于长期演进材料，需拆到具体版本节点后再归档。
