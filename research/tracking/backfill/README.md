# Historical Backfill By Month

阅读主入口是[2025 按季度、2026 按月的复盘](../monthly_reviews.md)；本页及按原始月份保存的 backfill 仅作为轻量来源账本。2026-09-22 复盘批次于 2026-10-08 继续补登记，不把补登记时间冒充材料首发时间。

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

- [2026-09](2026-09.md)：MiMo GAGAR 论文、工具调用重复修复与开源 RL 环境的后续证据

- [2026-08](2026-08.md)：B300 排障、kernel verifier、环境生产与 lazy-pull 历史重评
- [2026-07](2026-07.md)：BPO / kernel harness 历史重评；NVIDIA GR00T end-to-end embodied platform
- [2026-06](2026-06.md)：PyTorch Miles / RL post-training infra；MiMo × TileRT 推理 codesign
- [2026-05](2026-05.md)：Hugging Face TiTo / Agentic RL token correctness
- [2026-04](2026-04.md)：本轮历史精选，阅读主线见季度/月度复盘
- [2026-03](2026-03.md)：本轮历史精选，阅读主线见季度/月度复盘
- [2026-02](2026-02.md)
- [2026-01](2026-01.md)：美团 verl Fully Async / streaming / partial rollout；MiMo-V2-Flash
- [2025-12](2025-12.md)：DeepSeek-V3.2、Nemotron 3 Nano 历史精选；MiMo-Audio 论文背景核验
- [2025-11](2025-11.md)：Isaac Lab / GPU simulation infra；MiMo-Embodied 背景核验
- [2025-10](2025-10.md)：本轮历史精选，阅读主线见季度/月度复盘
- [2025-09](2025-09.md)：Gemini Robotics 1.5 / LeRobotDataset v3
- [2025-08](2025-08.md)：Agent Lightning / GLM-4.5 ARC
- [2025-07](2025-07.md)：本轮历史精选，阅读主线见季度/月度复盘
- [2025-06](2025-06.md)：Real-Time Action Chunking；MiMo-VL 混合 RL
- [2025-05](2025-05.md)：AReaL；MiMo-7B 有效样本调度
- [2025-04](2025-04.md)
- [2025-03](2025-03.md)
- [2025-02](2025-02.md)：本轮历史精选，阅读主线见季度/月度复盘
- [2025-01](2025-01.md)
- [2024-12](2024-12.md)
- [2024-10](2024-10.md)：π0 / flow-based VLA runtime
- [2024-09](2024-09.md)
- [2024-06](2024-06.md)：OpenVLA
- [2024-05](2024-05.md)
- [2023-10](2023-10.md)：Open X-Embodiment
- [2023-08](2023-08.md)
- [2023-07](2023-07.md)：RT-2 / VLA
- [2023-03](2023-03.md)：Diffusion Policy
- [2023-01](2023-01.md)：DreamerV3 / learned world-model rollout

待补：

- SkyRL：原始月份待确认，暂存于 [Historical Backfill](../historical_backfill.md)。
- NVIDIA NeMo RL / Megatron RL 等持续更新文档，需按具体 release 或文档版本确认。
- Ray RLlib / Ray Train：属于长期演进材料，需拆到具体版本节点后再归档。
