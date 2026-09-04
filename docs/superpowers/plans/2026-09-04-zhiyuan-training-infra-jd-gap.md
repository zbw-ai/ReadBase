# 智元机器人训练 Infra JD 临场补题实施计划

**Goal:** 在主面试文档中新增四道岗位差集题、扩展一条现有框架选型题，并提供一个 30 分钟稳定入口。

**Source of truth:** `docs/superpowers/specs/2026-09-04-zhiyuan-training-infra-jd-gap-design.md`

## Task 1：更新全局导航与计数

- 在控制台“现场救急”增加 `[智元训练 Infra（30min）](#vi-0a)`。
- 更新 Part 表：Part II=`27`、Part III=`16`、Part V=`10`；总题量=`78`，P0/P1/P2=`47/26/5`。
- 更新折叠式全量索引和各 Part 本地导航：
  - Part II 增加 `SFT-DATA-01`、`MLLM-01`；
  - Part III 增加 `DPO-01`；
  - Part V 增加 `TRAIN-ANOMALY-01`；
  - `MEGATRON-11` 标签扩展为四类框架的分层与选型。
- Core 10 的内容和顺序不变。

## Task 2：写入 Part II 三项内容

- 扩展 `MEGATRON-11`：Accelerate 是 facade/orchestration；DeepSpeed 是含 ZeRO/offload/pipeline 的 runtime；FSDP2 是 PyTorch-native DP sharding；Megatron 负责多维模型并行与高性能训练。强调它们可分层组合，不是简单四选一。
- 在 `MEGATRON-11` 后新增 `SFT-DATA-01`，覆盖清洗去重、schema/chat template、tokenizer、truncation、packing、position/attention/loss mask、分布式 shard/shuffle、data cursor、held-out eval。
- 新增 `MLLM-01`，覆盖媒体 decode/preprocess、dynamic shape/token、视觉编码器/投影层、时空 position/mask、负载均衡与长序列；具身只讲 observation/action/episode 时序和闭环评测约束。
- 明确经验边界：DeepSpeed/Accelerate/FSDP 是机制与选型；MLLM 直接证据只到 TX 模型迁移及 Capek Infra 承载。

## Task 3：写入 Part III 与 Part V

- 在 `RL-ALGO-01` 后新增 `DPO-01`：用 preferred/rejected 对的 policy-reference log-ratio margin 解释目标；与 SFT、PPO/GRPO 对比；列出 paired data、reference logprob、response mask 和 length bias 等系统检查。
- 在 Part V 的 P0 扩展首位新增 `TRAIN-ANOMALY-01`：按现场保护、数据、forward 数值、backward 梯度、optimizer/hyperparameter、distributed consistency、收敛评估排查；交叉链接 `INFRA-02` OOM 与 `INFRA-03` 通信故障，不复制正文。
- 四个新题均添加正确所属 Part 与总控制台双回链。

## Task 4：增加 VI.0A 30 分钟入口

- 在 `VI.0` 后、三天计划前增加 `<a id="vi-0a"></a>`。
- 写入 8/6/6/5/5 分钟表，依次链接异常、框架、SFT data contract、DPO、多模态/具身。
- 只提供跳转和每题必须记住的一句话，不复制完整答案。

## Task 5：验证、审查和发布

1. Fence-aware 统计 78 道 H4 P0/P1/P2 问题，期望 `47/26/5`，Part I–V=`7/27/16/18/10`。
2. 对每道问题 body 验证最后一个非空块恰好是正确 Part 的双回链，代码/Mermaid fence 内无伪回链。
3. 检查四个新锚点、`vi-0a`、所有显式 anchor 唯一；所有 same-file/local links 和 images 可解析；SVG XML 可解析。
4. 检查四个新题均进入全量索引、本 Part 导航和 VI.0A；`MEGATRON-11` 不新增第二个题目。
5. 检查 Core 10 未改变；`git diff --check`；允许变更仅为 spec、plan、主文档。
6. 发布前复核主工作树指纹：status 三行、dirty diff SHA-256=`d01efc17b7ebb0c2be22c762e5fa228823f3c60e2e5354f2a1d11a468a7540f8`、目标主文档 SHA-256=`a4af2de96006c02e27205eb49e0ca2a7beee7502163088bfda31bccc7976aaee`。
7. `git fetch origin` 后确认 `origin/main` 是 HEAD 祖先，只执行 `git push origin HEAD:main`；发布后确认远端 `main==HEAD` 且无其他远端分支。
