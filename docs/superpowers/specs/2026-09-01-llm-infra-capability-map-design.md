# LLM Infra 个人能力图设计规格

## 目标

替换现有信息密集、卡片化的个人能力地图，为面试主文档提供一张可快速扫读的论文式结构图。图只负责回答“能力结构是什么”，项目数字、职责边界和展开说明继续由 Markdown 正文承载。

## 已确认方向

采用“论文式双主线脊柱图”：顶部以纯文字 `LLM Infra` 定义主题，下方用两条水平主线分别表达 `Megatron / Training Systems` 与 `RL / Agentic / Post-training Systems`。不使用中心黑色块、放射树、卡片、外框、页眉页脚或水印。

## 信息结构

第一条主线为 `Megatron`，包含三个能力点：

- 并行与 MoE：`5D · Grouped GEMM · Fusion`
- 通信与显存：`Collective · Overlap · CP / SP`
- 性能与规模：`Profile · MFU · Stability`

第二条主线为 `RL / Agentic`，包含三个能力点：

- RLVR：`verl · GRPO · Fully Async`
- Agentic Rollout：`AReaL · Gateway · Cohort`
- 能力汇聚：`MOPD · Multi-Teacher · Correctness`

底部仅保留一句能力归纳：`Scale · Performance · Correctness`。

## 视觉规范

- 画布采用接近白色的暖色背景，正文为深灰黑。
- Training 使用克制的青绿色，Post-training 使用低饱和陶土色；颜色只标识主线和节点。
- 使用严格的水平网格、等距节点、细线和空心圆，避免树状曲线与视觉交叉。
- 中英文标题采用系统无衬线字体；英文分类标识采用 monospace、小字号和宽字距。
- 关键词只保留一行，不在图中放项目指标、长句、解释或图例。
- SVG 必须在 GitHub Markdown 中直接显示，并在常见宽度缩放后保持可读。

## 仓库落地

- 替换 `private_resume/assets/llm-infra-personal-capability-map.svg`，保持文件名不变，避免修改主文档链接。
- 保留 `private_resume/2026-08-llm-infra-interview-prep.md` 现有图片位置；根据新图语义缩短 alt text。
- 不修改主文档其他内容，也不把预览服务文件纳入版本控制。

## 验收标准

1. SVG 可通过 XML 解析。
2. 主文档中的相对图片路径可解析。
3. 画面无文字重叠、裁切或横向溢出。
4. 在不阅读正文时，读者能在约 10 秒内识别两条主线和六个能力点。
5. Git diff 只包含设计规格、正式 SVG 和必要的主文档 alt text 调整。

