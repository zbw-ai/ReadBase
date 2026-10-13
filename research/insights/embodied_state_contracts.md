# 具身系统判断：先定义正确状态，再优化局部速度

日期：2026-10-13。类型：由公开证据形成的工程判断；尚未通过本仓库实验验证。

## 判断与适用范围

在多相机、action chunk 与异步生成的机器人操作 workload 中，只优化模型 forward 或数据吞吐不足以保证动作可用。应该先固定样本、模型状态、执行提交与评估的契约，再比较局部加速是否改善闭环结果。这不等于否认模型能力：错误计划仍可能需要更好的训练与模型，不能靠系统契约弥补。

## 支撑与推断分开

- [VLA-Feedback](https://arxiv.org/abs/2609.21022v1)披露模块推理与端到端反馈延迟的明显差距；推断：优化目标应覆盖 sensor→actuation，不只 GPU forward。
- [LeRobot flow convention](https://github.com/huggingface/lerobot/pull/4075)通过方向、符号与 prefix 测试约束公共求解器；推断：同名 flow API 不代表同一种时间与动作语义。
- [TorchCodec surface 复用](https://github.com/meta-pytorch/torchcodec/pull/1758)提供时间信息以外的像素错误反例；推断：数据 lineage 还需配合内容 oracle，不能只验 index。
- [MEM](https://arxiv.org/abs/2603.03596v1)与 [RECAP](https://arxiv.org/abs/2511.14759v1)扩展记忆和经验学习接口；推断：checkpoint 之外，还需明确 episode memory、接管区间与训练标签版本的生命周期。

这些证据不证明所有系统都应采用同一个平台架构，更不证明额外校验没有成本。

## 如何证伪或修正

固定任务、模型、数据和控制配置，先设计[契约练习](../../practice/experiments/embodied_contract_lab.md)，再运行单变量对照：如果局部优化保持所有语义不变，且端到端与闭环结果同步改善，就可以把瓶颈归到该阶段；若没有改善，应查另一个阶段或模型能力，而不是继续无限增加验证流程。

应记录有效动作、观测年龄、deadline miss、任务成功／失败／超时／接管以及计算资源；指标必须适配具体任务，不能压成一个通用分数。实验未执行，不填预期收益。

[核心 topic](../../05-embodied-infra/topics/embodied_infra.md) · [数据正文](../../05-embodied-infra/topics/embodied_data_io.md) · [完整来源与边界](../tracking/embodied_followup_2026-10-13.md)
