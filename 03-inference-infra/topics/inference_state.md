# 推理状态：缓存成本、内容就绪与输入身份

[所属 Part](../README.md) · [推理路线](../roadmap.md) · [10/19来源](../../research/tracking/frontier_scan_2026-10-19.md) · [10/23增量](../../research/tracking/frontier_scan_2026-10-23.md)

## 问题：命中缓存为什么还可能更慢，甚至答错？

缓存同时回答两个问题：能否省掉计算，以及里面的状态是否属于这次请求。prefix hit rate 只描述复用机会；它没有计入远端访问、同步与恢复开销，也不证明页面已经写完。以下是公开方法/代码形成的阶段性理解，未运行模型或 GPU 实验；完整 serving 调度与容量规划仍需后续成章。

## 机制：先算状态，再判断复用

**第一层是模型状态的规模。** 单层每 token 的普通 KV 字节数可以按 `2 × KV heads × head dim × bytes/element` 估算，再按层类型区分随上下文增长、局部滑窗或固定大小的 recurrent state。乘并发以后，还要加分页、alignment、暂存和 allocator 开销；量化格式需要额外 scale，不能只乘名义位宽。

[Beam 的官方配置和实现](https://huggingface.co/blog/reflection-beam)提供一个实例：52 层中 43 层仅保留512-token窗口，9层全局 attention。只有后一部分随长上下文持续增长；这依赖 backend 实际实现了 sliding-cache 截断。MoE 的 active 参数用于理解计算路径，全部专家权重的驻留预算必须另算；“1M context、23B active”不能直接翻译成部署卡数。

**第二层是恢复所需的状态。** [Nemotron 3.5 报告 §5.3](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf)说明，attention KV 与 Mamba checkpoint 即使命中，drafter 所需的末 token 中间 hidden state 仍可能缺失。若只能从较早的 recurrent checkpoint 重放，就会重新计算一段 prefix；在恢复边界附近额外保留 checkpoint，是用内存换重放成本。不能把“SSM 状态大小固定”理解为缓存任意历史位置都免费。

**第三层是加载与重算的比较。** [Remote KV 论文](https://arxiv.org/abs/2610.21500v1)把加载拆为固定成本和传输成本。判断应是：

```text
排队 + metadata/调度 + 传输 + 同步/布局恢复 < 当前硬件上的重新 prefill
```

这只是工程分解，不是通用阈值公式。GPU 加快会使远端加载更难划算；增加存储带宽也不能消掉固定成本。在 context parallelism 中，各 rank 还必须对可恢复的连续 prefix 和 collective 次序达成一致，不能让一个 rank 失败后自行跳过通信。论文单 engine、热缓存回放的结果不覆盖多租户争用。

## 配置：容量、依赖与版本一起记录

| 必须记录 | 为什么影响判断 |
|---|---|
| checkpoint、layer layout、KV dtype、滑窗截断方式 | 决定每请求的状态增长；权重量化不自动量化 KV |
| 并发、长度分布、prefix 共享分布、冷/热比例 | 决定缓存容量、有效带宽和被淘汰状态 |
| checkpoint 粒度、draft 需要的 hidden state、replay 长度 | 决定一次命中的恢复工作 |
| GPU/CPU/remote tier、排队、copy stream、完成事件 | 决定能否及时且正确地复用 |
| encoder/processor、tool schema、模型与 backend revision | 决定相同消息是否编码成相同模型输入 |

没有适用于所有负载的 offload 阈值或缓存比例。先记录同一模型、相同有效输入和目标延迟下的冷启动、热缓存、强制重算三组结果，再决定是否启用。

## 生产陷阱与排查

[SGLang #44800](https://github.com/sgl-project/sglang/pull/44800)展示了 ownership 与 ready 的区别：拿到 cache page 并不意味着 forward 已把新 KV 写完。write-back copy 必须等 producer 完成，再允许发布/复用；否则缓存可以稳定地“命中”旧内容。对应 CUDA 回归刻意延迟 producer，随后驱逐再检查 host snapshot；源码存在测试不等于所有 CI 或硬件组合已通过。

[DeepSeek encoder 修复](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/commit/314113f8a7dadb388215396ba5fbee2c610a04d6)则发生在模型前：工具结果排序应先于图像提取，工具返回本身不应错误地启动新用户回合并清掉当前 reasoning。缓存与执行都正确，也可能因为 processor 改变了输入身份而得到不同输出。

| 现象 | 优先排查 | 建议验收（尚未执行） |
|---|---|---|
| hit 增多、TTFT 变差 | metadata 固定开销、排队、跨 rank 等待和重放长度 | 同条件 load-vs-recompute，分开 median 与 tail |
| 冷跑正确，命中后偶发错 | producer/copy/publish event，state owner 与版本 | 延迟 producer→驱逐→回读，检查内容而非只看 hit |
| 换 encoder 后工具图片或续写异常 | 最终 tool order、payload 顺序、user-turn identity | 乱序结果、混合图片、同回合多次 tool call 对照 |
| 标称长上下文却 OOM | 全权重、KV层布局、并发、临时 buffer 与分页 | 逐项内存账本；不要只按 active 参数或输出长度估算 |

<a id="oct23-capacity"></a>
## 2026-10-23 补充：容量、逻辑地址和允许释放不是同一件事

[UlaCache](https://arxiv.org/abs/2610.29820v1)说明稀疏attention减少当步访问量，却仍可能需要保存全部历史。稳定logical address可以把KV的逻辑身份与CPU/GPU位置分开；换页必须先等旧读者结束，完成数据搬移，再发布mapping。其评测先完成prefill后测decode，不能把峰值decode吞吐当在线TTFT/SLO结论。

[uGDS](https://arxiv.org/abs/2610.30868v1)进一步把CPU提交开销与payload DMA分开。超时意味着“不再等待”，未必意味着“设备已停止写入”；隔离失效queue和映射，待controller reset等停止条件成立才能安全回收。它需要块设备所有权，不是任意Python文件读取接口的直接替代。

[SGLang #45453](https://github.com/sgl-project/sglang/pull/45453)处理跨rank释放：局部ACK不同会使TP ranks看到不同容量，随后admission与collective次序也分叉。先共同决定是否hold，再共同决定是否ready，不能由一个rank自行提前释放；host目标仍要求drain，device既有timeout分支并不证明所有remote write已结束。

排查时至少列四列：**逻辑owner、物理位置、producer/readers完成、跨rank可回收条件**。测试要覆盖迟到写、错开ACK、映射重注册和独立请求推进。公开CPU/Gloo回归不替代真实GPU/RDMA验证；方法与CI边界见[10/23扫描](../../research/tracking/frontier_scan_2026-10-23.md)。

同样的生命周期判断适用于[OpenAI旁路广告服务](https://openai.com/index/building-chatgpt-ad-stack/)：主响应到期可以舍弃未就绪辅助结果，但“业务上不再接收”不自动证明后台计算已取消、资源已释放或共享资源无干扰。服务协议和资源回收应分别验收。

## 相邻系统

这些问题独立存在于普通 serving，也会传到 [RL rollout](../../04-rl-infra/topics/agentic_rl.md)：错误输入、未完成权重/状态传输会改变训练轨迹。具身模型的 [register memory、环境 latent 与动作工件](../../05-embodied-infra/topics/embodied_infra.md#oct19-learning)也有类似的 reset/owner/version 需求，但语义不同，不能直接套用 LLM KV 的淘汰策略。

下一步在共享阅读队列按问题选读，不把本章当作性能复现或完整推理手册。完整证据、原始日期和限制见 [10/19 扫描](../../research/tracking/frontier_scan_2026-10-19.md)。
