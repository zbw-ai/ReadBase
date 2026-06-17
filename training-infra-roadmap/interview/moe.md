# Interview: MoE

## 高频面试题

1. MoE 为什么能扩展参数量？
2. top-1 和 top-2 routing 有什么差异？
3. Expert Parallel 解决什么问题？
4. MoE 的 all-to-all 为什么难？
5. load balance 如何影响训练稳定性？

## 追问问题

- expert hot spot 如何监控？
- capacity factor 怎么选？
- MoE checkpoint 如何做 expert reshuffle？

## 生产环境案例

MoE 训练平均 step time 正常，但 p99 周期性升高。排查发现部分 expert token load 过高，触发 all-to-all tail latency。需要同时看 routing 分布、expert placement、网络拓扑和 batch 数据分布。

## 常见错误回答

- “MoE 参数更多所以一定更慢。”不一定，每 token 只激活部分 expert。
- “MoE 只需要加 router。”错误，系统难点主要在 expert parallel、all-to-all、负载均衡和 checkpoint。

## 优秀回答示例

MoE 用 sparse activation 扩展 total parameters，每个 token 只进入少量 expert，因此 activated compute 可控。但它把 dense all-reduce 问题变成动态 all-to-all 和负载均衡问题，生产上要重点监控 expert load、dispatch latency、dropped token 和 checkpoint 重分布。
