# Release Notes Tracking

用于记录模型、框架、训练栈、kernel 库、分布式训练组件的发布记录。

重点关注：

- Megatron-Core
- Transformer Engine
- PyTorch Distributed / FSDP
- DeepSpeed
- NCCL
- FlashAttention
- vLLM / SGLang / inference stack
- CUDA / Hopper / Blackwell training features

## 模板

```text
## YYYY-MM-DD

### 项目版本

- 来源：
- 链接：
- 类型：release note
- 影响等级：
- Decision：Ignore / Observe / Read / Deep Dive
- Reason：
- 建议动作：
- Status：
- 影响模块：parallelism / memory / kernel / checkpoint / network / scheduler / RL rollout
- 一句话价值：
- 需要验证的点：
```

## Backlog

暂无。

## 2026-09-22 GitHub 补扫

- [verl v0.9.1](https://github.com/verl-project/verl/releases/tag/v0.9.1)，9/20 发布，Impact 高 / Decision Read：V1 async、准入/refit 与迁移边界，关联 Agentic RL；下一步按实际 backend pin 核对兼容性。
- [vLLM v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)，9/22 发布，Impact 高 / Decision Read：PP speculative、低精度和服务入口契约，关联 rollout；下一步做版本验收表，不自动升级环境。
- [tokenizers v1.0.0-rc.2](https://github.com/huggingface/tokenizers/releases/tag/v1.0.0-rc.2)，9/21 prerelease，Impact 中 / Decision Observe：补此前博客证据，待真实 input pipeline 验证。

作者账号、精确发布时间、Source ID 与证据边界见[专项报告](github_audit_2026-09-22.md)和[release CSV](audits/2026-09-22-github/releases.csv)。TRL v1.14.0 尚无可核验 tag/release，不能按博客声明记为正式发布。
