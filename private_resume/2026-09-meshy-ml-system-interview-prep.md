# Meshy 技术专项旧入口（已并入主文档）

更新：2026-09-09。本页仅兼容此前保存的链接，**不再维护独立技术答案**，无需把它作为另一份学习材料。

日常复习与现场查询统一使用 [主文档](2026-08-llm-infra-interview-prep.md#interview-console)；新增基础题在 [GPU / PyTorch / 低精度 Part](2026-08-llm-infra-interview-prep.md#part-foundations)，长代码在 [Coding 题单](2026-09-interview-coding.md#coding-top)。原技术内容已归入主文档和已有 topic 章节，旧版本仍可在 Git 历史查看。

<a id="meshy-interview-top"></a>
[主文档 Meshy 复习入口 → 当前维护位置](2026-08-llm-infra-interview-prep.md#meshy-interview-sprint)

<a id="meshy-plan"></a>
[分日学习安排 → 当前维护位置](2026-08-llm-infra-interview-prep.md#meshy-plan)

<a id="meshy-a1"></a>
[浮点格式 → 当前维护位置](2026-08-llm-infra-interview-prep.md#precision-01)

<a id="meshy-a2"></a>
[FP8 scaling → 当前维护位置](2026-08-llm-infra-interview-prep.md#precision-02)

<a id="meshy-a3"></a>
[Linear 前后向与混合精度 → 当前维护位置](2026-08-llm-infra-interview-prep.md#pytorch-03)

<a id="meshy-a4"></a>
[低精度排障 → 当前维护位置](2026-08-llm-infra-interview-prep.md#precision-03)

<a id="meshy-a5"></a>
[MXFP8 / NVFP4 → 当前维护位置](2026-08-llm-infra-interview-prep.md#precision-04)

<a id="meshy-b1"></a>
[GPU 执行与存储 → 当前维护位置](2026-08-llm-infra-interview-prep.md#gpu-01)

<a id="meshy-b2"></a>
[Roofline → 当前维护位置](2026-08-llm-infra-interview-prep.md#gpu-02)

<a id="meshy-b3"></a>
[GPU 卡型与互联 → 当前维护位置](2026-08-llm-infra-interview-prep.md#gpu-01)

<a id="meshy-c1"></a>
[Autograd / stride → 当前维护位置](2026-08-llm-infra-interview-prep.md#pytorch-01)

<a id="meshy-c2"></a>
[torch.compile → 当前维护位置](2026-08-llm-infra-interview-prep.md#pytorch-02)

<a id="meshy-c3"></a>
[FSDP2 → 当前维护位置](2026-08-llm-infra-interview-prep.md#dist-01)

<a id="meshy-d1"></a>
[DataLoader → 当前维护位置](2026-08-llm-infra-interview-prep.md#infra-11)

<a id="meshy-d2"></a>
[扩容与稳定性 → 当前维护位置](2026-08-llm-infra-interview-prep.md#infra-09)

<a id="meshy-e1"></a>
[Diffusion / Flow Matching → 当前维护位置](2026-08-llm-infra-interview-prep.md#gen-01)

<a id="meshy-e2"></a>
[多阶段推理 → 当前维护位置](2026-08-llm-infra-interview-prep.md#gen-02)

<a id="meshy-e3"></a>
[3D 表征 → 当前维护位置](2026-08-llm-infra-interview-prep.md#gen-03)

<a id="meshy-f1"></a>
[Linear 梯度检查与模型代码 → 当前维护位置](2026-09-interview-coding.md#coding-05)

<a id="meshy-f2"></a>
[eager / compile benchmark → 当前维护位置](2026-09-interview-coding.md#coding-06)

<a id="meshy-f3"></a>
[Triton 融合 → 当前维护位置](2026-09-interview-coding.md#coding-07)

<a id="meshy-g1"></a>
[配置环境 → 当前维护位置](2026-09-interview-coding.md#coding-env)

<a id="meshy-g2"></a>
[项目开场与映射 → 当前维护位置](2026-08-llm-infra-interview-prep.md#meshy-project-fit)

<a id="meshy-g3"></a>
[公司公开工作交流 → 当前维护位置](2026-08-llm-infra-interview-prep.md#meshy-public-work)

<a id="meshy-g4"></a>
[主管 / CEO 交流 → 当前维护位置](2026-08-llm-infra-interview-prep.md#meshy-questions)
