# Meshy 公司资料与技术专项旧链接

更新：2026-09-11。本页保留公司公开工作资料并兼容此前保存的技术链接，**不维护独立技术答案**。主文档已恢复按技术主题组织，不再保留公司专项冲刺入口。

日常复习与现场查询统一使用 [主文档](2026-08-llm-infra-interview-prep.md#interview-console)；新增基础题在 [GPU / PyTorch / 低精度 Part](2026-08-llm-infra-interview-prep.md#part-foundations)，长代码在 [Coding 题单](2026-09-interview-coding.md#coding-top)。原技术内容已归入主文档和已有 topic 章节，旧版本仍可在 Git 历史查看。

<a id="meshy-interview-top"></a>
[通用复习与现场查询 → 主文档](2026-08-llm-infra-interview-prep.md#interview-console)

<a id="meshy-plan"></a>
[通用学习安排 → 主文档](2026-08-llm-infra-interview-prep.md#vi-0)

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
[自我介绍](2026-08-llm-infra-interview-prep.md#resume-01) · [项目证据卡](2026-08-llm-infra-interview-prep.md#vi-evidence-cards)

<a id="meshy-g3"></a>
[公司公开工作交流 → 本页历史资料](#meshy-public-work)

<a id="meshy-g4"></a>
[按面试官角色反问 → 主文档](2026-08-llm-infra-interview-prep.md#vi-questions-to-ask)

---

<a id="meshy-public-work"></a>
## 公司公开工作：历史准备资料

**用法**：选一个真正理解的点，只用 20–30 秒说明观察，再问系统取舍。不需要泛泛赞美，也不要把对方尚未公开的组织安排说成你已确认的事实。

### 1. Meshy T2：从逐 token 生成转向并行 Flow Matching

公开论文 **Meshy T2: Fast Native Mesh Generation with Flow Matching** 首发 2026-07-28，v3 更新于 2026-08-12。它以每顶点一个连续 latent 的 mesh VAE 表示几何，先用 voxel flow 生成粗占据结构，再由 mesh flow 结合图像、结构和 vertex budget 生成 mesh latent；不是逐 mesh token 自回归。[论文与作者元数据](https://arxiv.org/abs/2607.28675)

**可以这样问（工程推论，不是论文实测结论）**：

> 我看了 T2 的公开架构。它把生成变成 coarse-to-fine 的 flow，系统优化的重点会从逐 token decode 转到多步网络调用、可变 vertex 数和最终解码。我比较好奇，实际最值得优化的是去噪/flow 主干、VAE 解码，还是不同 vertex budget 下的 shape 和组批？

不背未经复现的速度数字。核验时[官方仓库](https://github.com/meshy-dev/meshy-t2)仍主要提供介绍/素材及待发布信息，不能声称已经运行其完整代码。公开研究路线也不等于公司所有线上模型。[历史补录与阅读边界](../training-infra-roadmap/tracking/backfill/2026-07.md#meshy-t2)

### 2. MakerWorld / Bambu：从生成完成到资产可用

Meshy 2026-03-17 发布的合作消息介绍了 Image-to-3D 接入 MakerWorld/MakerLab，并将多色 3MF 接入 Bambu Studio 工作流。这能支持“生成结果需要服务实际打印流程”，**不能推出模型在打印机上推理、联合设计 GPU 或终端芯片**。[公司发布的合作消息](https://www.prnewswire.com/news-releases/how-to-turn-any-image-into-a-full-color-3d-print-in-one-click--meshys-multi-color-printing-powered-by-meshy-6-is-now-live-on-makerworld-302714800.html)

> 我关注到你们和 MakerWorld 的合作。对 ML Systems 来说，我理解指标不应只看生成速度，还要看模型能否顺利进入切片和打印流程。做低精度或推理优化时，团队会用什么几何/可打印性指标做回归门禁？下游失败会反馈到数据还是模型评估里？

这是从个人“性能＋正确性”经验出发的讨论，不假定自己懂打印机硬件。

### 3. 视频方向：联系真实经验，再确认岗位覆盖

截至核验时，[官网视频入口](https://www.meshy.ai/video)可见 Image-to-Video/Text-to-Video 等产品选项；**仅凭入口无法确认模型是否自研、所用后端或视频团队成立时间**。用户转述的“新视频团队”保留为待面试确认的信息。

> 我也注意到官网的视频生成入口。我以前做过文生视频模型的适配和性能优化，比较熟悉从模型跑通、精度对齐到并行和算子瓶颈定位的过程。想了解这个岗位会主要服务 3D，还是也覆盖视频方向？两边在数据读取、编译、低精度和性能回归上，会更倾向共用基础设施还是先独立迭代？

**不要说**：“听说新组了视频团队，你们一定用某某 DiT”“和拓竹合作，所以一定会考端侧芯片”“T2 就是你们生产模型”。公开资料用于提出好问题，不用于填补未知事实。

资料保留原核验口径，不自动更新为当前招聘安排；技术机制以主文档通用题为准。

[返回主文档](2026-08-llm-infra-interview-prep.md#interview-console)
