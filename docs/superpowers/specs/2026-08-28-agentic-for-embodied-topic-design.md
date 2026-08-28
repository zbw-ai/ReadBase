# Agentic for Embodied Topic Design

## Goal

Create `training-infra-roadmap/topics/agentic_for_embodied.md` as an engineering-system map for an AI Infra engineer. The chapter should establish enough robotics background to understand the workload, then explain how an embodied-agent platform is implemented from data collection and simulation through training, evaluation, deployment, and fleet feedback.

The chapter is not a robotics-control textbook and not a model-release catalog.

## Audience And Success Criteria

The reader is familiar with LLM training, distributed systems, GPU clusters, inference engines, and Agentic RL, but not necessarily robotics.

After reading, the reader should be able to:

1. Distinguish an embodied agent from a text-only agent and explain the extra system boundaries.
2. Place VLA, world model, imitation learning, diffusion policy, RL, simulator, robot runtime, and safety controller in one end-to-end architecture.
3. Explain why robot data, rollout, reset, simulation, real-time inference, and sim-to-real are infrastructure problems.
4. Identify which AReaL/Agentic RL capabilities can be reused and which embodied capabilities require new services.
5. Design a phased production embodied-agent training platform and define its core interfaces, schemas, resource pools, SLOs, state transitions, recovery paths, and failure domains.

The chapter has three explicit exit criteria:

- **Approach A exit:** a workload-property -> infrastructure-consequence -> platform-decision -> evidence map.
- **A-to-C bridge:** one bounded reference workload with service ownership, data schemas/APIs, state machines, SLOs, failure recovery, and a capacity model.
- **Approach C exit:** a phased MVP blueprint with acceptance tests and at least one concrete follow-up candidate for `experiments/`, `playbooks/`, or an engineering decision record.

## Scope

### Included

- Minimal robotics vocabulary: observation, state, action space, control frequency, embodiment, teleoperation, episode, reset, calibration, localization, and safety envelope.
- Model families only at the depth needed to explain infrastructure consequences: VLM/VLA, policy model, diffusion/flow policy, world model, planner, and low-level controller.
- Learning routes only when they change the data path, rollout path, resource model, or correctness contract: behavior cloning/imitation learning, offline RL, online RL, reinforcement fine-tuning, and hybrid hierarchical control.
- Data systems: multimodal trajectory schema, timestamp synchronization, sensor calibration metadata, data quality, replay, versioning, and lineage.
- Simulation and rollout infrastructure: parallel environments, GPU simulation, environment reset, domain randomization, synthetic data, real-robot rollout, and scheduler design.
- Training and serving: heterogeneous GPU/CPU/robot resources, training-serving conversion, edge inference, latency budgets, batching limits, checkpointing, evaluation, and fleet feedback.
- Sim-to-real, safety, observability, and production failure modes.
- Representative primary-source systems and open implementations.
- A concrete production-platform reference architecture and an AReaL transferability analysis.

### Excluded Or Kept Shallow

- Control-theory derivations, kinematics, dynamics, SLAM mathematics, and motion-planning proofs.
- Detailed mechanical design, actuator design, and robot hardware selection.
- Exhaustive benchmark tables or chronological model-release summaries.
- Claims based only on promotional posts without primary technical evidence.

## Chapter Structure

1. Topic positioning and why Infra engineers should care.
2. Minimal robotics background.
3. Workload-property -> infrastructure-consequence -> platform-decision -> evidence map.
4. Core solution families, admitted only when they change an interface, resource pool, SLO, failure mode, or validation strategy.
5. End-to-end embodied learning loop.
6. Data and trajectory infrastructure.
7. Simulation, rollout, and environment infrastructure.
8. Training, inference, and real-time runtime.
9. Evaluation, sim-to-real, safety, and observability.
10. Transferability from Agentic RL/AReaL.
11. Reference workload, contracts, and capacity model.
12. Production reference architecture and phased MVP.
13. Configuration and capacity-planning judgments.
14. Common failures and troubleshooting order.
15. Representative work and staged reading path.
16. Interview questions and production design questions.
17. Engineering summary and concrete follow-up artifact.

## Core Narrative

The chapter should use one causal chain throughout:

```text
Physical task
  -> multimodal observation and action trajectory
  -> data curation / simulation / replay
  -> policy or world-model training
  -> evaluation and safety gates
  -> edge/robot deployment
  -> fleet rollout and failure collection
  -> new training data
```

The main engineering judgment is that embodied intelligence is not merely an LLM with camera input. It is a closed-loop cyber-physical training system whose data source, environment, evaluator, inference target, and failure cost are all different from text-only Agentic RL.

## Infrastructure Admission Test

A model, algorithm, paper, report, or repository enters the chapter body only if it changes at least one of:

- A service or API boundary.
- A resource pool or scheduling unit.
- A latency, throughput, freshness, safety, or availability SLO.
- A production failure mode or recovery procedure.
- A validation, deployment, or evidence strategy.

Other representative work belongs in the staged reading path. This keeps the chapter engineering-first rather than turning it into a robotics algorithm catalog.

## Evidence Strategy

Use current primary sources and clearly separate:

- Paper-verified mechanisms and benchmark claims.
- Official technical reports and model cards.
- Open-source repository capabilities that are visible in code or documentation.
- Vendor-reported production claims.
- Inference by the handbook author.

The initial research set should cover a small number of representative lines such as RT-1/RT-2/Open X-Embodiment, OpenVLA or Octo, Diffusion Policy, π0/π0.5, GR00T, Gemini Robotics, GPU simulation stacks, and open robotics data/runtime ecosystems. Final inclusion depends on primary-source verification and the infrastructure admission test.

Historical sources not already tracked should first enter the appropriate `tracking/backfill/YYYY-MM.md`; current sources should follow the frontier-scan and reading-queue workflow. The topic itself starts at lifecycle status `NEW` and advances only when the corresponding knowledge artifact exists.

## Reference Workload

Use one illustrative, bounded scenario throughout the A-to-C bridge:

- Fixed-base dual-arm manipulation in a warehouse or lab cell.
- Two RGB cameras plus wrist cameras, proprioception, gripper state, language task input, and optional force/torque signals.
- A high-level VLA/policy producing action chunks, with a deterministic low-level controller executing at a higher frequency.
- Mixed training data from teleoperation, replay, GPU simulation, and guarded real-robot rollouts.
- A small robot fleet and a larger simulator pool, with edge inference and a central training platform.

All numbers are illustrative rather than vendor claims. The chapter must calculate or show formulas for trajectory ingress/storage, simulator throughput, reset latency, training demand, edge deadline-miss rate, and fleet model-rollout rate.

## AReaL Transfer Decision Artifact

Require a matrix with these rows: `rollout`, `training`, `scheduler`, `weight sync`, `data/trajectory`, `checkpoint/recovery`, and `inference backend`. Each row must classify the capability as **reuse / adapt / replace / new**, identify the retained contract, and state the embodied-system gap.

## Production Architecture Deliverable

The chapter contains a concrete architecture with at least these planes:

- Robot/fleet and teleoperation plane.
- Data ingestion, synchronization, validation, and trajectory lake.
- Simulation/environment plane.
- Rollout and evaluation scheduler.
- Training plane for VLA/policy/world model.
- Model registry, conversion, and deployment plane.
- Robot runtime with policy inference, deterministic controller, and an independent safety authority.
- Observability, incident response, and feedback plane.

For each plane, document owner, inputs, outputs, schema/API, resource type, scaling unit, state transitions, critical SLOs, principal failure modes, and recovery behavior.

Safety must be an independent production boundary rather than a box inside the policy service. The design must include artifact-compatibility gates, offline validation, shadow/canary rollout, rollback, E-stop/fallback behavior, safety audit events, and incident criteria. Control algorithms remain shallow; authority and timing contracts must be explicit.

The phased MVP should distinguish at least:

1. Offline data and imitation-learning loop.
2. GPU simulation plus automated evaluation.
3. Guarded real-robot rollout and fleet feedback.

Each phase needs acceptance tests that cover schema compatibility, latency/deadline behavior, safety gates, recovery, observability, and reproducibility.

## Repository Integration

- Add the topic to `training-infra-roadmap/README.md` as an emerging cross-cutting engineering topic.
- Add bidirectional relationships in `training-infra-roadmap/KNOWLEDGE_GRAPH.md` with Agentic RL, long-context training, distributed training, and fault tolerance.
- Add a small relationship block in `topics/agentic_rl.md` pointing to the new topic.
- Update `MASTER_READING_LIST.md` only if the representative-source section creates a meaningful new reading route.
- Do not modify current frontier-scan content or mix historical embodied sources into a current scan.

## Quality And Verification

- Author the handbook prose in Chinese while retaining standard English technical terms.
- Prefer engineering judgments over definitions and model marketing.
- Every major system claim must link to a primary source.
- Explain evidence boundaries for vendor performance claims.
- Keep local links bidirectional and verify all Markdown links.
- Do not add an SVG in the first pass; use a compact Mermaid architecture only if it improves navigation without crowding the chapter.
- Verify that the final chapter includes the workload map, bounded reference workload, schemas/APIs, AReaL transfer matrix, SLOs, sizing formulas, state/recovery paths, safety boundary, phased MVP, acceptance tests, evidence labels, and navigation links.
- End with one concrete experiment, playbook, or engineering decision record candidate instead of only suggesting future work abstractly.
