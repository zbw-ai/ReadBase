# Original Resume Format Contract

## Reference

- Source: `/Users/zengbw/Downloads/曾柏炜-清华大学-华为在职-2年经验(简版).pdf`
- SHA-256: `1f707d179eea0a0a51c6988b421c57c3043a737cfca757d5055e58585abf2d4e`
- Page count: 2
- Page size: A4 portrait, 595.92 x 842.88 pt
- Visual evidence: `/Users/zengbw/ReadBase/private_resume/tmp/original-render/page-1.png`, `page-2.png`
- Extracted portrait: `/Users/zengbw/ReadBase/private_resume/tmp/original-photo.png`, 407 x 488 px

## Page System

- A4 portrait, single column.
- Main body begins about 38 pt from the left edge and ends about 31 pt from the right edge.
- Header occupies the top 112-122 pt of page 1; body uses compact line rhythm.
- No running header, footer, page number, background, colored bands, or decorative fill.
- Page 2 uses the same margins and typography; content continues naturally.

## Typography And Rules

- Black/white system with black text and rules only.
- Candidate name: centered, bold, approximately 19-21 pt.
- Header metadata: centered 9.5-10.5 pt with small monochrome line icons.
- Section headings: left aligned, bold, approximately 12-13 pt, with a thin black bottom rule.
- Organization/project titles: bold 9.5-10.5 pt, date right aligned on the same line.
- Body: compact sans-serif Chinese type, approximately 8.3-9.2 pt, single line spacing.
- Project lists use real round bullets with hanging indentation; work descriptions use compact labeled lines.
- Education badges are pale blue rounded labels in the source. Preserve the semantic labels and approximate treatment.

## Components

- First-page header: centered name and four metadata rows; passport photo anchored at upper right.
- Section order: 教育经历 -> 工作技能 -> 工作/实习经历 -> 项目经历 -> 科研经历 -> 竞赛经历.
- All section headings use the same text-plus-rule pattern.
- Dates are aligned to the right edge; content remains in one column.

## Editable Slot Map

- Header role slot: replace `大模型训练优化工程师` with current role positioning.
- Header focus slot: replace the old multimodal-focused phrase with current Xiaopeng work focus.
- Header age slot: retain current age `29岁`; preserve other personal metadata.
- Work skills slot: rewrite around current training/inference Infra responsibilities while retaining the section and bullet form.
- Work experience slot: insert Xiaopeng before Huawei; update Huawei end date to `2025年11月`; preserve Huawei and internship wording.
- Project experience slot: insert one Xiaopeng post-training Infra project before Huawei projects; preserve existing Huawei projects.
- Education, research, and competition slots: preserve wording and order.

## Capacity And Flow

- Keep the original two-page A4 form factor.
- The source left most of page 2 blank, so added Xiaopeng content should flow there without shrinking the source typography below its existing density.
- Page 1 should still begin with education/skills and show Xiaopeng as the newest work entry.
- Page 2 may continue project, research, and competition content.

## Fidelity Gates

- Preserve the black/white visual system, top portrait header, section order, rule style, compact single-column rhythm, and right-aligned dates.
- Do not add colored sidebars, cards, timelines, a new summary section, or modern resume furniture.
- Render every final page and compare against the source for recognizable structural continuity.
