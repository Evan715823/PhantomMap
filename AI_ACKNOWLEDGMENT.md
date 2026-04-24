# AI assistance acknowledgment

Per the COMP 646 syllabus ("Any code you submit is assumed to be yours
unless indicated otherwise. Code from AI assistants counts as your
code. [...] Acknowledge AI assistance at the end of the report."), we
document here exactly where and how we used AI.

## Assistants used

- **Claude Opus 4.7 (Anthropic)** — primary assistant throughout the
  project. Accessed via Claude Code.

## What AI was used for

### Project scoping
- Brainstormed candidate project topics with the constraint *vision +
  language, modest compute, original*. Claude suggested roughly ten
  candidate angles; we read recent VLM-hallucination literature
  ourselves and chose **PhantomMap** after rejecting several candidates
  whose idea overlapped directly with already-published work (e.g.
  visual-prior-vs-pixels, which is already done by
  [arXiv:2505.17127](https://arxiv.org/abs/2505.17127)).

### Code scaffolding
- Claude drafted the initial skeletons of `src/prompts.py`,
  `src/parse_bbox.py`, `src/run_vlm.py`, `src/features.py`,
  `src/detector.py`, `src/atlas.py`, `src/make_figures.py`, and
  `src/hits_cross_ref.py`. We reviewed each file, fixed edge cases
  specific to Qwen2.5-VL's attention-layer indexing, validated the bbox
  parser against real model outputs, and made every final choice
  (feature set, detector baseline, figure layout, KDE bandwidth).

### Report drafting
- The first pass of the LaTeX report was assembled from our outline
  with Claude's help. Every paragraph was edited by us; the
  differentiator paragraph, the ethics statement, and the final
  analysis prose are our own writing.

### Related-work literature scan
- We used Claude's `WebSearch` tool to find adjacent papers (EAZY,
  Woodpecker, HalLoc, OPERA, HALP, DASH) and verify the novelty gap.
  For each cited paper we read the abstract and at least the method
  section ourselves.

## What AI was **not** used for

- Inventing experimental results. Every number in the report comes
  from a real run; any cell we haven't produced is marked `\todo{...}`
  in red.
- Deciding novelty. We independently verified via Google Scholar /
  arXiv that no direct prior work publishes a spatial atlas of
  VLM-self-reported phantom bounding boxes.
- Authoring citations. `egbib.bib` contains only entries we looked up
  and verified in official venue proceedings / arXiv listings.

## Standing by our submission

Every word, figure, and number in the submitted report is our
responsibility. Where AI wrote the initial draft of a sentence or
function, we read it, understood it, and accept accountability for it.

---

*Authors: Author 1 (user), Author 2 (teammate). Updated 2026-04-23.*
