## Citation-Based Scoring & Selection (Selection Gate)

This document defines the **selection-first** workflow for turning an uploaded document/transcript into carousel topics **before** any hooks or slides are written.

The goal is to make selection:
- **Rigorous**: comparable scores using a clear rubric + weights
- **Citation-based**: every candidate idea is traceable to **verbatim** source excerpts
- **Testable**: easy to validate with deterministic checks

---

## Recommended Architecture (Keep It Simple)

### Opinionated recommendation

Use a **single orchestrator** running a **structured, staged pipeline** (multiple prompt stages) over a shared JSON state.

This is not a “true multi-agent system.” Roles like “extractor” and “scorer” can be implemented as **separate prompt templates** and (optionally) different model configs, but they should not operate autonomously or debate.

### Why this architecture fits the selection gate

- **Enforces selection-before-writing**: the orchestrator can hard-block downstream stages until `selection_frozen_at` exists.
- **Preserves source fidelity**: a single shared state makes citations first-class and prevents provenance loss.
- **Practical to build**: fewer moving parts, easy to test each stage with fixtures, easy to iterate on scoring weights.

### When to add “light agents”

Only add specialization if you see repeatable failures:
- **Extractor/scorer** benefits from stricter structured-output settings.
- **Hook/slide writing** benefits from a stronger writing model.

Keep it a pipeline either way: no free-form loops, no debates, no separate memory systems.

---

## Principles

- **Selection before writing**: do not generate hooks or slides until shortlist selection is frozen.
- **Source fidelity**: candidates must include **verbatim excerpts** and a **source location** reference for each supporting excerpt.
- **Comparable scoring**: score all candidates using the same dimensions, scale, and weights.
- **Claim safety** (health workflows): any specific health claim must be supported by the source or softened to a general statement.
- **Auditability**: a reviewer should be able to answer: “Where did this come from?” in under 10 seconds.

---

## DocMap Carousel Strategy (Performance Guidelines)

These guidelines shape extraction, scoring, selection, and writing.

### Goal

Carousels are DocMap’s **utility engine**: create **saveable, shareable** posts that help women:
- **recognise themselves**
- **understand what may be happening**
- **feel confident about what to do next**

Carousels do not need to “win reach” versus reels. They win on **clarity**, **depth**, and **intent**.

### Pillars (use consistently)

Use the same three pillars across formats:
- **recognition**: symptoms, myths, wrongly normalised experiences
- **validation**: emotional/social reality, dismissal, feeling minimised
- **access_or_decision**: NHS vs private, specialist choice, questions to ask, next steps

### Best-performing formats (prioritise)

- **checklist**
- **myth_vs_truth**
- **what_to_do_next**

### What makes a good carousel

- One clear topic + one clear takeaway
- Slide 1 hooks quickly and makes relevance obvious
- Clean, structured progression (no overload)
- Specific, practical, readable

### Metrics to optimise for

- Primary: **saves**, **shares**, **profile visits**
- Secondary: views/impressions
- Access/decision posts: watch for **DMs/qualified intent**

---

## Workflow Overview

1. **Ingest + chunk** (non-LLM): produce stable `chunks[]` with IDs and locations.
2. **Extract candidates**: create a wide set of candidate ideas, each with citations.
3. **Score + rank**: apply rubric to every candidate and compute a weighted total.
4. **Select shortlist**: pick top N and freeze selection.
5. **Only then** generate hooks and slides for the selected ideas.

---

## MVP Build Plan (Lean, High-Quality)

Build the system in this order to maximize quality early.

### MVP1: Citation-first extraction + scoring + freeze

- **A. Ingest + chunk (code)**
  - Output: `chunks[]` with stable IDs and `source_location`.
- **B. Candidate extraction (LLM)**
  - Output: `candidates[]` with `topic/angle/promise` + `source_citations[]` (verbatim excerpts).
  - Must pass “Candidate extraction checks.”
- **C. Scoring + ranking (LLM)**
  - Output: all `scores.*` filled + `total_weighted` + `rank`.
  - Must pass “Scoring checks.”
- **D. Selection freeze (code or LLM+code)**
  - Output: `shortlist.selected_idea_ids` + `selection_frozen_at` + per-idea `selection_reason`.
  - Must pass “Selection checks.”

At the end of MVP1 you can reliably answer: **document → candidate ideas → scored shortlist**.

### MVP2: Hooks (selected only)

- Generate `hooks[]` only for `shortlist.selected_idea_ids`.
- Choose `best_hook_id` and write a 7–10 slide outline per idea (still citation-aware).

### MVP3: Slides + optional single-pass QA

- Draft slides for the final selected idea(s).
- Run a single QA/edit pass that checks for:
  - uncited specific claims
  - slide length overflow
  - redundancy / weak progression

---

## Practical Model Splitting (Ideation / Hook Judge / Writer / Editor)

Use separate model “roles” only where it improves quality. Keep orchestration simple (still a pipeline).

### Role 1: Ideation (Extractor + Scorer)

**Job**
- Produce citation-grounded candidates and apply the scoring rubric consistently.

**Why split**
- These stages require **precision**, **schema adherence**, and **citation discipline** more than creative copywriting.

**Recommended behavior**
- Strict JSON output.
- Conservative wording when source support is ambiguous.
- Willing to downscore ideas that are compelling but poorly supported.

**Typical stages**
- Candidate extraction
- Scoring + ranking
- Selection freeze recommendation (with reasons)

### Role 2: Writer (Hooks + Slides)

**Job**
- Turn a selected idea into scroll-stopping hooks and clean, structured slides.

**Why split**
- Hooks and slide copy benefit from stronger writing ability and style control.

**Recommended behavior**
- High variety in hooks (multiple styles), then converge on a best option.
- Strict slide constraints (count, max chars, progression rules).
- No new “facts” beyond what can be supported by citations.

**Typical stages**
- Hook generation + hook scoring
- Outline + slide drafting

### Role 3: Editor (QA / Compliance)

**Job**
- Single-pass cleanup and safety checks; ensure claims are supportable and wording is responsible.

**Why split**
- Editing is best treated as a deterministic checklist + rewrite pass, not creative ideation.

**Recommended behavior**
- Flag uncited specifics and either add citations or soften the claim.
- Enforce length, clarity, redundancy removal, tone consistency.

---

## Practical Iteration Plan (Human-in-the-Loop Eval)

Assume you are the evaluator during early iterations. Optimize the system by testing the highest-leverage stages first.

### What’s worth testing most (in order)

1. **Candidate extraction quality** (recall + citation integrity)
2. **Scoring + selection calibration** (does top-N match your instincts?)
3. **Hook generation** (does each shortlisted idea yield at least one great hook?)
4. **Slide drafting** (structure and clarity; less leverage than selection)
5. **QA/edit pass** (consistency + safety; should not be the main quality engine)

### Minimal evaluation form (fast + repeatable)

Capture these on a 1–5 scale per stage output:
- **Extraction**: “Did it surface the real winners?” + “Are citations usable?”
- **Selection**: “Are top 3 actually draft-worthy?” + “Are the reasons sensible?”
- **Hooks**: “Is there at least 1 hook I’d post?” + “Does hook match the promise?”
- **Slides**: “Is the slide arc tight?” + “Any fluff?” + “Any risky claims?”

### Experiment knobs (start small)

- **Chunking**: chunk size, overlap, section-aware chunking
- **Candidate set**: 20 vs 40 candidates; dedup aggressiveness
- **Rubric**: weights, thresholds, tie-breakers
- **Hook styles mix**: contrarian vs how-to vs identity vs numbered
- **Slide templates**: steps vs framework vs checklist vs myth-bust

Keep experiments to **one knob at a time** so you can attribute improvements.

---

## Performance Flywheel (Using Real-World Results)

Later, you can feed post performance back into selection so the system improves over time.

### What to store per published carousel

Store both the creative artifact and its upstream lineage.

```json
{
  "post_id": "p001",
  "published_at": "iso8601",
  "platform": "instagram",
  "idea_id": "i003",
  "hook_id": "h014",
  "pillar": "sleep",
  "topic": "string",
  "rubric_version": "v1",
  "model_roles": {
    "ideation": "model_name_or_version",
    "hook_judge": "optional_model_for_hook_policy",
    "writer": "model_name_or_version",
    "editor": "model_name_or_version"
  },
  "creative": {
    "cover_text": "string",
    "slides_hash": "short_hash_or_id",
    "cta": "string"
  },
  "predicted": {
    "total_weighted": 86.2,
    "dimension_scores": {
      "hook_strength": 4,
      "clarity": 5,
      "source_support": 4
    }
  },
  "observed": {
    "impressions": 0,
    "likes": 0,
    "comments": 0,
    "shares": 0,
    "saves": 0,
    "follows": 0,
    "profile_visits": 0,
    "avg_watch_time_sec": null
  },
  "derived": {
    "save_rate": 0.0,
    "share_rate": 0.0,
    "engagement_rate": 0.0
  },
  "human_notes": "What worked / what didn’t"
}
```

### How to use performance data (simple and practical)

- **Calibrate weights**: if one dimension (e.g. hook_strength) predicts shares but hurts saves, adjust weights depending on your goal.
- **Improve selection**: learn which dimensions best predict *your* outcomes (often saves/shares).
- **Prompt tuning**: identify recurring patterns in winners (hook style, template, pillar) and bake into extraction/hook prompts.
- **Guardrails**: if a style drives clicks but increases misinformation risk, constrain it at scoring time (e.g., lower `source_support` tolerance).

### Don’t overbuild early

Start by logging performance + lineage. Only later add:
- automatic regression/weight fitting
- per-platform normalization
- cohort controls (posting time, follower count changes)

---

## Required Data Model (Minimum)

### Chunk model (created during ingest)

Each chunk must be stable and referenceable.

```json
{
  "chunk_id": "c001",
  "text": "string",
  "source_location": {
    "page": 3,
    "line_start": 120,
    "line_end": 162,
    "timestamp_start": null,
    "timestamp_end": null
  },
  "section_path": ["Heading 1", "Heading 2"],
  "char_start": 10234,
  "char_end": 11890
}
```

### Candidate idea model (produced by extraction)

Each candidate must be expressible as a carousel and must include citations.

```json
{
  "idea_id": "i001",
  "content_pillar": "sleep",
  "topic": "The 2-minute wind-down that fixes 'tired but wired'",
  "angle": "micro-habit + physiological explanation",
  "core_claim": "A short, consistent wind-down cue can reduce bedtime mental rumination.",
  "audience_pain": "I’m exhausted but my brain won’t shut off.",
  "promise": "A tiny routine you can do tonight to fall asleep easier.",
  "format_suggestion": "steps",
  "source_citations": [
    {
      "chunk_id": "c014",
      "excerpt": "verbatim excerpt from the source…",
      "note": "What this excerpt supports"
    }
  ],
  "safety_flags": ["medical_claim_risk"],
  "scores": {
    "hook_strength": null,
    "clarity": null,
    "save_share": null,
    "strategic_fit": null,
    "source_support": null,
    "total_weighted": null
  },
  "rank": null,
  "selected": false,
  "selection_reason": null
}
```

---

## Citation Rules (Non-Negotiables)

For each candidate idea:
- **At least 2 citations** (2 different excerpts) unless the source is extremely short.
- Each citation must include:
  - `chunk_id`
  - **verbatim** `excerpt` copied from the chunk
  - `note` describing what the excerpt supports
- Excerpts should be short and surgical:
  - Prefer **1–3 sentences** or a short bullet block
  - Avoid long paragraphs unless necessary

For claims in health content:
- If a claim is **specific** (quantified, causal, or medical), it must be directly supported by the cited excerpt(s).
- If not supported, the extractor/scorer must either:
  - **soften** the claim to a general, non-medical statement, or
  - mark `safety_flags` and lower `source_support`.

---

## Scoring Rubric

### Scale (0–5 for each dimension)
- **0**: unusable (unclear, not supported, not carousel-suitable)
- **1**: weak
- **2**: below average
- **3**: good
- **4**: strong
- **5**: exceptional

### Dimensions (five — keeps the judge focused)

- **Hook strength**: Would a cold viewer stop scrolling? (Relatability and urgency live here; do not score “pillar fit” twice.)
- **Clarity**: Crisp takeaway *and* whether it can become a natural multi-slide arc (problem → insight → steps → recap).
- **Save/share potential**: Actionable or reference-worthy (checklist, myth vs truth, what to do next).
- **Strategic fit**: Matches content pillars and brand voice.
- **Source support**: How strongly the document supports this idea with quotable lines.

### Recommended weights (health-content default)

Prioritise hook + citations, then fit and utility.

```json
{
  "weights": {
    "hook_strength": 1.35,
    "source_support": 1.35,
    "strategic_fit": 1.1,
    "save_share": 1.1,
    "clarity": 1.1
  }
}
```

### Weighted total score (0–100)

\[
total=\frac{\sum(score_{0..5}\times weight)}{\sum(weights)}\times 100
\]

Store `scores.total_weighted` as an integer or 1-decimal float.

### Selection thresholds (suggested)

- **80–100**: shortlist priority
- **70–79**: shortlist backup
- **<70**: do not draft (unless you need volume)

---

## Scoring Guidance (Quick Anchors)

Use these anchors to keep scoring consistent.

### Source support
- **5**: multiple quotable lines directly supporting the claim and steps
- **3**: supported generally but missing specifics; may require careful wording
- **1**: mostly inference; easy to hallucinate details

### Clarity (includes slide arc)
- **5**: one-sentence takeaway is crisp and concrete, and a natural 7–10 slide progression is obvious
- **3**: understandable but jargon, multi-part, or needs shaping to avoid repetition
- **1**: muddy, or basically a single thought that would repeat across slides

---

## Selection Gate (Freeze Before Writing)

The system must explicitly freeze the shortlist selection.

Minimum freeze record:

```json
{
  "shortlist": {
    "selected_idea_ids": ["i003", "i009", "i014"],
    "selection_frozen_at": "2026-04-15T12:34:56Z",
    "rubric_version": "v1",
    "notes": "Selected based on weighted score + source support; excluded medical-claim-heavy candidates."
  }
}
```

After `selection_frozen_at` is set:
- Hooks/slides may be generated **only** for `selected_idea_ids`.
- Any changes require creating a **new freeze record** (or bumping `rubric_version`) to keep audit history clean.

---

## Deterministic Validation Checks

Run these checks after extraction and again after scoring.

### Candidate extraction checks
- Every candidate has `idea_id`, `topic`, `angle`, and `content_pillar`.
- `source_citations.length >= 2` (configurable).
- Each citation has valid `chunk_id` that exists in `chunks[]`.
- Each `excerpt` is **present verbatim** inside the referenced chunk text.

### Scoring checks
- All required score dimensions are populated (0–5).
- Weighted total is computed and stored.
- Candidates are ranked by total score; ties are broken by `source_support`, then `hook_strength`, then lexicographic `content_pillar`, then `idea_id` (stable ordering).

### Selection checks
- `selected=true` only for IDs in `shortlist.selected_idea_ids`.
- `selection_reason` is present for selected candidates (1–3 sentences).

---

## What to Avoid (Common Failure Modes)

- Generating hooks/slides before the shortlist is frozen.
- Letting candidates exist without citations (guarantees hallucinations later).
- Using long, vague excerpts that don’t clearly support the claim.
- Scoring without clear anchors (scores become random and non-actionable).
- Over-weighting flashy hooks over `source_support` in health workflows (increases misinformation risk).

---

## Practical Defaults (If You Need Them)

- **Candidate count**: 20–40 extracted candidates per document.
- **Shortlist size**: 3–7 selected ideas.
- **Slide count**: 7–10 slides per carousel.
- **Tone**: clear, grounded, non-alarmist; avoid absolutes unless the source explicitly supports them.

