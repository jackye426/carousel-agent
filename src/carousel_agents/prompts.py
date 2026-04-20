from __future__ import annotations

from typing import Any

from .performance_digest import digest_prompt_body, digest_writer_hints


DOCMAP_CONTEXT_SUMMARY = (
    "DocMap context (use for framing):\n"
    "- DocMap helps UK women (starting with endometriosis journeys) make sense of symptoms, prepare for appointments, and navigate next steps.\n"
    "- **Services (two parts — be accurate in hooks/CTAs):**\n"
    "  1) **Symptom note for your GP** — we help you structure a clear symptoms note you can take to your GP. The goal is to help you move toward a **diagnosis or a concrete next step** in the NHS pathway (not to replace clinical judgement).\n"
    "  2) **Private specialist matching** — when a next step involves specialist care, we can help you **find a private specialist** who fits that next step (no hype; no guarantees).\n"
    "- We reduce confusion/dismissal by helping patients organise their story (often via WhatsApp) into a clear summary + questions for GP/specialist.\n"
    "- We are NOT a diagnostic tool, not medical advice, not emergency care, and we do not promise outcomes.\n"
    "- Tone: clear, calm, compassionate, validating, practical, credible; avoid fear-based/salesy generic wellness.\n\n"
)

HOOK_BRIEF_SUMMARY = (
    "Hook brief (DocMap):\n"
    "- Navigation + appointment prep (not generic education). Hooks close a gap DocMap fills: urgent, specific, solvable friction.\n"
    "- Prefer concrete moments over abstract validation; voice: direct, calm/credible, human.\n"
    "- Modes that work: granular failure; ‘a better path exists’; before/after conversation beat leaflet tone.\n\n"
)

HOOK_LANGUAGE_POLICY = (
    """Hook language policy (hard rules):
- BANNED PHRASES (do not use them anywhere):
  - "Here's what no one tells you"
  - "changes everything"
  - "you're not alone"
  - "every second count"
  - "no more endless GP ping-pong" (avoid tidy alliteration; use real patient language)
  - "here's why" as a reflex construction
- Avoid caps emphasis like "YOU NEED TO".
- If you feel tempted to explain, drop the reader into a moment instead.
- Replace abstraction with one concrete, uncomfortable, countable detail.

"""
)

DOCMAP_CTA_VOICE = (
    "CTA voice (DocMap — natural, not brochure):\n"
    "- Position DocMap as the solution when it fits, but **do not** sound like a hard sell or a repeated brand tagline.\n"
    "- **Pick one focus per CTA (never both in the same line):** either **structured symptom note → GP** (diagnosis / next step on the pathway) **or** **help finding a private specialist** for a specialist next step—whichever matches this carousel. Do not combine both services in a single CTA.\n"
    "- Prefer **second person** (you / your): speak to the reader directly. **We** is fine for a warm, human invite (e.g. organising your story on WhatsApp).\n"
    "- Avoid defaulting to **third-person** brand boilerplate every time "
    '(e.g. "DocMap helps women…", "DocMap provides users with…"). If you name DocMap, keep it light; lead with the outcome they want.\n'
    "- Vary the shape across posts: soft invite, one clear next step, or a short question — not the same CTA template every time.\n"
    "- Stay product-true: one concrete next step (GP prep **or** private specialist path), WhatsApp summary when natural, navigation — no diagnosis or promised outcomes.\n\n"
)


def _persona_preset_block(idea: dict[str, Any] | None) -> str:
    """Compact block from `idea['persona_preset']` (set by persona card loader)."""
    if not idea or not isinstance(idea, dict):
        return ""
    err = idea.get("persona_preset_error")
    if isinstance(err, str) and err.strip():
        return f"Persona preset note: card not loaded ({err}). Proceed with default audience only.\n\n"
    pp = idea.get("persona_preset")
    if not isinstance(pp, dict) or not pp.get("id"):
        return ""
    lines = [
        "Primary persona (tone and emphasis only; facts must still match citations):",
        f"- id: {pp.get('id')}",
        f"- name: {pp.get('name', '')}",
        f"- summary: {pp.get('summary', '')}",
        f"- primary_fear: {pp.get('primary_fear', '')}",
        f"- primary_hope: {pp.get('primary_hope', '')}",
        f"- content_job: {pp.get('content_job', '')}",
    ]
    lex = pp.get("lexicon") or []
    if isinstance(lex, list) and lex:
        lines.append("- lexicon (cadence only; do NOT quote or copy lines):")
        for x in lex[:12]:
            lines.append(f"  - {str(x).strip()}")
    dos = pp.get("dos") or []
    if isinstance(dos, list) and dos:
        lines.append("- do:")
        for x in dos[:5]:
            lines.append(f"  - {str(x).strip()}")
    donts = pp.get("donts") or []
    if isinstance(donts, list) and donts:
        lines.append("- don't:")
        for x in donts[:5]:
            lines.append(f"  - {str(x).strip()}")
    return "\n".join(lines) + "\n\n"


def _editorial_direction_block(idea: dict[str, Any] | None) -> str:
    if not idea or not isinstance(idea, dict):
        return ""
    d = idea.get("editorial_direction")
    if not d or not str(d).strip():
        return ""
    return (
        "Campaign / editorial direction (honour alongside audience; do not contradict source evidence):\n"
        f"{str(d).strip()}\n\n"
    )


def _reviewer_clarification_block(idea: dict[str, Any] | None) -> str:
    if not idea or not isinstance(idea, dict):
        return ""
    parts: list[str] = []
    r = idea.get("reviewer_brief")
    if r and str(r).strip():
        parts.append(
            "Reviewer notes (honour when consistent with source evidence):\n" + str(r).strip(),
        )
    c = idea.get("writer_clarification_context")
    if c and str(c).strip():
        parts.append("Clarification Q&A with reviewer:\n" + str(c).strip())
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"


def _reader_benefit_block(idea: dict[str, Any] | None) -> str:
    if not idea or not isinstance(idea, dict):
        return ""
    rb = idea.get("reader_benefit")
    if not rb or not str(rb).strip():
        return ""
    return (
        "Patient-facing benefit (from ideation; honour when consistent with source evidence):\n"
        f"{str(rb).strip()}\n\n"
    )


def _writer_context_blocks(idea: dict[str, Any] | None) -> str:
    return (
        _editorial_direction_block(idea)
        + _reader_benefit_block(idea)
        + _reviewer_clarification_block(idea)
    )


def user_writer_preflight_clarification(
    *,
    selected_ideas: list[dict[str, Any]],
    audience: dict[str, Any] | None,
    editorial_direction: str | None,
    reviewer_notes_global: str | None,
    prior_qa: str | None = None,
) -> str:
    """Prompt Writer model to optionally ask clarifying questions before hooks/slides."""
    aud = _audience_block(audience)
    ed = (editorial_direction or "").strip()
    rg = (reviewer_notes_global or "").strip()
    pq = (prior_qa or "").strip()
    ideas_blob = "\n\n".join(
        [
            f"- **{x.get('idea_id')}** ({x.get('content_pillar')}): {x.get('topic')}\n"
            f"  angle: {x.get('angle')}\n"
            f"  reader_benefit: {(str(x.get('reader_benefit') or '').strip()) or '_(none)_'}"
            for x in selected_ideas
        ]
    )
    prior = ""
    if pq:
        prior = (
            "You already asked questions; the reviewer answered:\n"
            f"{pq}\n\n"
            "If that is sufficient, set need_clarification=false and questions=[].\n\n"
        )
    return "".join(
        [
            "You are the Writer for DocMap carousel copy. The human selected ideas and may have added notes.\n"
            "Before generating hooks, decide if you need **at most 3** short clarification questions.\n\n",
            aud,
            f"Editorial / campaign direction (if any):\n{ed or '_(none)_'}\n\n",
            f"Global reviewer notes:\n{rg or '_(none)_'}\n\n",
            "Selected ideas:\n",
            f"{ideas_blob}\n\n",
            prior,
            "Return JSON:\n",
            '{ "need_clarification": true|false, "questions": ["..."], "assistant_message": "short friendly summary" }\n'
            "- If nothing is ambiguous, need_clarification=false and questions=[].\n"
            "- Questions must be specific and answerable in one reply.\n",
        ]
    )


def _writer_source_evidence_block(source_evidence: str | None) -> str:
    """Verbatim chunk text for writer-stage grounding (optional)."""
    if not (source_evidence or "").strip():
        return ""
    return (
        "SOURCE TEXT (verbatim from the ingested document chunks below; use for factual grounding):\n"
        "- Hooks/slides must not promise or imply specific claims that are not supported here.\n"
        "- Patient-voice / OCR references elsewhere are style only; this block is the evidence layer.\n\n"
        f"{source_evidence.strip()}\n\n"
    )


def _audience_block(audience: dict[str, Any] | None) -> str:
    if not audience:
        return ""
    locale = audience.get("locale") or "en-GB"
    region = audience.get("region") or "UK"
    variant = audience.get("english_variant") or "UK"
    persona = audience.get("target_persona") or "UK women with suspected/diagnosed endometriosis"
    pillars = audience.get("pillars_allowed") or ["recognition", "validation", "access_or_decision"]
    return (
        "Audience requirements:\n"
        f"- Write in {variant} English (locale {locale}).\n"
        f"- Use {region}-specific wording and references where relevant.\n"
        f"- Target persona: {persona}.\n"
        f"- Content pillars must be one of: {pillars}.\n"
        "- Avoid US-only references (e.g., US insurance/brands/spellings) unless the source explicitly mentions them.\n\n"
        + DOCMAP_CONTEXT_SUMMARY
    )


def system_ideation() -> str:
    return (
        "You are the Ideation model. You extract candidate carousel ideas and score them rigorously.\n"
        "Rules:\n"
        "- Output MUST be valid JSON only.\n"
        "- Preserve source fidelity. Use verbatim excerpts for citations.\n"
        "- If source support is weak, say so and score it lower.\n"
        "- Do not generate hooks or slides unless explicitly asked.\n"
    )


def system_ideation_scoring() -> str:
    return (
        "You are the Ideation scoring model. Another model already proposed candidate ideas; you did not write them.\n"
        "Your job is to score each candidate independently using the rubric — avoid favouring phrasing you might have "
        "written yourself.\n"
        "Rules:\n"
        "- Output MUST be valid JSON only.\n"
        "- Be strict on source_support and strategic_fit.\n"
        "- Do not generate hooks or slides unless explicitly asked.\n"
    )


def system_hook_judge() -> str:
    return (
        "You are the Hook Judge. Another model (the Writer) already drafted hooks; your job is to evaluate them.\n"
        "You do not write marketing copy yourself here — you decide whether hooks meet policy and quality, "
        "or must be revised/vetoed.\n"
        "Rules:\n"
        "- Output MUST be valid JSON only.\n"
        "- Be strict about banned templated phrasing and subheading voice.\n"
        "- Prefer concrete, human, DocMap-aligned fixes in directives when you ask for a rewrite.\n"
    )


def system_writer() -> str:
    return (
        "You are the Writer. You write high-performing hooks and carousel slides from a selected idea.\n"
        "Rules:\n"
        "- Output MUST be valid JSON only.\n"
        "- Do not invent facts. Stay consistent with provided citations.\n"
        "- Keep slides tight and scannable.\n"
    )


def system_editor() -> str:
    return (
        "You are the Editor. You do a single-pass QA and polish.\n"
        "Rules:\n"
        "- Output MUST be valid JSON only.\n"
        "- Flag uncited specific claims and either soften or propose a fix.\n"
        "- Enforce length/clarity and remove fluff.\n"
    )


def user_extract_candidates(
    *,
    chunks: list[dict[str, Any]],
    candidate_count: int,
    audience: dict[str, Any] | None = None,
    performance_digest: dict[str, Any] | None = None,
) -> str:
    perf_block = ""
    if performance_digest:
        perf_block = (
            "Past carousel performance (weak priors only):\n"
            "- Prefer angles, pillars, and format_suggestion values that match patterns below **when** the chunks still "
            "provide strong verbatim citations.\n"
            "- Do NOT force-fit: if the document does not support a 'winning' shape, prioritise source fidelity.\n"
            "- Do NOT copy phrasing from cover_text or metrics into citations.\n\n"
            f"{digest_prompt_body(performance_digest)}\n\n"
        )
    return "".join(
        [
            "Extract carousel-worthy candidate ideas from these chunks.\n\n",
            _audience_block(audience),
            perf_block,
            f"Target candidate count: {candidate_count}\n",
            "For each idea:\n",
            "- Provide: content_pillar, topic, angle, core_claim, audience_pain, promise, format_suggestion\n",
            "- Provide `reader_benefit`: 1–3 sentences for the **reader** (patient/carer): why they should save/share this — "
            "the concrete payoff (clarity, validation, or next step), not a restatement of what a clinician/service *does*. "
            "Ground it in the citations; if the doc only implies the benefit, say so plainly without inventing outcomes.\n",
            "- content_pillar MUST be exactly one of: recognition | validation | access_or_decision\n",
            "- Prefer these formats: checklist | myth_vs_truth | what_to_do_next\n",
            "- Provide at least 2 source_citations with chunk_id + verbatim excerpt + note\n",
            "- Excerpts must be copied EXACTLY from the chunk text: no ellipses (\"...\"), no paraphrasing, no bracketed inserts\n",
            "- Add safety_flags if claims risk being too medical/specific\n\n",
            "Also provide `editorial_direction`: 2–5 sentences on how you would position this batch of ideas for the audience "
            "(themes, tone, what to avoid). Guidance only — every idea must remain grounded in the chunks.\n\n",
            "Return JSON with shape:\n",
            '{ "candidates": [ { ..., "reader_benefit": "..." } ], "editorial_direction": "..." }\n\n',
            "Chunks:\n",
            f"{chunks}",
        ]
    )


def user_score_candidates(
    *,
    candidates: list[dict[str, Any]],
    weights: dict[str, float],
    audience: dict[str, Any] | None = None,
    performance_digest: dict[str, Any] | None = None,
) -> str:
    perf_block = ""
    if performance_digest:
        perf_block = (
            "Performance context (weak priors; scoring tie-breakers only):\n"
            "- This is based on our past carousel performance and baseline engagement.\n"
            "- Use it to break ties and to prefer proven shapes when scores are close.\n"
            "- Do NOT overfit: if an idea is straightforward but strongly supported + saveable, it can still score highly.\n"
            "- Do NOT copy text from examples.\n\n"
            f"{digest_prompt_body(performance_digest)}\n\n"
        )
    return "".join(
        [
            "Score each candidate idea using a 0–5 scale for each dimension (five dimensions only):\n",
            "hook_strength, clarity, save_share, strategic_fit, source_support.\n\n",
            _audience_block(audience),
            perf_block,
            "Use these weights (for context):\n",
            f"{weights}\n\n",
            "Scoring anchors (use these to stay consistent):\n"
            "- hook_strength: 5=scroll-stopper in 1 line (relatable + urgent), 3=decent, 1=bland/unclear\n"
            "- clarity: 5=crisp takeaway + natural multi-slide arc (problem→insight→steps→recap); "
            "use `reader_benefit` when present — reward ideas where the patient-facing payoff is obvious, not just a literal summary of the doc; "
            "3=mostly clear but wordy or needs shaping; 1=muddy or one-note/repetitive\n"
            "- save_share: 5=highly saveable utility (checklist/myth-bust/what-to-do-next) **for the reader**; "
            "penalise ideas that only describe services/roles without a clear reader benefit line; "
            "3=some utility; 1=mostly 'interesting'\n"
            "- strategic_fit: 5=matches DocMap pillars + UK audience + brand voice, 3=okay fit, 1=off-strategy\n"
            "- source_support: 5=multiple quotable lines directly support the core claim + structure, "
            "3=generally supported but missing specifics, 1=inference/paraphrase risk\n\n",
            "Rules:\n",
            "- Be strict on source_support.\n",
            "- If citations do not clearly support the claim, reduce source_support and clarity.\n",
            "- Optimise for carousels as a utility engine: saves/shares and clarity beat reach.\n",
            "- strategic_fit should strongly reflect DocMap content pillars (recognition/validation/access_or_decision).\n",
            "- Scores must be INTEGERS 0–5 (no decimals).\n",
            "- Do NOT compute weighted totals; only return per-dimension integer scores.\n\n",
            "Return JSON with shape:\n",
            '{ "scores": [ { "idea_id": "i001", "hook_strength": 3, ... } ] }\n\n',
            "Candidates:\n",
            f"{candidates}",
        ]
    )


def user_select_shortlist(
    *,
    ranked: list[dict[str, Any]],
    select_n: int,
    audience: dict[str, Any] | None = None,
    pool_k: int | None = None,
) -> str:
    pool_note = ""
    if pool_k is not None and pool_k > 0:
        pool_note = (
            f"HARD CONSTRAINT: The list below is ONLY the top {pool_k} candidates by weighted score (rank 1 = best). "
            f"You MUST choose every selected_idea_id from this list. Do not pick any idea_id not shown below. "
            f"If you are unsure, prefer higher-ranked (lower rank number) ideas after applying the criteria.\n\n"
        )
    return "".join(
        [
            "Select a shortlist of ideas to draft, selection-first.\n\n",
            _audience_block(audience),
            pool_note,
            f"Select exactly {select_n} idea_ids.\n",
            "Selection criteria:\n"
            "- Prioritise saveable/shareable utility (checklists, myth-vs-truth, what-to-do-next).\n"
            "- Prefer ideas that help women recognise themselves, feel validated, or make a next-step decision.\n"
            "- Avoid overly broad topics; choose one clear topic + takeaway.\n\n",
            "Return JSON with shape:\n",
            '{ "selected_idea_ids": ["i003", ...], "notes": "short reason" , "reasons": [ { "idea_id": "i003", "selection_reason": "..." } ] }\n\n',
            "Ranked candidates (already scored):\n",
            f"{ranked}",
        ]
    )


def user_review_shortlist(
    *,
    selected: list[dict[str, Any]],
    audience: dict[str, Any] | None = None,
    performance_digest: dict[str, Any] | None = None,
) -> str:
    perf_block = ""
    if performance_digest:
        perf_block = (
            "Performance alignment (weak priors for veto/approve — DocMap strategy still wins):\n"
            "- Favour ideas whose pillar/format resemble top-performing patterns below when citations support them.\n"
            "- Veto ideas that look off-strategy *and* weakly supported, even if they mimic past hooks.\n\n"
            f"{digest_prompt_body(performance_digest)}\n\n"
        )
    return "".join(
        [
            "Review the selected shortlist (Ideation pass). You are allowed to VETO ideas.\n\n",
            _audience_block(audience),
            perf_block,
            "Decision policy:\n"
            "- Approve only ideas that are tightly aligned to DocMap (clarity, prep, navigation, next steps).\n"
            "- Veto ideas that are too generic, too medical-advice-y, or not clearly useful/saveable.\n"
            "- If you veto, give a crisp reason and (optionally) directives for what a replacement idea should be.\n\n"
            "Return JSON with shape:\n"
            '{ "reviewed": [ { "idea_id": "i003", "decision": { "status": "approve|revise|veto", "reason": "...", "directives": ["..."] } } ], "notes": "..." }\n\n'
            "Selected ideas:\n"
            f"{selected}",
        ]
    )


def user_generate_hooks(
    *,
    idea: dict[str, Any],
    hook_count: int,
    audience: dict[str, Any] | None = None,
    performance_digest: dict[str, Any] | None = None,
    source_evidence: str | None = None,
) -> str:
    # Optional style references (non-factual). Callers may append these fields onto `idea` or pass via audience.constraints.
    # To keep backwards compatibility, we read them from idea if present.
    pv = (idea.get("patient_voice_snippets") or []) if isinstance(idea, dict) else []
    pv_block = "\n".join([f"- {str(s).strip()}" for s in pv if str(s).strip()]) if pv else ""
    if pv_block:
        pv_block = (
            "Patient voice references (cadence/word choice only; DO NOT quote or closely paraphrase any single line):\n"
            f"{pv_block}\n\n"
        )

    ocr = (idea.get("ocr_structure_examples") or []) if isinstance(idea, dict) else []
    ocr_block = "\n".join([f"- {str(s).strip()}" for s in ocr if str(s).strip()]) if ocr else ""
    if ocr_block:
        ocr_block = (
            "OCR carousel structure references (structure/cadence only; DO NOT copy):\n"
            f"{ocr_block}\n\n"
        )

    perf_w = ""
    if performance_digest:
        perf_w = (
            "Performance-informed copy hints (north stars + what resonated; do NOT lift cover lines):\n"
            f"{digest_writer_hints(performance_digest)}\n\n"
        )

    return "".join(
        [
            "Generate hook options for this selected idea.\n\n",
            _writer_source_evidence_block(source_evidence),
            _audience_block(audience),
            _writer_context_blocks(idea if isinstance(idea, dict) else None),
            _persona_preset_block(idea if isinstance(idea, dict) else None),
            perf_w,
            HOOK_BRIEF_SUMMARY,
            HOOK_LANGUAGE_POLICY,
            "Anti-leakage rule (hard):\n"
            "- Use the references only to learn *how patients talk* and *how carousels are paced*.\n"
            "- Do NOT quote or closely paraphrase any reference line.\n"
            "- Blend multiple influences and write fresh language.\n\n",
            pv_block,
            ocr_block,
            "Hook style tags:\n"
            "- Set `style` to EXACTLY ONE of: contrarian | numbered | myth_bust | how_to | identity | validation | checklist | what_to_do_next | quote_led | statistic\n\n",
            f"Generate {hook_count} hook options with varied styles (contrarian, how-to, identity, numbered, myth-bust).\n",
            "Make relevance obvious fast. Hooks should be endometriosis-specific where appropriate.\n",
            "Then pick the best hook and provide a 7–10 slide outline.\n\n",
            "Return JSON with shape:\n",
            '{ "hooks": [ { "hook_id": "h001", "text": "...", "style": "...", "rationale": "..." } ],'
            '  "best_hook_id": "h001", "outline": [ { "slide": 1, "purpose": "cover", "key_point": "...", "citations": ["c001"] } ] }\n\n',
            "Selected idea (with citations):\n",
            f"{idea}",
        ]
    )


def user_rewrite_hooks_from_policy(
    *,
    idea: dict[str, Any],
    hook_count: int,
    previous_hooks: list[dict[str, Any]],
    violations: dict[str, list[str]],
    judge_directives: list[str],
    ocr_style_snippets: list[str],
    audience: dict[str, Any] | None = None,
    performance_digest: dict[str, Any] | None = None,
    source_evidence: str | None = None,
) -> str:
    snippets = "\n\n".join([f"- {s}" for s in ocr_style_snippets]) if ocr_style_snippets else "_(none)_"
    perf_w = ""
    if performance_digest:
        perf_w = (
            "Performance-informed copy hints (keep compliance; texture only):\n"
            f"{digest_writer_hints(performance_digest)}\n\n"
        )
    return "".join(
        [
            "Rewrite hook options to comply with the hook language policy.\n\n",
            _writer_source_evidence_block(source_evidence),
            _audience_block(audience),
            _persona_preset_block(idea if isinstance(idea, dict) else None),
            perf_w,
            HOOK_BRIEF_SUMMARY,
            HOOK_LANGUAGE_POLICY,
            "Anti-leakage rule (hard):\n"
            "- OCR style references are cadence/texture only.\n"
            "- Do NOT copy lines; write fresh language.\n\n",
            "Hook Judge directives (follow exactly):\n",
            "\n".join([f"- {d}" for d in judge_directives]) + "\n\n" if judge_directives else "- (none)\n\n",
            "Banned-phrase violations detected in previous hooks:\n",
            f"{violations}\n\n",
            "OCR style references (cadence/texture only; DO NOT copy facts):\n",
            f"{snippets}\n\n",
            f"Return {hook_count} NEW hook options (do not reuse the banned phrasing). Then pick best_hook_id.\n\n",
            "Return JSON with shape:\n",
            '{ "hooks": [ { "hook_id": "h001", "text": "...", "style": "...", "rationale": "..." } ], "best_hook_id": "h001" }\n\n',
            "Selected idea (with citations):\n",
            f"{idea}\n\n",
            "Previous hooks:\n",
            f"{previous_hooks}",
        ]
    )


def user_generate_ctas(
    *,
    idea: dict[str, Any],
    cta_count: int = 5,
    audience: dict[str, Any] | None = None,
    performance_digest: dict[str, Any] | None = None,
    source_evidence: str | None = None,
) -> str:
    """
    CTA-only generation (no slides). Keep it product-true: navigation + appointment prep.
    """
    pv = (idea.get("patient_voice_snippets") or []) if isinstance(idea, dict) else []
    pv_block = "\n".join([f"- {str(s).strip()}" for s in pv if str(s).strip()]) if pv else ""
    if pv_block:
        pv_block = (
            "Patient voice references (cadence/word choice only; DO NOT quote or closely paraphrase any single line):\n"
            f"{pv_block}\n\n"
        )

    perf_w = ""
    if performance_digest:
        perf_w = (
            "Performance-informed hints (utility/saves mindset; not vanity metrics):\n"
            f"{digest_writer_hints(performance_digest)}\n\n"
        )

    return "".join(
        [
            "Generate CTA options for this selected idea.\n\n",
            _writer_source_evidence_block(source_evidence),
            _audience_block(audience),
            _writer_context_blocks(idea if isinstance(idea, dict) else None),
            _persona_preset_block(idea if isinstance(idea, dict) else None),
            perf_w,
            DOCMAP_CONTEXT_SUMMARY,
            DOCMAP_CTA_VOICE,
            "CTA rules:\n"
            "- Do NOT diagnose or promise outcomes.\n"
            "- Avoid salesy language; be practical.\n"
            "- Pick **one** clear DocMap action that fits this idea—either symptom note for GP (diagnosis/next step) **or** help finding a private specialist—not both. You can still use save, track, prepare, questions, WhatsApp summary as the *how*.\n"
            "- CTA should match the pillar and format.\n\n",
            pv_block,
            f"Generate {cta_count} CTA options. Then pick the best.\n\n",
            "Return JSON with shape:\n",
            '{ "cta_options": ["..."], "best_cta": "..." }\n\n',
            "Selected idea:\n",
            f"{idea}",
        ]
    )


def user_generate_caption(
    *,
    idea: dict[str, Any],
    cta: str,
    audience: dict[str, Any] | None = None,
    performance_digest: dict[str, Any] | None = None,
) -> str:
    """
    IG caption draft. Use CTA as the backbone, and mimic formatting patterns from past captions.
    """
    perf = ""
    if performance_digest:
        perf = (
            "Caption formatting references (structure/line breaks/emoji/hashtag density only — do NOT copy wording):\n"
            f"{digest_writer_hints(performance_digest)}\n\n"
        )
    return "".join(
        [
            "Write an Instagram caption for this carousel.\n\n",
            _audience_block(audience),
            _writer_context_blocks(idea if isinstance(idea, dict) else None),
            perf,
            "Inputs:\n",
            f"- Selected CTA (use as the spine; keep product-true): {cta.strip()}\n\n",
            "Caption rules:\n"
            "- Keep it scannable: short paras + line breaks.\n"
            "- Start with 1 strong opening line that matches the hook.\n"
            "- Include the CTA once, near the end.\n"
            "- Keep claims aligned with the idea + citations; avoid medical promises.\n"
            "- Match our existing caption formatting style (as shown in references): same kind of spacing, bullets, emoji use, and hashtag density.\n"
            "- Hashtags: include a small set (0–6), relevant and non-spammy.\n\n",
            "Return JSON with EXACT shape:\n",
            '{ "caption": "..." }\n\n',
            "Idea:\n",
            f"{idea}\n",
        ]
    )


def user_draft_slides(
    *,
    idea: dict[str, Any],
    max_slides: int,
    audience: dict[str, Any] | None = None,
    performance_digest: dict[str, Any] | None = None,
    source_evidence: str | None = None,
) -> str:
    pv = (idea.get("patient_voice_snippets") or []) if isinstance(idea, dict) else []
    pv_block = "\n".join([f"- {str(s).strip()}" for s in pv if str(s).strip()]) if pv else ""
    if pv_block:
        pv_block = (
            "Patient voice references (cadence/word choice only; DO NOT quote or closely paraphrase any single line):\n"
            f"{pv_block}\n\n"
        )

    ocr = (idea.get("ocr_structure_examples") or []) if isinstance(idea, dict) else []
    ocr_block = "\n".join([f"- {str(s).strip()}" for s in ocr if str(s).strip()]) if ocr else ""
    if ocr_block:
        ocr_block = (
            "OCR carousel structure references (structure/cadence only; DO NOT copy):\n"
            f"{ocr_block}\n\n"
        )

    perf_w = ""
    if performance_digest:
        perf_w = (
            "Performance-informed hints (pacing/utility; do NOT copy cover lines):\n"
            f"{digest_writer_hints(performance_digest)}\n\n"
        )

    ab_packaging = ""
    if isinstance(idea, dict) and idea.get("ab_variant") == "B":
        ab_packaging = (
            "Packaging arm **B** (A/B test): on **slide 1 only**, use a clearly different cover headline angle "
            "than arm A would use for the same idea — same factual core and CTA intent; body slides (2+) stay aligned "
            "with one coherent narrative.\n\n"
        )

    return "".join(
        [
            "Draft a carousel from the selected idea.\n\n",
            ab_packaging,
            _writer_source_evidence_block(source_evidence),
            _audience_block(audience),
            _writer_context_blocks(idea if isinstance(idea, dict) else None),
            _persona_preset_block(idea if isinstance(idea, dict) else None),
            perf_w,
            "Anti-leakage rule (hard):\n"
            "- Patient-voice / OCR references below: learn *how patients talk* and *how carousels are paced* only.\n"
            "- Do NOT quote or closely paraphrase those reference lines.\n"
            "- Blend multiple influences and write fresh language.\n"
            "- SOURCE TEXT above (if present) is factual grounding, not style reference.\n\n",
            pv_block,
            ocr_block,
            f"Constraints: choose the RIGHT length for the idea (you do NOT need to use {max_slides}).\n"
            f"- Target 5–{max_slides} slides.\n"
            "- Prefer fewer slides if you can't add *new* value on each one.\n"
            "- Each slide MUST include main_text and subtext.\n\n"
            "Quality bar (critical): depth + polish in few words.\n"
            "- Every slide must earn its place: add a distinct, specific insight or actionable next step.\n"
            "- No fluff: avoid vague reassurance, filler transitions, or repeating the hook.\n"
            "- Unless the pillar is validation, DO NOT include generic validation slides (e.g., 'you're not alone', 'your pain is valid', 'awareness is power').\n"
            "- Be specific without being long: compress ideas into tight language.\n"
            "- Write like the 'Silent Endo Symptoms' reference: short headline + dense, meaningful subtext.\n\n"
            "Carousel guidelines:\n"
            "- One clear topic + one clear takeaway.\n"
            "- Slide 1 hooks quickly and makes relevance obvious.\n"
            "- Use clean, structured formats (checklist, myth_vs_truth, what_to_do_next).\n"
            "- Prefer concrete wording (what it is / why it matters / what to do next).\n"
            "- Aim for 'I need to save this' utility.\n\n"
            "Tightness constraints (hard):\n"
            "- main_text: ≤ 10 words (headline-style)\n"
            "- subtext: 1–2 short sentences (≤ ~35 words total)\n"
            "- subtext must contain at least one concrete element: a mechanism, example, or explicit next-step.\n"
            "- Avoid long lists inside subtext; if you need a list, split it into multiple slides.\n\n",
            "Add citations (chunk_ids) on slides that contain specific claims.\n",
            DOCMAP_CTA_VOICE,
            "Include one closing CTA line and (if relevant) a short disclaimer.\n"
            "- Put disclaimer text **only** in the `disclaimer` JSON field (after the CTA in the schema)—do **not** add a separate slide whose only job is disclaimer/legal text.\n"
            "- The `cta` field should read like the natural end of the carousel (you/your; optional we), not a third-person brand footer every time.\n"
            "- **One** DocMap angle in that line: GP symptom-note prep **or** private specialist help—never both in the same CTA.\n\n",
            "Return JSON with shape:\n",
            '{ "carousel_draft": { "slides": [ { "slide_number": 1, "main_text": "...", "subtext": "...", "citations": ["c001"] } ], "cta": "...", "disclaimer": "..." } }\n\n',
            "Idea (with best hook + outline if available):\n",
            f"{idea}",
        ]
    )


def user_ideation_carousel_review(
    *,
    idea: dict[str, Any],
    carousel_draft: dict[str, Any],
    audience: dict[str, Any] | None = None,
    source_evidence: str | None = None,
) -> str:
    return "".join(
        [
            "Critique this drafted carousel and decide: approve, revise, or veto.\n\n",
            _writer_source_evidence_block(source_evidence),
            _audience_block(audience),
            "Quality bar:\n"
            "- Depth without fluff: every slide must add distinct value.\n"
            "- Tight, polished language; no generic filler unless pillar is validation.\n"
            "- No invented facts; specific claims must be supported by the provided citations.\n"
            "- Must fit DocMap framing (prep/navigation/clarity; not diagnosis/medical advice).\n\n"
            "If status=revise:\n"
            "- Provide 3–8 rewrite directives that the Writer MUST follow.\n"
            "- Be specific (which slides to cut/merge, what to replace, what to add).\n\n"
            "If status=veto:\n"
            "- This carousel should not be published; explain why.\n\n"
            "Return JSON with shape:\n"
            '{ "idea_id": "i003", "decision": { "status": "approve|revise|veto", "reason": "...", "directives": ["..."] },'
            '  "issues": [ { "slide_number": 3, "type": "fluff|uncited|vague|off_brand|redundant|too_long", "detail": "..." } ],'
            '  "one_sentence_good": "...", "one_sentence_fix": "..." }\n\n'
            "Idea:\n"
            f"{idea}\n\n"
            "Draft:\n"
            f"{carousel_draft}",
        ]
    )


def user_rewrite_slides_from_ideation(
    *,
    idea: dict[str, Any],
    carousel_draft: dict[str, Any],
    ideation_review: dict[str, Any],
    max_slides: int,
    audience: dict[str, Any] | None = None,
    performance_digest: dict[str, Any] | None = None,
    source_evidence: str | None = None,
) -> str:
    perf_w = ""
    if performance_digest:
        perf_w = (
            "Performance-informed hints (keep after applying directives):\n"
            f"{digest_writer_hints(performance_digest)}\n\n"
        )
    return "".join(
        [
            "Rewrite this carousel draft to address the Ideation pass directives.\n\n",
            _writer_source_evidence_block(source_evidence),
            _audience_block(audience),
            _writer_context_blocks(idea if isinstance(idea, dict) else None),
            _persona_preset_block(idea if isinstance(idea, dict) else None),
            perf_w,
            f"Constraints: target 5–{max_slides} slides; do not add fluff.\n"
            "- Keep the same topic and pillar.\n"
            "- Keep depth and specificity; cut redundancy.\n"
            "- Do not invent facts; keep citations on specific claims.\n"
            "- Follow the Ideation directives exactly.\n"
            "- Keep disclaimer text only in the `disclaimer` field—do not add a standalone disclaimer-only slide.\n"
            f"{DOCMAP_CTA_VOICE}"
            "Return JSON with shape:\n"
            '{ "carousel_draft": { "slides": [ { "slide_number": 1, "main_text": "...", "subtext": "...", "citations": ["c001"] } ], "cta": "...", "disclaimer": "..." } }\n\n'
            "Idea:\n"
            f"{idea}\n\n"
            "Current draft:\n"
            f"{carousel_draft}\n\n"
            "Ideation review (directives to apply):\n"
            f"{ideation_review}",
        ]
    )


def user_qa_edit(*, draft: dict[str, Any], source_evidence: str | None = None) -> str:
    head = (
        "Do a single QA/edit pass on this carousel draft.\n"
        "Rules:\n"
        "- If a slide contains a specific claim that is not cited, FIX it (either add a citation if available, or rewrite to remove/soften the specificity).\n"
        "- Prefer aligning strong claims to the SOURCE TEXT below when it is provided.\n"
        "- Remove fluff and redundancy.\n"
        "- Tighten wording and improve information density.\n"
        "- Enforce clarity and scannability.\n"
        "- Cold-audience / assumed knowledge: if the copy assumes UK-specific roles, clinic pathways, or jargon (e.g. specialist job titles) without a one-line primer, FIX it—add a brief clarifying phrase on the same slide or a dedicated explainer slide, or soften the claim.\n"
        "- If there is a slide that only repeats disclaimer/legal text that already belongs in `disclaimer`, remove that slide and keep a single `disclaimer` field.\n"
        "- If the CTA sounds like third-person brand boilerplate or a hard sell, rewrite it to match DocMap CTA voice (you/your; natural; product-true).\n\n"
        "Goal:\n"
        "- Return an updated carousel_draft that is publishable, not just a list of issues.\n\n"
        "Return JSON with shape:\n"
        '{ "carousel_draft": { ...updated... }, "qa": { "passed": true|false, "issues": [ { "type": "...", "detail": "...", "slide": 4 } ] } }\n\n'
    )
    return head + _writer_source_evidence_block(source_evidence) + f"{draft}"


def system_orchestrator() -> str:
    return (
        "You are the experiment program planner for a UK health-education Instagram carousel pipeline.\n"
        "Your job is to propose 1–3 queued A/B hypotheses that are feasible with the current system:\n"
        "- Each experiment tests contrasting hook styles on ONE shortlisted idea (same topic), split into two arms after hooks.\n"
        "- Dimensions you can emphasise: hook framing (how_to vs myth_bust vs checklist), pillar fit, or topic focus.\n"
        "- Use performance data as weak signals; do not overfit to noise. Prefer clear, falsifiable hypotheses.\n"
        "Return JSON matching the schema exactly. Be concise."
    )


def user_orchestrator_suggest(*, signals_json: str, queue_summary: str) -> str:
    return (
        "Below is `signals_json`: performance aggregates, optional digest, recent concluded experiments.\n"
        "Below is `queue_summary`: current experiment queue (may be empty).\n\n"
        "Task:\n"
        "- Propose hypotheses that complement the queue (avoid duplicates).\n"
        "- Set target_pillar to one of: recognition | validation | access_or_decision when possible.\n"
        "- treatment_key should be a short snake_case label (e.g. hook_style_myth_vs_howto).\n"
        "- Optionally set topic_hint to help match a future shortlist idea (substring of topic title).\n"
        "- Add program_notes if you want to warn about confounding, seasonality, or sample size.\n\n"
        f"signals_json:\n{signals_json}\n\n"
        f"queue_summary:\n{queue_summary}\n"
    )

