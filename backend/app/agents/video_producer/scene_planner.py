"""Deterministic scene planning from a :class:`ScriptResult`.

Scene conversion is pure computation: the scriptwriter already decided
what is said and which sources back it, so this module only *lays that
out* — order, timing and visual instructions. No LLM is involved, which
is what makes the resulting timeline reproducible.

Durations come from narration word counts divided by the script's
words-per-minute value (default 150). The backend calculation is
authoritative; nothing here trusts a model-supplied duration.
"""

from __future__ import annotations

from app.agents.scriptwriter.schemas import ScriptResult
from app.agents.video_producer.schemas import Scene, VisualType

#: Fallback narration speed when a script does not carry one.
WORDS_PER_MINUTE_DEFAULT = 150

#: A scene with no narration still needs screen time to be readable.
MIN_SCENE_SECONDS = 1.0

#: Cap on a single scene's duration so one wall of narration cannot
#: swallow the whole video when WPM is low.
MAX_SCENE_SECONDS = 120.0

# Scene kinds in playback order. Hook/intro/conclusion/CTA come from
# the script's own fields; sections fill the middle.
_HOOK = "hook"
_INTRODUCTION = "introduction"
_SECTION = "section"
_CONCLUSION = "conclusion"
_CTA = "cta"

#: Deterministic visual instruction per scene kind. These are
#: *production notes* for a human or a later asset phase — they never
#: cause an image to be generated in this phase.
_VISUAL_BY_KIND: dict[str, tuple[VisualType, str]] = {
    _HOOK: (VisualType.TEXT, "Title card: bold hook text over background"),
    _INTRODUCTION: (VisualType.BACKGROUND, "Opening background with topic title"),
    _SECTION: (VisualType.IMAGE, "Illustrative still or B-roll for this section"),
    _CONCLUSION: (VisualType.BACKGROUND, "Closing background summarising the topic"),
    _CTA: (VisualType.TEXT, "End card: call-to-action text"),
}


def _words(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def _duration_for(narration: str, wpm: int) -> float:
    """Narration length at ``wpm``, bounded to a sane range."""
    if wpm <= 0:
        wpm = WORDS_PER_MINUTE_DEFAULT
    seconds = _words(narration) / wpm * 60.0
    return round(min(max(seconds, MIN_SCENE_SECONDS), MAX_SCENE_SECONDS), 3)


def _scene_rows(
    script: ScriptResult,
) -> list[tuple[str, str, str, tuple[dict, ...]]]:
    """Flatten the script into ``(kind, heading, narration, claims)`` rows.

    Order mirrors how a scriptwriter thinks: hook → intro → sections →
    conclusion → CTA. Empty parts are skipped rather than rendered as
    blank scenes.
    """
    rows: list[tuple[str, str, str, tuple[dict, ...]]] = []
    if script.hook.strip():
        rows.append((_HOOK, script.title or "Hook", script.hook, ()))
    if script.introduction.strip():
        rows.append((_INTRODUCTION, "Introduction", script.introduction, ()))

    # Sectioned scripts contribute one scene per section; a script that
    # only carries flat narration gets that narration as a single body
    # scene, attributed with the script-level claim references.
    flat_claims = tuple(claim.model_dump() for claim in script.claims_used)
    if script.sections:
        for section in script.sections:
            rows.append(
                (
                    _SECTION,
                    section.heading,
                    section.narration,
                    tuple(section.claim_references),
                )
            )
    elif script.narration.strip():
        rows.append((_SECTION, script.title, script.narration, flat_claims))

    if script.conclusion.strip():
        rows.append((_CONCLUSION, "Conclusion", script.conclusion, ()))
    if script.cta.strip():
        rows.append((_CTA, "Call to action", script.cta, ()))

    # A title-only script still needs one readable scene.
    if not rows and script.title.strip():
        rows.append((_SECTION, script.title, script.title, flat_claims))
    return rows


def _sources_for(claims: tuple[dict, ...], sources: tuple[dict, ...]) -> tuple[dict, ...]:
    """Return only the sources the scene's claims actually cite.

    A scene that cites nothing gets no source references: attributing
    the whole script's bibliography to every scene would be a claim
    the script never made.
    """
    if not claims or not sources:
        return ()
    cited: set[str] = set()
    for claim in claims:
        source_ids = claim.get("source_ids") or ()
        cited.update(str(sid) for sid in source_ids)
    if not cited:
        return ()
    return tuple(
        source for source in sources if str(source.get("id", "")) in cited
    )


def plan_scenes(script: ScriptResult, wpm: int | None = None) -> list[Scene]:
    """Convert ``script`` into an ordered, timed list of scenes.

    Claim and source references are carried through unchanged: a
    section's ``claim_references`` (each with its ``source_ids``) is
    preserved verbatim, and the scene's source references are narrowed
    to the sources those claims cite.
    """
    effective_wpm = wpm or script.words_per_minute or WORDS_PER_MINUTE_DEFAULT
    sources = script.sources
    scenes: list[Scene] = []

    for index, (kind, heading, narration, claims) in enumerate(
        _scene_rows(script), start=1
    ):
        visual_type, visual_prompt = _VISUAL_BY_KIND[kind]
        scenes.append(
            Scene(
                scene_id=f"scene_{index:03d}",
                order=index,
                heading=heading,
                narration=narration,
                duration_seconds=_duration_for(narration, effective_wpm),
                visual_type=visual_type,
                visual_prompt=visual_prompt,
                asset_requirements=(
                    ({"media_type": "image", "prompt": visual_prompt},)
                    if visual_type is VisualType.IMAGE
                    else ()
                ),
                transition="fade" if index > 1 else "",
                claim_references=claims,
                source_references=_sources_for(claims, sources),
            )
        )
    return scenes
