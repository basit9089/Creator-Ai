"""Captions / Thumbnail / SEO agent tests.

The agent's contract: three artifacts (caption files, a real PNG,
validated metadata) produced from one script, each honestly reported.
Everything that can fail is made to fail here — AI down, malformed AI
output, renderer crashes, a blocked filesystem — and the result must
say exactly what was and was not produced. Never a fake file, never a
claimed success without one.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from app.agents.base import AgentContext, PlaceholderAgent
from app.agents.captions_thumbnail import (
    REQUEST_KEY,
    CaptionsThumbnailAgent,
    CaptionsThumbnailRequest,
)
from app.agents.captions_thumbnail.exceptions import (
    CaptionsThumbnailCaptionError,
    CaptionsThumbnailInputError,
)
from app.agents.captions_thumbnail.schemas import (
    ArtifactStatus,
    CaptionsThumbnailStatus,
)
from app.agents.registry import (
    IMPLEMENTED_AGENT_NAMES,
    PLANNED_AGENT_NAMES,
    build_default_registry,
)
from app.agents.runner import AgentRunner
from app.agents.scriptwriter.schemas import (
    ScriptClaimReference,
    ScriptResult,
    Section,
)
from app.agents.video_producer.schemas import (
    Scene,
    Timeline,
    TimelineScene,
    VideoProductionResult,
)
from app.core.config import Settings
from app.services.ai.provider import ProviderUnavailableError
from app.services.media.captions import parse_srt, parse_vtt
from tests.conftest import FakeAIProvider

#: A schema-valid metadata draft that still contains every trap the
#: normaliser must handle: an unsourced URL, an ungrounded tag and an
#: ungrounded search phrase.
DRAFT = {
    "title_candidates": [
        "How YouTube Started in 2005: A Short History",
        "The Real Story Behind YouTube's Launch",
    ],
    "description": (
        "In this video we explain how YouTube started in 2005. "
        "See https://evil.example.com for more."
    ),
    "tags": ["youtube", "history of youtube", "2005", "martian colonization"],
    "primary_keyword": "youtube history",
    "secondary_keywords": ["video sharing", "early internet"],
    "search_phrases": ["how youtube started", "aliens land on pluto"],
    "content_category": "Education",
}

#: The title the deterministic scorer must pick (60 points: friendly
#: length plus a specificity digit), over the runner-up at 50.
EXPECTED_TITLE = "How YouTube Started in 2005: A Short History"


def make_script(**overrides: object) -> ScriptResult:
    base: dict[str, object] = {
        "title": "How YouTube Started",
        "hook": "What if one video site changed the internet forever?",
        "introduction": "Today we cover the launch of YouTube in 2005.",
        "sections": (
            Section(
                section_id="s1",
                heading="The launch",
                narration="YouTube launched in 2005 as a video sharing site.",
                claim_references=({"claim_id": "c1", "source_ids": ("src1",)},),
            ),
        ),
        "narration": (
            "YouTube was founded in 2005 and changed how people share "
            "videos online. It grew quickly around the world."
        ),
        "conclusion": "That is how YouTube started.",
        "cta": "Subscribe for more history videos.",
        "claims_used": (
            ScriptClaimReference(claim_id="c1", source_ids=("src1",), status="verified"),
        ),
        "sources": (
            {
                "id": "src1",
                "url": "https://en.wikipedia.org/wiki/YouTube",
                "domain": "en.wikipedia.org",
            },
        ),
    }
    base.update(overrides)
    return ScriptResult(**base)  # type: ignore[arg-type]


def make_agent(
    tmp_path: Path, *, text: str | None = None, error: Exception | None = None, **kwargs: object
) -> CaptionsThumbnailAgent:
    provider = FakeAIProvider(text=json.dumps(DRAFT) if text is None else text, error=error)
    return CaptionsThumbnailAgent(
        provider=provider,  # type: ignore[arg-type]
        settings=Settings(storage_root=str(tmp_path)),
        **kwargs,  # type: ignore[arg-type]
    )


def run(agent: CaptionsThumbnailAgent, request: CaptionsThumbnailRequest, **extra: object):
    context = AgentContext(config={REQUEST_KEY: request, **extra})
    return asyncio.run(AgentRunner().run(agent, context, []))


def make_request(**overrides: object) -> CaptionsThumbnailRequest:
    base: dict[str, object] = {"script": make_script(), "project_id": "demo"}
    base.update(overrides)
    return CaptionsThumbnailRequest(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Full success path
# --------------------------------------------------------------------------


class TestCompletedRun:
    def test_all_three_artifacts_are_produced_and_verified(
        self, tmp_path: Path
    ) -> None:
        agent = make_agent(tmp_path)
        result = run(agent, make_request())

        assert result.success is True
        outcome = result.data
        assert outcome.status is CaptionsThumbnailStatus.COMPLETED
        assert outcome.success is True
        assert outcome.errors == ()

        # Captions: real files that parse to the reported count.
        assert outcome.captions.status is ArtifactStatus.COMPLETED
        assert outcome.captions.timing_source == "script_wpm"
        srt = Path(outcome.captions.srt_path)
        vtt = Path(outcome.captions.vtt_path)
        assert srt.is_file() and vtt.is_file()
        cues = parse_srt(srt.read_text(encoding="utf-8"))
        assert len(cues) == outcome.captions.caption_count > 0
        assert list(parse_vtt(vtt.read_text(encoding="utf-8"))) == cues
        assert outcome.captions.duration_seconds == pytest.approx(cues[-1].end_ms / 1000, abs=0.001)

        # Thumbnail: a real PNG at the default geometry.
        assert outcome.thumbnail.status is ArtifactStatus.COMPLETED
        assert outcome.thumbnail.renderer == "local"
        png = Path(outcome.thumbnail.path)
        assert png.is_file()
        assert outcome.thumbnail.size_bytes == png.stat().st_size
        with Image.open(png) as opened:
            assert opened.format == "PNG"
            assert opened.size == (1280, 720)

        # Metadata: the selected title, the score, a written JSON file.
        assert outcome.metadata.status is ArtifactStatus.COMPLETED
        assert outcome.metadata.title == EXPECTED_TITLE
        assert outcome.metadata.title_score == 60
        assert 0 <= (outcome.metadata.title_score or 0) <= 100
        meta_path = Path(outcome.metadata.path)
        assert meta_path.is_file()
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        assert payload["metadata"]["title"] == EXPECTED_TITLE
        assert payload["project_id"] == "demo"

        # Grounding: claim -> source -> section survived.
        assert [ref.claim_id for ref in outcome.grounding.claim_refs] == ["c1"]
        assert outcome.grounding.claim_refs[0].source_ids == ("src1",)
        assert outcome.grounding.claim_refs[0].sections == ("The launch",)
        assert outcome.grounding.source_refs[0].url == "https://en.wikipedia.org/wiki/YouTube"

        # Every produced path lives under this run's project directory.
        root = (tmp_path / "projects" / "demo").resolve()
        for path in (srt, vtt, png, meta_path):
            assert path.resolve().is_relative_to(root)

    def test_metadata_is_normalised_and_grounded(self, tmp_path: Path) -> None:
        outcome = run(make_agent(tmp_path), make_request()).data

        # Ungrounded tag and phrase are gone; grounded ones remain, in
        # order, deduplicated and whitespace-normalised.
        assert outcome.metadata.tags == ("youtube", "history of youtube", "2005")
        assert outcome.metadata.search_phrases == ("how youtube started",)
        assert outcome.metadata.secondary_keywords == ("video sharing", "early internet")
        assert outcome.metadata.primary_keyword == "youtube history"
        assert outcome.metadata.content_category == "Education"
        assert outcome.metadata.alternate_titles == (
            "The Real Story Behind YouTube's Launch",
        )

        # The AI-written URL was scrubbed; only script sources are linked.
        description = outcome.metadata.description
        assert "evil.example.com" not in description
        assert "Removed 1 URL(s)" in " ".join(outcome.warnings)
        urls = set(re.findall(r"https?://\S+", description))
        assert urls == {"https://en.wikipedia.org/wiki/YouTube"}
        assert "Sources:" in description

    def test_run_is_deterministic(self, tmp_path: Path) -> None:
        first = run(make_agent(tmp_path), make_request()).data
        first_srt = Path(first.captions.srt_path).read_bytes()
        first_png = Path(first.thumbnail.path).read_bytes()
        first_meta = Path(first.metadata.path).read_bytes()

        second = run(make_agent(tmp_path), make_request()).data
        assert Path(second.captions.srt_path).read_bytes() == first_srt
        assert Path(second.thumbnail.path).read_bytes() == first_png
        assert Path(second.metadata.path).read_bytes() == first_meta
        assert second.metadata.title == first.metadata.title

    def test_caption_format_subset_writes_only_the_requested_format(
        self, tmp_path: Path
    ) -> None:
        outcome = run(
            make_agent(tmp_path), make_request(caption_formats=("srt",))
        ).data

        assert outcome.captions.formats == ("srt",)
        assert outcome.captions.vtt_path == ""
        captions_dir = tmp_path / "projects" / "demo" / "captions"
        assert sorted(p.name for p in captions_dir.iterdir()) == ["captions.srt"]

    def test_custom_resolution_and_watermark_reach_the_thumbnail(
        self, tmp_path: Path
    ) -> None:
        outcome = run(
            make_agent(tmp_path),
            make_request(thumbnail_resolution="640x360", watermark="CREATOR AI"),
        ).data

        assert (outcome.thumbnail.width, outcome.thumbnail.height) == (640, 360)
        with Image.open(outcome.thumbnail.path) as opened:
            assert opened.size == (640, 360)

    def test_very_long_narration_truncates_the_prompt_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        script = make_script(
            sections=(),
            narration="YouTube grew fast and kept changing. " * 400,
        )
        outcome = run(make_agent(tmp_path), make_request(script=script)).data

        assert outcome.status is CaptionsThumbnailStatus.COMPLETED
        assert any("truncated for the metadata prompt" in w for w in outcome.warnings)


# --------------------------------------------------------------------------
# Timing sources
# --------------------------------------------------------------------------


class TestTimingSources:
    @staticmethod
    def video(*, with_timeline: bool) -> VideoProductionResult:
        scenes = (
            Scene(scene_id="scene_001", narration="First scene words here.", duration_seconds=4.0),
            Scene(scene_id="scene_002", narration="Second scene words here.", duration_seconds=6.0),
        )
        timeline = None
        if with_timeline:
            timeline = Timeline(
                scenes=(
                    TimelineScene(scene_id="scene_001", start_time=0.5, end_time=4.5),
                    TimelineScene(scene_id="scene_002", start_time=4.5, end_time=12.5),
                )
            )
        return VideoProductionResult(scenes=scenes, timeline=timeline)

    def test_video_timeline_drives_caption_times(self, tmp_path: Path) -> None:
        outcome = run(
            make_agent(tmp_path),
            make_request(video=self.video(with_timeline=True)),
        ).data

        assert outcome.captions.timing_source == "video_timeline"
        assert outcome.captions.duration_seconds == pytest.approx(12.5, abs=0.001)

    def test_video_without_a_timeline_uses_scene_durations(self, tmp_path: Path) -> None:
        outcome = run(
            make_agent(tmp_path),
            make_request(video=self.video(with_timeline=False)),
        ).data

        assert outcome.captions.timing_source == "video_scene_durations"
        assert outcome.captions.duration_seconds == pytest.approx(10.0, abs=0.001)

    def test_script_alone_uses_the_wpm_convention(self, tmp_path: Path) -> None:
        outcome = run(make_agent(tmp_path), make_request()).data
        assert outcome.captions.timing_source == "script_wpm"


# --------------------------------------------------------------------------
# Partial outcomes: what failed, honestly
# --------------------------------------------------------------------------


class TestPartialOutcomes:
    def test_ai_unavailable_still_produces_captions_and_thumbnail(
        self, tmp_path: Path
    ) -> None:
        agent = make_agent(
            tmp_path, error=ProviderUnavailableError("connection refused")
        )
        result = run(agent, make_request())
        outcome = result.data

        assert result.success is True  # the runner saw usable output
        assert outcome.status is CaptionsThumbnailStatus.PARTIAL
        assert outcome.success is False  # the payload claims nothing false
        assert outcome.metadata.status is ArtifactStatus.FAILED
        assert outcome.metadata.error_code == "ai_provider_unavailable"
        assert outcome.captions.status is ArtifactStatus.COMPLETED
        assert outcome.thumbnail.status is ArtifactStatus.COMPLETED
        assert Path(outcome.captions.srt_path).is_file()
        assert Path(outcome.thumbnail.path).is_file()
        assert any(e.startswith("metadata: ai_provider_unavailable") for e in outcome.errors)

    def test_malformed_ai_output_fails_only_the_metadata(self, tmp_path: Path) -> None:
        outcome = run(make_agent(tmp_path, text="this is not json"), make_request()).data

        assert outcome.status is CaptionsThumbnailStatus.PARTIAL
        assert outcome.metadata.status is ArtifactStatus.FAILED
        assert outcome.metadata.error_code == "structured_output_invalid"
        assert outcome.metadata.title == ""
        assert outcome.captions.status is ArtifactStatus.COMPLETED
        assert outcome.thumbnail.status is ArtifactStatus.COMPLETED
        assert any(e.startswith("metadata: structured_output_invalid") for e in outcome.errors)

        # Without a chosen title the thumbnail falls back to the
        # script's own title — and says so.
        assert outcome.thumbnail.filename == "thumbnail.png"
        assert any("Thumbnail uses the script title" in w for w in outcome.warnings)
        with Image.open(outcome.thumbnail.path) as opened:
            assert opened.format == "PNG"

    def test_renderer_crash_is_reported_as_a_render_failure(self, tmp_path: Path) -> None:
        class ExplodingRenderer:
            name = "exploding"

            def health_check(self) -> bool:
                return True

            def render(self, spec, output_path):  # noqa: ANN001
                from app.services.media.thumbnails import ThumbnailRenderError

                raise ThumbnailRenderError("simulated renderer crash")

        outcome = run(
            make_agent(tmp_path, renderer=ExplodingRenderer()),  # type: ignore[arg-type]
            make_request(),
        ).data

        assert outcome.status is CaptionsThumbnailStatus.PARTIAL
        assert outcome.thumbnail.status is ArtifactStatus.FAILED
        assert outcome.thumbnail.error_code == "captions_thumbnail_render_error"
        assert outcome.captions.status is ArtifactStatus.COMPLETED
        assert outcome.metadata.status is ArtifactStatus.COMPLETED
        assert Path(outcome.metadata.path).is_file()
        assert any("simulated renderer crash" in e for e in outcome.errors)

    def test_missing_fonts_fail_only_the_thumbnail(self, tmp_path: Path) -> None:
        from app.services.media.thumbnails import LocalThumbnailRenderer

        outcome = run(
            make_agent(
                tmp_path,
                renderer=LocalThumbnailRenderer(
                    font_candidates=(), font_bold_candidates=()
                ),
            ),
            make_request(),
        ).data

        assert outcome.status is CaptionsThumbnailStatus.PARTIAL
        assert outcome.thumbnail.status is ArtifactStatus.FAILED
        assert outcome.thumbnail.error_code == "captions_thumbnail_render_error"
        assert any("font" in e for e in outcome.errors)

    def test_a_missing_supplied_image_degrades_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        outcome = run(
            make_agent(tmp_path),
            make_request(thumbnail_image="does/not/exist.png"),
        ).data

        # The render still completes — without the image, and honestly
        # warning about it.
        assert outcome.status is CaptionsThumbnailStatus.COMPLETED
        assert any("does not exist" in w for w in outcome.warnings)


# --------------------------------------------------------------------------
# Total failure and retries
# --------------------------------------------------------------------------


class TestTotalFailureAndRetry:
    @staticmethod
    def block_project(tmp_path: Path, settings: Settings) -> Path:
        """Make ``projects/demo`` a file so every artifact's write fails."""
        settings.projects_dir.mkdir(parents=True, exist_ok=True)
        blocker = settings.projects_dir / "demo"
        blocker.write_bytes(b"blocked")
        return blocker

    def test_nothing_produced_raises_a_typed_error(self, tmp_path: Path) -> None:
        agent = make_agent(tmp_path)
        self.block_project(tmp_path, agent.settings)

        result = run(agent, make_request())
        assert result.success is False
        assert result.error is not None
        assert "No artifacts were produced" in result.error
        assert result.metadata["code"] == "captions_thumbnail_caption_error"
        assert result.metadata["retryable"] is True

    def test_a_retry_succeeds_once_the_blocker_is_gone(self, tmp_path: Path) -> None:
        agent = make_agent(tmp_path)
        blocker = self.block_project(tmp_path, agent.settings)
        runner = AgentRunner()

        first = asyncio.run(
            runner.run(agent, AgentContext(config={REQUEST_KEY: make_request()}), [])
        )
        assert first.success is False

        blocker.unlink()
        # Pass the same agent instance: a bare retry() would re-resolve
        # the agent from the registry, whose fresh instance resolves the
        # real AI provider (fine in production, wrong for a fake).
        second = asyncio.run(runner.retry(first.execution_id, agent=agent))

        assert second.success is True
        assert second.metadata["attempt"] == 2
        outcome = second.data
        assert outcome.status is CaptionsThumbnailStatus.COMPLETED
        assert Path(outcome.captions.srt_path).is_file()

    def test_missing_request_is_an_input_error(self, tmp_path: Path) -> None:
        agent = make_agent(tmp_path)
        with pytest.raises(CaptionsThumbnailInputError, match="No captions/thumbnail"):
            asyncio.run(agent.run(AgentContext(config={}), []))

    def test_invalid_raw_request_is_an_input_error(self, tmp_path: Path) -> None:
        agent = make_agent(tmp_path)
        context = AgentContext(config={REQUEST_KEY: {"project_id": "../evil"}})
        with pytest.raises(CaptionsThumbnailInputError, match="Invalid"):
            asyncio.run(agent.run(context, []))


# --------------------------------------------------------------------------
# Request schema guards
# --------------------------------------------------------------------------


class TestRequestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"project_id": "../evil"},
            {"project_id": ""},
            {"thumbnail_resolution": "banana"},
            {"thumbnail_resolution": "100x100"},
            {"caption_formats": ["doc"]},
            {"caption_formats": []},
            {"thumbnail_image": "../../etc/passwd"},
            {"thumbnail_image": "C:\\evil.png"},
            {"language": "x"},
            {"target_audience": "   "},
        ],
    )
    def test_bad_inputs_are_rejected_by_the_schema(self, overrides: dict) -> None:
        with pytest.raises(ValidationError):
            make_request(**overrides)

    def test_a_script_without_narratable_content_is_rejected(self) -> None:
        empty = ScriptResult(title="Blank")
        with pytest.raises(ValidationError, match="narratable"):
            make_request(script=empty)

    def test_thumbnail_size_is_derived_from_the_resolution(self) -> None:
        assert make_request(thumbnail_resolution="640x360").thumbnail_size == (640, 360)


# --------------------------------------------------------------------------
# Registry and architecture
# --------------------------------------------------------------------------


class TestRegistryAndArchitecture:
    def test_the_registry_returns_the_real_agent(self) -> None:
        registry = build_default_registry()
        agent = registry.get_agent("captions_thumbnail")
        assert isinstance(agent, CaptionsThumbnailAgent)
        assert agent.name == "captions_thumbnail"

    def test_only_the_implemented_agents_are_executable(self) -> None:
        assert "captions_thumbnail" in IMPLEMENTED_AGENT_NAMES
        assert "publisher" in PLANNED_AGENT_NAMES
        assert "publisher" not in IMPLEMENTED_AGENT_NAMES
        registry = build_default_registry()
        assert isinstance(registry.get_agent("publisher"), PlaceholderAgent)

    def test_the_injected_provider_is_the_one_used(self, tmp_path: Path) -> None:
        agent = make_agent(tmp_path)
        # Laziness never hides a different provider: the injected one is
        # resolved on first access and then kept.
        assert isinstance(agent.provider, FakeAIProvider)
        assert agent.provider is agent.provider

    def test_metadata_generation_uses_documented_conservative_options(
        self, tmp_path: Path
    ) -> None:
        agent = make_agent(tmp_path)
        run(agent, make_request())

        messages, options = agent.provider.calls[-1]
        assert options is not None
        assert options.think is False  # DEFAULT_AI_THINKING=false continues
        assert options.temperature == 0.3
        assert options.top_p == 0.9
        assert options.max_tokens == 1500
        assert len(messages) >= 2

    def test_thinking_can_be_enabled_per_execution(self, tmp_path: Path) -> None:
        agent = make_agent(tmp_path)
        run(agent, make_request(), thinking=True)
        _, options = agent.provider.calls[-1]
        assert options is not None and options.think is True

    def test_no_module_in_this_phase_imports_ollama(self) -> None:
        backend = Path(__file__).resolve().parents[1]
        roots = [
            backend / "app" / "agents" / "captions_thumbnail",
            backend / "app" / "services" / "media",
        ]
        scanned = 0
        for root in roots:
            for file in root.rglob("*.py"):
                scanned += 1
                source = file.read_text(encoding="utf-8")
                assert "import ollama" not in source, file
                assert "from ollama" not in source, file
        assert scanned >= 10

    def test_the_agent_reports_itself_as_structured_and_retryable(self) -> None:
        from app.agents.base import Capability

        agent = CaptionsThumbnailAgent()
        assert Capability.STRUCTURED_OUTPUT in agent.capabilities
        assert Capability.RETRYABLE in agent.capabilities
        assert agent.version == "1.0.0"
