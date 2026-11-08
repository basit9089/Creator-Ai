"""Caption subsystem tests: timing, chunking, formats, and files.

The contract under test: captions are *real*. Timing is deterministic
and derived from actual narration (the video timeline when present,
the project's 150 WPM convention otherwise), cue text is narration
verbatim, SRT/WebVTT output round-trips through the strict parsers,
and every written file lands inside the configured projects root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.scriptwriter.schemas import (
    ScriptClaimReference,
    ScriptResult,
    Section,
)
from app.agents.video_producer.scene_planner import (
    WORDS_PER_MINUTE_DEFAULT,
    plan_scenes,
)
from app.agents.video_producer.schemas import (
    Scene,
    Timeline,
    TimelineScene,
    VideoProductionResult,
)
from app.services.media.captions import (
    CaptionCue,
    CaptionError,
    CaptionFormatError,
    CaptionService,
    SRTFormatter,
    Segment,
    WebVTTFormatter,
    build_cues,
    parse_srt,
    parse_vtt,
    segments_from_script,
    segments_from_video,
    split_narration,
    validate_cues,
    wrap_cue_text,
)
from app.services.media.captions.timing import validate_segments
from app.services.media.paths import MediaPathError, safe_filename, validate_path


def make_script(**overrides: object) -> ScriptResult:
    """A small but complete script with one claim and one real source."""
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


def make_video(*, timeline: Timeline | None = None) -> VideoProductionResult:
    """A two-scene production result; timeline optional."""
    scenes = (
        Scene(
            scene_id="scene_001",
            order=1,
            heading="Opening",
            narration="First scene narration here.",
            duration_seconds=4.0,
        ),
        Scene(
            scene_id="scene_002",
            order=2,
            heading="Middle",
            narration="Second scene keeps going with more words.",
            duration_seconds=6.0,
        ),
    )
    return VideoProductionResult(scenes=scenes, timeline=timeline)


def make_cues() -> list[CaptionCue]:
    return build_cues(segments_from_script(make_script()))


# --------------------------------------------------------------------------
# Segment timing
# --------------------------------------------------------------------------


class TestSegmentTiming:
    def test_script_segments_follow_the_wpm_convention(self) -> None:
        assert WORDS_PER_MINUTE_DEFAULT == 150
        segments = segments_from_script(make_script())

        # Every narratable scene contributes one segment, in order.
        assert len(segments) == 5
        assert segments[0].start_ms == 0

        # The section's window is exactly its word count at 150 WPM:
        # one word takes 400 ms.
        section = next(s for s in segments if s.label == "The launch")
        assert section.end_ms - section.start_ms == len(section.text.split()) * 400

    def test_script_segments_are_contiguous_and_match_plan_scenes(self) -> None:
        script = make_script()
        scenes = plan_scenes(script)
        segments = segments_from_script(script)

        expected = []
        cursor = 0
        for scene in scenes:
            duration = round(scene.duration_seconds * 1000)
            if scene.narration.strip():
                expected.append((cursor, cursor + duration))
            cursor += duration
        assert [(s.start_ms, s.end_ms) for s in segments] == expected

        for previous, current in zip(segments, segments[1:], strict=False):
            assert current.start_ms == previous.end_ms

    def test_script_without_narration_is_rejected(self) -> None:
        # Title-only scripts get a readable title scene from the planner;
        # a script with nothing at all produces no segments.
        empty = ScriptResult(title="")
        with pytest.raises(CaptionError, match="no caption segments"):
            segments_from_script(empty)

    def test_video_timeline_windows_drive_the_segments(self) -> None:
        timeline = Timeline(
            scenes=(
                TimelineScene(scene_id="scene_001", start_time=0.5, end_time=4.5),
                TimelineScene(scene_id="scene_002", start_time=4.5, end_time=12.5),
            )
        )
        segments = segments_from_video(make_video(timeline=timeline))
        assert [(s.start_ms, s.end_ms) for s in segments] == [
            (500, 4500),
            (4500, 12500),
        ]

    def test_video_without_timeline_accumulates_scene_durations(self) -> None:
        segments = segments_from_video(make_video())
        assert [(s.start_ms, s.end_ms) for s in segments] == [
            (0, 4000),
            (4000, 10000),
        ]

    def test_video_with_empty_narration_scene_skips_it(self) -> None:
        video = VideoProductionResult(
            scenes=(
                Scene(scene_id="s1", narration="Spoken words.", duration_seconds=2.0),
                Scene(scene_id="s2", narration="   ", duration_seconds=3.0),
                Scene(scene_id="s3", narration="More spoken words here.", duration_seconds=2.0),
            )
        )
        segments = segments_from_video(video)
        assert len(segments) == 2
        # The silent scene still consumed its three seconds of window.
        assert segments[1].start_ms == 5000

    def test_video_without_scenes_is_rejected(self) -> None:
        with pytest.raises(CaptionError, match="no scenes"):
            segments_from_video(VideoProductionResult())

    def test_video_with_only_silent_scenes_is_rejected(self) -> None:
        video = VideoProductionResult(
            scenes=(Scene(scene_id="s1", narration="  ", duration_seconds=2.0),)
        )
        with pytest.raises(CaptionError, match="no narratable"):
            segments_from_video(video)

    def test_overlapping_video_timeline_is_rejected(self) -> None:
        timeline = Timeline(
            scenes=(
                TimelineScene(scene_id="scene_001", start_time=0.0, end_time=10.0),
                TimelineScene(scene_id="scene_002", start_time=5.0, end_time=15.0),
            )
        )
        with pytest.raises(CaptionError, match="overlaps"):
            segments_from_video(make_video(timeline=timeline))

    @pytest.mark.parametrize(
        "segment",
        [
            Segment(text="x", start_ms=500, end_ms=500),
            Segment(text="x", start_ms=900, end_ms=400),
            Segment(text="x", start_ms=-1, end_ms=100),
        ],
    )
    def test_invalid_segment_windows_are_rejected(self, segment: Segment) -> None:
        with pytest.raises(CaptionError):
            validate_segments([segment])

    def test_segment_overlap_is_rejected(self) -> None:
        with pytest.raises(CaptionError, match="overlaps"):
            validate_segments(
                [
                    Segment(text="a", start_ms=0, end_ms=1000),
                    Segment(text="b", start_ms=500, end_ms=1500),
                ]
            )


# --------------------------------------------------------------------------
# Cue building
# --------------------------------------------------------------------------


class TestCueBuilding:
    def test_cues_are_valid_ordered_and_deterministic(self) -> None:
        segments = segments_from_script(make_script())
        cues = build_cues(segments)
        assert cues  # something was produced
        validate_cues(cues)  # raises on any rule violation

        assert [c.index for c in cues] == list(range(1, len(cues) + 1))
        for previous, current in zip(cues, cues[1:], strict=False):
            assert previous.end_ms <= current.start_ms
        # Cues cover the script's segments from zero to the very end.
        assert cues[0].start_ms == 0
        assert cues[-1].end_ms == segments[-1].end_ms

        assert build_cues(segments) == cues

    def test_cue_text_is_narration_verbatim(self) -> None:
        segments = segments_from_script(make_script())
        cues = build_cues(segments)
        cue_words = " ".join(word for cue in cues for word in cue.text.split())
        segment_words = " ".join(word for segment in segments for word in segment.text.split())
        assert cue_words == segment_words

    def test_oversized_sentence_is_split_into_bounded_cues(self) -> None:
        long_text = (
            "This sentence goes on and on, repeating itself many times "
            "over, until it is far too long for one caption cue to hold. "
        ) * 3
        segments = [Segment(text=long_text, start_ms=0, end_ms=60_000)]
        cues = build_cues(segments)

        assert len(cues) > 1
        for cue in cues:
            flat = cue.text.replace("\n", " ")
            assert len(flat) <= 84
        # Even after splitting, every word survives in order.
        assert " ".join(c.text.replace("\n", " ") for c in cues) == " ".join(
            long_text.split()
        )

    def test_a_window_too_short_for_its_cues_fails_loudly(self) -> None:
        words = " ".join(f"word{i}" for i in range(60))
        with pytest.raises(CaptionError, match="too short"):
            build_cues([Segment(text=words, start_ms=0, end_ms=3)])

    def test_empty_narration_produces_no_cues_and_says_so(self) -> None:
        with pytest.raises(CaptionError, match="no captions"):
            build_cues([Segment(text="   ", start_ms=0, end_ms=1000)])

    def test_wrap_cue_text_bounds_lines_and_preserves_characters(self) -> None:
        text = (
            "One two three four five six seven eight nine ten eleven "
            "twelve thirteen fourteen"
        )
        wrapped = wrap_cue_text(text)
        lines = wrapped.split("\n")
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 42 or " " not in line
        assert wrapped.replace("\n", " ") == text

    def test_split_narration_preserves_every_word(self) -> None:
        text = (
            "First sentence. Second sentence, with a clause; and more — "
            "plus a dash-word. " + " filler" * 30
        )
        chunks = split_narration(text)
        assert chunks
        assert all(len(chunk) <= 84 for chunk in chunks)
        assert " ".join(" ".join(chunks).split()) == " ".join(text.split())

    def test_unicode_narration_survives_cue_building(self) -> None:
        script = make_script(
            sections=(
                Section(
                    section_id="s1",
                    heading="Ünïcode",
                    narration="Héllo — ça va? Ünïcode narration survives intact.",
                ),
            )
        )
        cues = build_cues(segments_from_script(script))
        assert any("Héllo" in cue.text for cue in cues)


# --------------------------------------------------------------------------
# SRT / WebVTT formats
# --------------------------------------------------------------------------


class TestFormatters:
    def test_srt_uses_crlf_and_comma_timestamps_and_round_trips(self) -> None:
        cues = make_cues()
        content = SRTFormatter().format(cues)

        assert content.startswith("1\r\n00:00:00,")
        assert content.endswith("\r\n\r\n")
        # CRLF everywhere: no bare LF survives the formatter.
        assert "\n" not in content.replace("\r\n", "")
        assert "--> 00:00:0" in content
        assert list(parse_srt(content)) == cues

    def test_vtt_uses_lf_header_and_dot_timestamps_and_round_trips(self) -> None:
        cues = make_cues()
        content = WebVTTFormatter().format(cues)

        assert content.startswith("WEBVTT\n")
        assert "\r" not in content
        assert content.count("-->") == len(cues)
        assert list(parse_vtt(content)) == cues

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("1\n00:00:00;000 --> 00:00:01,000\nhello\n", "wrong separator"),
            ("x\n00:00:00,000 --> 00:00:01,000\nhello\n", "non-numeric index"),
            ("1\n00:00:75,000 --> 00:01:00,000\nhello\n", "impossible clock"),
            ("nonsense", "not a caption block"),
        ],
    )
    def test_parse_srt_rejects_malformed_text(self, text: str, reason: str) -> None:
        with pytest.raises(CaptionFormatError):
            parse_srt(text)

    def test_parse_srt_rejects_overlapping_cues(self) -> None:
        text = (
            "1\n00:00:00,000 --> 00:00:02,000\nfirst\n\n"
            "2\n00:00:01,000 --> 00:00:03,000\nsecond\n"
        )
        with pytest.raises(CaptionFormatError, match="overlap"):
            parse_srt(text)

    def test_parse_vtt_rejects_a_missing_header(self) -> None:
        with pytest.raises(CaptionFormatError, match="header"):
            parse_vtt("1\n00:00:00.000 --> 00:00:01.000\nhello\n")

    def test_parse_vtt_rejects_impossible_clock_values(self) -> None:
        with pytest.raises(CaptionFormatError):
            parse_vtt("WEBVTT\n\n00:00:61.000 --> 00:01:00.000\nhello\n")


# --------------------------------------------------------------------------
# Cue-list validation
# --------------------------------------------------------------------------


class TestValidateCues:
    @pytest.mark.parametrize(
        ("cues", "reason"),
        [
            (
                [
                    CaptionCue(index=1, start_ms=0, end_ms=1000, text="a"),
                    CaptionCue(index=2, start_ms=500, end_ms=1500, text="b"),
                ],
                "overlaps",
            ),
            ([CaptionCue(index=1, start_ms=500, end_ms=500, text="x")], "not before end"),
            ([CaptionCue(index=1, start_ms=0, end_ms=100, text="   ")], "empty text"),
            ([CaptionCue(index=2, start_ms=0, end_ms=100, text="x")], "expected 1"),
        ],
    )
    def test_invalid_cue_lists_are_rejected(
        self, cues: list[CaptionCue], reason: str
    ) -> None:
        with pytest.raises(CaptionFormatError, match=reason):
            validate_cues(cues)

    def test_a_negative_start_bypassing_the_model_is_still_rejected(self) -> None:
        cue = CaptionCue.model_construct(index=1, start_ms=-5, end_ms=10, text="x")
        with pytest.raises(CaptionFormatError, match="negative"):
            validate_cues([cue])

    def test_formatters_validate_before_writing(self) -> None:
        overlapping = [
            CaptionCue(index=1, start_ms=0, end_ms=1000, text="a"),
            CaptionCue(index=2, start_ms=500, end_ms=1500, text="b"),
        ]
        with pytest.raises(CaptionFormatError):
            SRTFormatter().format(overlapping)
        with pytest.raises(CaptionFormatError):
            WebVTTFormatter().format(overlapping)


# --------------------------------------------------------------------------
# CaptionService: real files inside the projects root
# --------------------------------------------------------------------------


class TestCaptionService:
    def test_writes_verified_files_that_round_trip(self, tmp_path: Path) -> None:
        service = CaptionService(tmp_path)
        cues = make_cues()

        files = service.write(cues, project_id="demo")

        assert files.formats == ("srt", "vtt")
        assert files.caption_count == len(cues)
        assert files.duration_seconds == pytest.approx(cues[-1].end_ms / 1000, abs=0.001)

        for fmt, path_string in (("srt", files.srt_path), ("vtt", files.vtt_path)):
            path = Path(path_string)
            assert path.is_file()
            assert path.parent == tmp_path / "demo" / "captions"
            assert path.stat().st_size > 0
            assert files.size_bytes[fmt] == path.stat().st_size
            parser = parse_srt if fmt == "srt" else parse_vtt
            assert list(parser(path.read_text(encoding="utf-8"))) == list(cues)

    def test_a_format_subset_writes_only_that_format(self, tmp_path: Path) -> None:
        service = CaptionService(tmp_path)
        files = service.write(make_cues(), project_id="demo", formats=("srt",))

        assert files.formats == ("srt",)
        assert files.srt_path.endswith(".srt")
        assert files.vtt_path == ""
        captions_dir = tmp_path / "demo" / "captions"
        assert sorted(p.name for p in captions_dir.iterdir()) == ["captions.srt"]

    def test_repeated_writes_are_byte_identical(self, tmp_path: Path) -> None:
        service = CaptionService(tmp_path)
        cues = make_cues()

        first = service.write(cues, project_id="demo")
        first_bytes = {
            fmt: Path(path).read_bytes()
            for fmt, path in (("srt", first.srt_path), ("vtt", first.vtt_path))
        }
        second = service.write(cues, project_id="demo")
        second_bytes = {
            fmt: Path(path).read_bytes()
            for fmt, path in (("srt", second.srt_path), ("vtt", second.vtt_path))
        }
        assert first_bytes == second_bytes

    def test_empty_cue_list_cannot_produce_an_empty_file(self, tmp_path: Path) -> None:
        service = CaptionService(tmp_path)
        with pytest.raises(CaptionError, match="empty"):
            service.write([], project_id="demo")
        # The failure leaves no caption file claiming to exist: whatever
        # the write attempt left behind is zero bytes, never content.
        path = tmp_path / "demo" / "captions" / "captions.srt"
        assert not path.exists() or path.stat().st_size == 0

    @pytest.mark.parametrize("bad_project", ["../evil", "..", "/outside", "a/../../out"])
    def test_project_traversal_is_rejected(
        self, tmp_path: Path, bad_project: str
    ) -> None:
        service = CaptionService(tmp_path)
        with pytest.raises(MediaPathError):
            service.write(make_cues(), project_id=bad_project)
        # Nothing was written anywhere.
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("bad_stem", ["../evil", "a/b", "a\\b", ""])
    def test_unsafe_file_stem_is_rejected(self, tmp_path: Path, bad_stem: str) -> None:
        service = CaptionService(tmp_path)
        with pytest.raises(MediaPathError):
            service.write(make_cues(), project_id="demo", stem=bad_stem)

    def test_unknown_format_is_rejected(self, tmp_path: Path) -> None:
        service = CaptionService(tmp_path)
        with pytest.raises(CaptionError, match="unsupported"):
            service.write(make_cues(), project_id="demo", formats=("doc",))

    def test_no_formats_is_rejected(self, tmp_path: Path) -> None:
        service = CaptionService(tmp_path)
        with pytest.raises(CaptionError, match="no caption formats"):
            service.write(make_cues(), project_id="demo", formats=())

    def test_written_files_stay_inside_the_projects_root(self, tmp_path: Path) -> None:
        service = CaptionService(tmp_path)
        files = service.write(make_cues(), project_id="demo")
        root = tmp_path.resolve()
        for path_string in (files.srt_path, files.vtt_path):
            assert Path(path_string).resolve().is_relative_to(root)

    def test_unicode_content_survives_the_written_file(self, tmp_path: Path) -> None:
        script = make_script(
            sections=(
                Section(
                    section_id="s1",
                    heading="Unicode",
                    narration="Héllo — ça va? Ünïcode stays readable on disk.",
                ),
            )
        )
        cues = build_cues(segments_from_script(script))
        files = CaptionService(tmp_path).write(cues, project_id="demo")

        content = Path(files.srt_path).read_text(encoding="utf-8")
        assert "Héllo" in content
        assert list(parse_srt(content)) == list(cues)


# --------------------------------------------------------------------------
# Path-safety primitives shared by the media services
# --------------------------------------------------------------------------


class TestPathSafety:
    @pytest.mark.parametrize("bad", ["../evil", "..", "nested/../../out", "/outside"])
    def test_validate_path_rejects_escapes(self, tmp_path: Path, bad: str) -> None:
        with pytest.raises(MediaPathError):
            validate_path(bad, tmp_path)

    @pytest.mark.parametrize("good", ["demo", "demo/captions", "a/b/c.png"])
    def test_validate_path_accepts_contained_paths(self, tmp_path: Path, good: str) -> None:
        resolved = validate_path(good, tmp_path)
        assert resolved.is_relative_to(tmp_path.resolve())

    def test_validate_path_accepts_an_absolute_path_inside_the_root(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "demo" / "file.txt"
        assert validate_path(target, tmp_path) == target.resolve()

    @pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", "C:evil", "", ".", ".."])
    def test_safe_filename_rejects_unsafe_names(self, bad: str) -> None:
        with pytest.raises(MediaPathError):
            safe_filename(bad)

    def test_safe_filename_accepts_plain_names(self) -> None:
        assert safe_filename("captions.srt") == "captions.srt"
