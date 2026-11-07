"""The Researcher: turn a topic into grounded, cited research.

What this agent is
------------------
The first real Creator AI agent, and the first consumer of the Phase 3
architecture. It searches, retrieves and reads, then asks the AI provider
to analyse what it actually read.

The central constraint
----------------------
**Qwen3 is the analysis engine, not the web browser.** The model has no
internet access and is never asked to imply that it does. Retrieval is the
job of :class:`~app.services.research.search.base.SearchProvider` and
:class:`~app.services.research.fetching.base.SourceFetcher`; the model
only ever sees text this agent retrieved and quotes back.

That separation is what makes the grounding chain hold::

    Claim -> source id -> URL -> extracted text the model actually read

Every link is verified by code. A claim citing an identifier that does not
exist has that reference dropped; a claim left with no references is graded
``unsupported`` and handed to the Fact Checker rather than being hidden.

The pipeline
------------
#. **Search** - one query, through the configured provider.
#. **Deduplicate** - drop unsafe URLs, repeated URLs and repeated titles.
#. **Fetch** - bounded and concurrent, with failures recorded not raised.
#. **Extract** - HTML to readable text, dropping scripts and page chrome.
#. **Analyse** - the provider, with reasoning off by default.
#. **Ground** - verify every reference, recompute confidence.
#. **Report** - a typed result carrying its own partial-success counts.

Partial success is the normal case, not an edge case. Six sources
requested, four retrieved and two blocked is a usable result: it returns
``sources_found=4``, ``sources_failed=2`` and the reason for each. Only
when the evidence is too thin to analyse does the agent report
``success=False`` while still completing, following the Phase 3 distinction
between a run that failed and a run that produced a negative verdict.

Concurrency
-----------
Source fetching runs with bounded parallelism. That is not in tension with
the Phase 3 decision that agent execution is sequential: that decision is
about *generation*, where two concurrent calls would contend for one CPU
and 16 GB of RAM. Fetching web pages is idle network waiting, and
serialising it would add seconds of pure latency per research run. The AI
call remains strictly one at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import anyio

from app.agents.base import AgentContext, AgentError, AgentOutput, BaseAgent, Capability, utc_now
from app.agents.researcher.exceptions import (
    ResearchAIError,
    ResearchError,
    ResearchGroundingError,
    ResearchInputError,
    ResearchErrorCode,
)
from app.agents.researcher.grounding import (
    confidence_counts,
    ground_analysis,
    verify_references,
)
from app.agents.researcher.prompts import (
    NO_EVIDENCE_EXPLANATION,
    SOURCE_ID_TEMPLATE,
    format_evidence_block,
)
from app.agents.researcher.schemas import (
    Confidence,
    ResearchAnalysis,
    ResearchClaim,
    ResearchRequest,
    ResearchResult,
    ResearchSource,
    SourceFailure,
    SourceOutcome,
)
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.prompts.builder import PromptBuilder
from app.services.ai.provider import (
    AIProviderError,
    GenerationOptions,
    Message,
    StructuredOutputError,
)
from app.services.research.dedupe import deduplicate
from app.services.research.extraction.base import (
    ContentExtractor,
    ExtractedContent,
    ExtractionError,
)
from app.services.research.fetching.base import FetchError, FetchedDocument, SourceFetcher
from app.services.research.search.base import (
    SearchError,
    SearchProvider,
    SearchQuery,
    SearchResult,
)

logger = get_logger("agents.researcher")

#: Context key holding a pre-validated :class:`ResearchRequest`. The API
#: puts the parsed request here so it is validated once, at the boundary,
#: rather than re-parsed inside the agent.
REQUEST_CONTEXT_KEY = "research_request"


@dataclass(frozen=True, slots=True)
class _RetrievedSource:
    """A source that was fetched, extracted and is ready as evidence."""

    source_id: str
    result: SearchResult
    document: FetchedDocument
    content: ExtractedContent


class ResearcherAgent(BaseAgent):
    """Research a topic from live web sources and cite them.

    Every collaborator is injected, and every default is resolved lazily.
    Constructing the agent therefore performs no I/O, which keeps the
    registry cheap to build at import time and makes the agent trivially
    testable with fakes.
    """

    name = "researcher"
    version = "1.0.0"
    description = (
        "Searches the web, reads the sources it finds, and returns "
        "structured research whose claims cite the sources that support them."
    )

    capabilities = frozenset(
        {
            Capability.STRUCTURED_OUTPUT,
            # Retrieval is a network operation, so a transient outage is
            # worth another attempt.
            Capability.RETRYABLE,
        }
    )

    def __init__(
        self,
        provider: AIProvider | None = None,
        *,
        search_provider: SearchProvider | None = None,
        fetcher: SourceFetcher | None = None,
        extractor: ContentExtractor | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(provider)
        self._search_provider = search_provider
        self._fetcher = fetcher
        self._extractor = extractor
        self._settings = settings

    # ---------- Collaborators ----------

    @property
    def search_provider(self) -> SearchProvider:
        """The configured search backend, resolved on first access."""
        if self._search_provider is None:
            from app.dependencies import get_search_provider

            self._search_provider = get_search_provider()
        return self._search_provider

    @property
    def fetcher(self) -> SourceFetcher:
        """The configured source fetcher, resolved on first access."""
        if self._fetcher is None:
            from app.dependencies import get_source_fetcher

            self._fetcher = get_source_fetcher()
        return self._fetcher

    @property
    def extractor(self) -> ContentExtractor:
        """The configured content extractor, resolved on first access."""
        if self._extractor is None:
            from app.dependencies import get_content_extractor

            self._extractor = get_content_extractor()
        return self._extractor

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    # ---------- Agent ----------

    async def run(self, context: AgentContext, messages: Sequence[Message]) -> AgentOutput:
        """Research the requested topic.

        Args:
            context: carries a validated
                :class:`~app.agents.researcher.schemas.ResearchRequest`
                under ``research_request``. When absent, the topic is
                taken from the last user message instead, which is what
                lets the agent be driven through the Phase 3 runner with
                nothing but a prompt.
            messages: the incoming prompt. Used only as a fallback topic;
                the analysis prompt is built here, because it cannot
                exist until the evidence has been retrieved.

        Returns:
            An :class:`~app.agents.base.AgentOutput`. A result that
            completed with too little evidence reports
            ``success=False`` and still carries the partial data, leaving
            the status as COMPLETED.
        """
        request = self._resolve_request(context, messages)
        settings = self.settings

        limit = self._source_limit(request, settings)
        candidates = await self._search(request, limit=limit)
        usable, failures = await self._collect_evidence(request, candidates)

        result = self._build_result(
            request,
            candidates=candidates,
            usable=usable,
            failures=failures,
            search_provider=self.search_provider.name,
        )

        if len(usable) < settings.min_usable_sources:
            # Completed, not failed: the pipeline worked and the web had
            # too little to offer. Distinct from an execution error.
            code = (
                ResearchErrorCode.NO_SOURCES
                if not usable
                else ResearchErrorCode.INSUFFICIENT_SOURCES
            )
            logger.info(
                "research_insufficient_sources found=%d required=%d code=%s",
                len(usable),
                settings.min_usable_sources,
                code.value,
            )
            return AgentOutput(
                success=False,
                data=result,
                error=(
                    "Not enough usable sources were retrieved to analyse "
                    f"({len(usable)} of {limit} requested)."
                ),
                metadata={
                    "code": code.value,
                    "retryable": True,
                    "sources_found": len(usable),
                    "sources_failed": len(failures),
                },
            )

        analysis = await self._analyse(request, usable, context)
        grounded = ground_analysis(analysis, self._sources(usable), research_topic=request.topic)

        dangling = verify_references(
            self._sources(usable),
            grounded.key_points,
            grounded.claims,
            grounded.conflicts,
            grounded.uncertainties,
            grounded.statistics,
        )
        if dangling:
            # Grounding filters every collection it knows about, so this
            # can only fire if a schema grew a field that grounding was
            # not updated for. Better to say so than to ship a citation
            # nobody can resolve.
            raise ResearchGroundingError(
                "The analysis referenced sources that were not retrieved."
            )

        return AgentOutput.ok(
            data=result.model_copy(
                update={
                    "summary": grounded.summary,
                    "key_points": tuple(grounded.key_points),
                    "claims": tuple(grounded.claims),
                    "conflicts": tuple(grounded.conflicts),
                    "uncertainties": tuple(grounded.uncertainties),
                    "statistics": tuple(grounded.statistics),
                    "confidence_counts": confidence_counts(grounded.claims),
                    "unsupported_claims": sum(
                        1
                        for claim in grounded.claims
                        if claim.confidence is Confidence.UNSUPPORTED
                    ),
                    "generated_at": utc_now(),
                    "sufficient": True,
                }
            ),
            sources_found=len(usable),
            sources_failed=len(failures),
            search_provider=self.search_provider.name,
        )

    # ---------- Stages ----------

    def _resolve_request(
        self, context: AgentContext, messages: Sequence[Message]
    ) -> ResearchRequest:
        """Find the request in the context, or build one from the prompt.

        The context wins because the API has already validated it. The
        prompt fallback exists so the agent is usable through the Phase 3
        runner with nothing but a message, and so an operator can trigger
        it from the prompt library.
        """
        supplied = context.option(REQUEST_CONTEXT_KEY)
        if isinstance(supplied, ResearchRequest):
            return supplied

        if isinstance(supplied, dict):
            try:
                return ResearchRequest.model_validate(supplied)
            except Exception as exc:  # noqa: BLE001 - re-raised as a research error
                raise ResearchInputError(
                    "The research request could not be understood."
                ) from exc

        topic = ""
        for message in reversed(list(messages)):
            if message.role == "user" and message.content.strip():
                topic = message.content.strip()
                break

        if not topic:
            raise ResearchInputError(
                "No research topic was supplied."
            )

        try:
            return ResearchRequest(topic=topic)
        except Exception as exc:  # noqa: BLE001 - re-raised as a research error
            raise ResearchInputError("The research topic is not valid.") from exc

    async def _search(
        self, request: ResearchRequest, *, limit: int
    ) -> list[SearchResult]:
        """Ask the search backend for candidates.

        Raises:
            ResearchAIError is not used here; a search failure that leaves
                nothing to work with is reported as an empty result plus a
                failure entry, which keeps partial success working.
        """
        query = SearchQuery(
            terms=request.topic,
            depth=request.search_depth,
            language=request.language,
        )

        try:
            results = await self.search_provider.search(query, limit=limit)
        except SearchError as exc:
            logger.warning("research_search_failed code=%s message=%s", exc.code, exc.message)
            return []
        except AgentError:
            raise
        except Exception as exc:  # noqa: BLE001 - a bug must not look like "no results"
            raise ResearchError(
                "The search backend failed unexpectedly."
            ) from exc

        logger.info(
            "research_search_complete provider=%s candidates=%d limit=%d",
            self.search_provider.name,
            len(results),
            limit,
        )
        return results

    async def _collect_evidence(
        self, request: ResearchRequest, candidates: Sequence[SearchResult]
    ) -> tuple[list[_RetrievedSource], list[SourceFailure]]:
        """Deduplicate, fetch and extract, recording failures as they happen.

        One dead website must not end the run, so every failure is caught
        and turned into a :class:`SourceFailure`. The caller gets both the
        usable sources and an account of what went missing.
        """
        settings = self.settings
        limit = self._source_limit(request, settings)

        report = deduplicate(candidates, limit=limit)
        failures = [
            SourceFailure(
                url=result.url,
                domain=result.domain,
                reason="Duplicate of another source.",
                code="duplicate_source",
                outcome=SourceOutcome.SKIPPED,
            )
            for result in report.dropped
        ] + [
            SourceFailure(
                url=result.url,
                domain=result.domain,
                reason="Not a permitted source URL.",
                code="unsafe_url",
                outcome=SourceOutcome.SKIPPED,
            )
            for result in report.rejected_unsafe
        ]

        selected = report.kept
        if not selected:
            return [], failures

        collected: dict[str, _RetrievedSource] = {}
        failed: dict[str, SourceFailure] = {}

        # Bounded parallelism. Each source's own timeout is enforced inside
        # the fetcher, and the semaphore keeps the number of open
        # connections small on a laptop.
        limiter = anyio.Semaphore(settings.max_concurrent_source_fetches)

        async def retrieve(index: int, candidate: SearchResult) -> None:
            async with limiter:
                try:
                    document = await self.fetcher.fetch(candidate.url)
                    content = self.extractor.extract(document)
                except FetchError as exc:
                    failed[candidate.url] = SourceFailure(
                        url=candidate.url,
                        domain=candidate.domain,
                        reason=exc.message,
                        code=exc.code,
                        outcome=SourceOutcome.FAILED,
                    )
                    return
                except ExtractionError as exc:
                    failed[candidate.url] = SourceFailure(
                        url=candidate.url,
                        domain=candidate.domain,
                        reason=exc.message,
                        code=exc.code,
                        outcome=SourceOutcome.EMPTY,
                    )
                    return
                except Exception as exc:  # noqa: BLE001 - one source must not end the run
                    failed[candidate.url] = SourceFailure(
                        url=candidate.url,
                        domain=candidate.domain,
                        reason=f"Retrieval failed: {type(exc).__name__}.",
                        code="fetch_error",
                        outcome=SourceOutcome.FAILED,
                    )
                    return

                if not content.is_usable:
                    # Too little text to be evidence. Counted as empty
                    # rather than failed: nothing went wrong, the page was
                    # just not worth citing.
                    failed[candidate.url] = SourceFailure(
                        url=candidate.url,
                        domain=candidate.domain,
                        reason="The page contained too little text to use.",
                        code="empty_content",
                        outcome=SourceOutcome.EMPTY,
                    )
                    return

                collected[candidate.url] = _RetrievedSource(
                    source_id=SOURCE_ID_TEMPLATE.format(index=index),
                    result=candidate,
                    document=document,
                    content=content,
                )

        async with anyio.create_task_group() as group:
            for index, candidate in enumerate(selected, start=1):
                group.start_soon(retrieve, index, candidate)

        # Results are rebuilt in search-rank order. The task group finishes
        # in completion order, which would otherwise shuffle the evidence
        # and make the output unstable between identical runs.
        ordered = [
            collected[candidate.url]
            for candidate in selected
            if candidate.url in collected
        ]

        ordered_failures = [
            failed[candidate.url]
            for candidate in selected
            if candidate.url in failed
        ]

        logger.info(
            "research_retrieval_complete requested=%d usable=%d failed=%d",
            len(selected),
            len(ordered),
            len(ordered_failures),
        )

        return ordered, failures + ordered_failures

    async def _analyse(
        self, request: ResearchRequest, usable: Sequence[_RetrievedSource], context: AgentContext
    ) -> ResearchAnalysis:
        """Send the retrieved evidence to the provider for analysis.

        The provider's blocking ``generate`` is moved onto a worker thread
        so the event loop keeps serving other requests while a local model
        thinks. It is called synchronously rather than pretended to be
        async, and exactly once per run.
        """
        messages = self._build_analysis_messages(request, usable)
        options = self._analysis_options(context)

        logger.info(
            "research_analysis_start model=%s sources=%d think=%s",
            self.provider.name,
            len(usable),
            options.think,
        )

        try:
            analysis = await anyio.to_thread.run_sync(
                lambda: self.provider.generate_structured(
                    messages, ResearchAnalysis, options=options
                )
            )
        except StructuredOutputError as exc:
            logger.warning("research_analysis_invalid_output reason=%s", exc.message)
            raise ResearchGroundingError(
                "The model did not return usable research output."
            ) from exc
        except AIProviderError as exc:
            logger.warning("research_analysis_provider_error code=%s", exc.code)
            raise ResearchAIError(
                "The AI provider could not analyse the retrieved sources."
            ) from exc

        logger.info(
            "research_analysis_complete claims=%d conflicts=%d uncertainties=%d",
            len(analysis.claims),
            len(analysis.conflicts),
            len(analysis.uncertainties),
        )
        return analysis

    # ---------- Assembly ----------

    def _source_limit(self, request: ResearchRequest, settings: Settings) -> int:
        """How many sources this request may use.

        Three caps apply and the smallest wins: the depth's preference, the
        caller's ``max_sources``, and the deployment-wide maximum. A
        caller cannot ask for more pages than the machine is configured to
        retrieve.
        """
        return max(
            1,
            min(request.effective_source_limit, settings.max_research_sources),
        )

    def _analysis_options(self, context: AgentContext) -> GenerationOptions:
        """Generation options for the analysis call.

        Reasoning is inherited from the provider's deployment defaults,
        which is the whole point on the target hardware: with Qwen3's
        reasoning on, one analysis measured 55-88 seconds against 0.9-1.3
        seconds off, because the model enables it by default otherwise.

        A caller may still override it per run by putting ``"thinking"`` in
        the agent context config, which is how a slow, careful analysis can
        be requested for a topic that warrants one without changing the
        deployment default.

        Nothing here names a backend, so a cloud provider receives exactly
        the same options.
        """
        override = context.option("thinking")
        if isinstance(override, bool):
            return self.resolve_options(GenerationOptions(think=override))
        return self.resolve_options()

    def _build_analysis_messages(
        self, request: ResearchRequest, usable: Sequence[_RetrievedSource]
    ) -> list[Message]:
        """Assemble the analysis prompt from the system and user templates."""
        from app.agents.researcher.prompts import (
            RESEARCH_SYSTEM_PROMPT,
            RESEARCH_USER_PROMPT,
        )

        evidence = format_evidence_block(
            [
                (source.source_id, source.result.url, source.content.text)
                for source in usable
            ],
            max_characters=self.settings.max_extracted_text * len(usable),
        )

        builder = PromptBuilder()
        builder.with_system_template(RESEARCH_SYSTEM_PROMPT)

        user_text = RESEARCH_USER_PROMPT.render(
            topic=request.topic,
            depth=request.search_depth.value,
            source_count=str(len(usable)),
            evidence=evidence or NO_EVIDENCE_EXPLANATION,
        ).text
        builder.with_user(user_text)

        return builder.build()

    def _sources(self, usable: Sequence[_RetrievedSource]) -> tuple[ResearchSource, ...]:
        """Convert retrieved sources into their citable, public form."""
        return tuple(
            ResearchSource(
                id=source.source_id,
                title=source.content.title or source.result.title,
                url=source.result.url,
                domain=source.result.domain,
                snippet=source.result.snippet[:600],
                retrieved_at=source.document.retrieved_at,
                text_length=len(source.content.text),
                truncated=source.content.truncated,
            )
            for source in usable
        )

    def _build_result(
        self,
        request: ResearchRequest,
        *,
        candidates: Sequence[SearchResult],
        usable: Sequence[_RetrievedSource],
        failures: Sequence[SourceFailure],
        search_provider: str,
    ) -> ResearchResult:
        """Build the grounded result skeleton, before the analysis exists.

        Built before the model is called so that a run with too little
        evidence still returns a complete, typed object describing what was
        attempted and what came back.
        """
        sources = self._sources(usable)
        claims: tuple[ResearchClaim, ...] = ()

        return ResearchResult(
            topic=request.topic,
            summary="",
            claims=claims,
            sources=sources,
            sources_requested=self._source_limit(request, self.settings),
            sources_found=len(sources),
            sources_failed=len(failures),
            failed_sources=tuple(failures),
            search_query=request.topic,
            search_provider=search_provider,
            search_depth=request.search_depth,
            language=request.language,
            distinct_domains=len({source.domain for source in sources if source.domain}),
            confidence_counts={},
            unsupported_claims=0,
            warnings=self._warnings(request, len(candidates), len(usable)),
            sufficient=False,
            generated_at=utc_now(),
        )

    def _warnings(
        self, request: ResearchRequest, candidates: int, usable: int
    ) -> tuple[str, ...]:
        """Client-safe notes about what was and was not honoured.

        The forward-compatible request fields are accepted by the schema
        but have no backend behaviour yet. Saying so in the response is
        the honest option: silently ignoring a field a caller set is how a
        user concludes the filter worked.
        """
        warnings: list[str] = []

        if request.date_range:
            warnings.append("date_range is accepted but not yet applied to search.")
        if request.region:
            warnings.append("region is accepted but not yet applied to search.")
        if request.preferred_domains:
            warnings.append(
                "preferred_domains is accepted but not yet used to rank sources."
            )
        if request.excluded_domains:
            warnings.append(
                "excluded_domains is accepted but not yet used to filter sources."
            )
        if not candidates:
            warnings.append("The search backend returned no candidates.")
        elif not usable:
            warnings.append("No candidate source could be retrieved.")

        return tuple(warnings)
