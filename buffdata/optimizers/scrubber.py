"""Local PII detection and redaction."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from buffdata.models.schemas import ChatMessage, DatasetFormat, DatasetItem
from buffdata.plugins import load_pii_recognizer_plugins
from buffdata.security.policy import SecurityPolicy, presidio_anonymizer_python

def _scrub_worker(
    items: List[DatasetItem],
    language: str,
    entities: Optional[List[str]],
) -> List[DatasetItem]:
    """Scrub one process chunk without losing the caller's PII policy."""
    scrubber = PIIScrubber(language=language, entities=entities)
    return [scrubber.scrub_item(it) for it in items]


class PIIScrubber:
    """Use Presidio when available and a deterministic local fallback otherwise."""

    _patterns = {
        "EMAIL_ADDRESS": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
        "PHONE_NUMBER": re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)"),
        "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        "API_KEY": re.compile(r"(?:AIza[0-9A-Za-z-_]{35}|hf_[A-Za-z0-9]{34}|sk-[A-Za-z0-9-_]{48}|sk-proj-[A-Za-z0-9-_]{48,}|sk-ant-[A-Za-z0-9-_]{48,})"),
    }

    IDENTIFIER_ENTITIES = frozenset({
        "API_KEY",
        "CREDIT_CARD",
        "EMAIL_ADDRESS",
        "IBAN_CODE",
        "IP_ADDRESS",
        "PHONE_NUMBER",
        "US_BANK_NUMBER",
        "US_ITIN",
        "US_PASSPORT",
        "US_SSN",
    })

    # Presidio's generic phone recognizer treats slash-separated radio bands and
    # similar numeric product specs as phone numbers. The deterministic fallback
    # below is deliberately stricter for this policy.
    _LOCAL_ONLY_ENTITIES = frozenset({"API_KEY", "PHONE_NUMBER"})

    def __init__(self, language: str = "en", entities: List[str] | None = None):
        self.language = language
        self.entities = sorted(set(entities)) if entities is not None else None
        self.presidio_entities = (
            sorted(set(self.entities) - self._LOCAL_ONLY_ENTITIES)
            if self.entities is not None
            else None
        )
        # analyzer (detection, presidio-analyzer -- no cryptography dependency, always
        # installable) and _anonymize (redaction, presidio-anonymizer -- pulls in an
        # older-pinned cryptography, so it may live in a separate isolated venv instead of
        # in-process) are deliberately independent: a missing anonymizer must not also take
        # down detection, and detection without any way to act on it must not happen at all
        # (see the coupling check at the end of this method).
        self.analyzer = None
        try:
            from presidio_analyzer import AnalyzerEngine

            self.analyzer = AnalyzerEngine()
            for recognizer in load_pii_recognizer_plugins():
                try:
                    self.analyzer.registry.add_recognizer(recognizer)
                except Exception:
                    # A broken plugin recognizer must not disable PII scrubbing entirely.
                    pass
        except Exception:
            self.analyzer = None

        self._anonymize = None  # Callable[[str, list[RecognizerResult]], str] | None
        try:
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import RecognizerResult as AnonymizerResult

            engine = AnonymizerEngine()

            def local_anonymize(text, findings):
                return engine.anonymize(text=text, analyzer_results=[
                    AnonymizerResult(entity_type=f.entity_type, start=f.start, end=f.end, score=f.score)
                    for f in findings]).text

            self._anonymize = local_anonymize
        except Exception:
            # presidio-anonymizer isn't importable in this process -- the expected case once
            # it's split into its own venv (docs/dependency-release-blocker.md). Fall back to
            # the isolated-venv sandbox if one's configured; regex-only redaction otherwise.
            executable = presidio_anonymizer_python()
            if executable:
                self._anonymizer_sandbox = None
                self._anonymizer_executable = executable
                self._anonymize = self._sandboxed_anonymize
        if self._anonymize is None:
            # Detected-but-unredactable entities would otherwise report as redacted in
            # item.metadata["pii"]["entity_counts"] while the raw text is untouched -- so
            # detection without a way to act on it doesn't run at all. Same fallback shape as
            # today: regex-only redaction for the fixed identifier patterns below.
            self.analyzer = None

    def _sandboxed_anonymize(self, text, findings):
        if self._anonymizer_sandbox is None:
            import atexit
            from pathlib import Path
            from buffdata.security.sandbox import PluginSandbox
            worker_path = Path(__file__).resolve().parents[1] / "security" / "anonymizer_worker.py"
            self._anonymizer_sandbox = PluginSandbox(SecurityPolicy(),
                python_executable=self._anonymizer_executable, worker_argv=[str(worker_path)])
            atexit.register(self._anonymizer_sandbox.close)
        response = self._anonymizer_sandbox.call("anonymize", {"text": text,
            "analyzer_results": [{"entity_type": f.entity_type, "start": f.start, "end": f.end, "score": f.score}
                                 for f in findings]})
        return response["text"]

    def scrub_text_with_audit(self, text: str) -> Tuple[str, Dict[str, int]]:
        if not text:
            return text, {}
        counts: Counter[str] = Counter()
        result = text
        if self.analyzer is not None and (self.presidio_entities is None or self.presidio_entities):
            try:
                findings = self.analyzer.analyze(
                    text=text,
                    entities=self.presidio_entities or [],
                    language=self.language,
                )
                # Only recorded once redaction actually succeeds -- counts and the redacted
                # text must never disagree about what was actually removed.
                redacted = self._anonymize(text, findings)
                counts.update(finding.entity_type for finding in findings)
                result = redacted
            except Exception:
                result = text
        for entity, pattern in self._patterns.items():
            if self.entities is not None and entity not in self.entities:
                continue
            matches = pattern.findall(result)
            if matches:
                counts[entity] += len(matches)
                result = pattern.sub(f"<{entity}>", result)
        return result, dict(counts)

    def scrub_text(self, text: str) -> str:
        return self.scrub_text_with_audit(text)[0]

    def scrub_item(self, item: DatasetItem) -> DatasetItem:
        counts: Counter[str] = Counter()

        def scrub(value: str | None) -> str | None:
            if value is None:
                return None
            redacted, found = self.scrub_text_with_audit(value)
            counts.update(found)
            return redacted

        if item.format == DatasetFormat.ALPACA:
            item.instruction = scrub(item.instruction)
            item.input = scrub(item.input)
            item.output = scrub(item.output)
        elif item.format == DatasetFormat.CHAT:
            item.messages = [
                ChatMessage(role=message.role, content=scrub(message.content) or "")
                for message in (item.messages or [])
            ]
        elif item.format == DatasetFormat.DPO:
            item.prompt = scrub(item.prompt)
            item.chosen = scrub(item.chosen)
            item.rejected = scrub(item.rejected)
        elif item.format == DatasetFormat.RAW:
            item.text = scrub(item.text)
        else:
            for key, value in list(item.raw_data.items()):
                if isinstance(value, str):
                    item.raw_data[key] = scrub(value)

        item.metadata["pii"] = {
            "scrubbed": True,
            "entity_counts": dict(counts),
        }
        return item

    def scrub_batch(self, items: List[DatasetItem]) -> List[DatasetItem]:
        import concurrent.futures
        import os
        import math

        if not items:
            return items

        cpu_count = max(1, os.cpu_count() or 1)
        # Avoid heavy multiprocess overhead for small batches
        if len(items) < 100 or cpu_count == 1:
            return [self.scrub_item(item) for item in items]

        chunk_size = math.ceil(len(items) / cpu_count)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        results = []
        worker_count = min(cpu_count, 4, len(chunks))
        import multiprocessing
        # spawn, not the platform default (fork on Linux): a fork duplicates every open file
        # descriptor, including a live plugin-sandbox pipe if one is open in this process, and
        # PEP 446's FD_CLOEXEC only takes effect across exec() -- which fork-mode
        # multiprocessing never calls. Forked workers inheriting that pipe could corrupt its
        # framing or reach the sandbox directly.
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            for chunk_res in executor.map(
                _scrub_worker,
                chunks,
                [self.language] * len(chunks),
                [self.entities] * len(chunks),
            ):
                results.extend(chunk_res)

        return results
