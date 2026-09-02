"""Local PII detection and redaction."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from buffdata.models.schemas import ChatMessage, DatasetFormat, DatasetItem
from buffdata.plugins import load_pii_recognizer_plugins

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
        self.analyzer = None
        self.anonymizer = None
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            for recognizer in load_pii_recognizer_plugins():
                try:
                    self.analyzer.registry.add_recognizer(recognizer)
                except Exception:
                    # A broken plugin recognizer must not disable PII scrubbing entirely.
                    pass
        except Exception:
            # The fallback still guarantees local redaction for common identifiers.
            pass

    def scrub_text_with_audit(self, text: str) -> Tuple[str, Dict[str, int]]:
        if not text:
            return text, {}
        counts: Counter[str] = Counter()
        result = text
        if (
            self.analyzer is not None
            and self.anonymizer is not None
            and (self.presidio_entities is None or self.presidio_entities)
        ):
            try:
                findings = self.analyzer.analyze(
                    text=text,
                    entities=self.presidio_entities or [],
                    language=self.language,
                )
                counts.update(finding.entity_type for finding in findings)
                result = self.anonymizer.anonymize(text=text, analyzer_results=findings).text
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
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            for chunk_res in executor.map(
                _scrub_worker,
                chunks,
                [self.language] * len(chunks),
                [self.entities] * len(chunks),
            ):
                results.extend(chunk_res)

        return results
