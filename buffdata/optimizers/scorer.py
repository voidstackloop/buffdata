import asyncio
import json
import re
from typing import List, Optional, Tuple
from buffdata.engine.client import GeminiClient, LLMClient, ProviderError
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.models.schemas import DatasetItem, QualityAuditBatch, QualityScore

class FastRuleFilter:
    """Heuristic rule-based pre-filter to detect degenerate, corrupted, or low-effort data."""

    def __init__(
        self,
        min_prompt_len: int = 4,
        max_prompt_len: int = 15000,
        min_response_len: int = 5,
        max_response_len: int = 50000,
        max_ngram_repeat_ratio: float = 0.35,
    ):
        self.min_prompt_len = min_prompt_len
        self.max_prompt_len = max_prompt_len
        self.min_response_len = min_response_len
        self.max_response_len = max_response_len
        self.max_ngram_repeat_ratio = max_ngram_repeat_ratio

        self.bad_substrings = [
            "\ufffd", "<|im_end|>", "<|im_start|>", "<|endoftext|>",
            "[TODO:", "[insert code here]", "lorem ipsum"
        ]

    def evaluate(self, prompt: str, response: str) -> Tuple[bool, List[str]]:
        issues = []
        if len(prompt.strip()) < self.min_prompt_len:
            issues.append(f"Prompt too short (< {self.min_prompt_len} chars)")
        if len(prompt) > self.max_prompt_len:
            issues.append(f"Prompt exceeds max length ({len(prompt)} > {self.max_prompt_len})")
        if len(response.strip()) < self.min_response_len:
            issues.append(f"Response too short (< {self.min_response_len} chars)")
        if len(response) > self.max_response_len:
            issues.append(f"Response exceeds max length ({len(response)} > {self.max_response_len})")

        # Check bad substrings / corrupted tokens
        for bad in self.bad_substrings:
            if bad in prompt or bad in response:
                issues.append(f"Contains corrupted token/placeholder: {bad}")

        # Check repetition loops (e.g. model caught in loop)
        words = response.split()
        if len(words) > 30:
            ngrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            unique_ngrams = len(set(ngrams))
            ratio = 1.0 - (unique_ngrams / len(ngrams))
            if ratio > self.max_ngram_repeat_ratio:
                issues.append(f"Degenerate repetition loop detected (repeat ratio: {ratio:.2f})")

        passed = len(issues) == 0
        return passed, issues


SCORER_SYSTEM_PROMPT = """You are an expert AI dataset quality auditor.
Your job is to rigorously evaluate training data samples across 5 core dimensions:
1. Clarity: Readability, grammatical correctness, and clear structure.
2. Factual Accuracy: Factual correctness, absence of hallucinations, valid logic.
3. Reasoning Depth: Step-by-step analytical reasoning, thoroughness, addressing nuance.
4. Instruction Following: Strict compliance with all prompt constraints, formats, and edge cases.
5. Safety: Ensure content is helpful, harmless, and free of toxicity.

Score each dimension from 0.0 (unusable) to 10.0 (state-of-the-art gold standard).
Provide an overall_score (0.0 - 10.0), list any specific issues detected, and provide concise improvement recommendations.
"""

class QualityScorer:
    """Evaluates datasets using heuristic pre-filters and Gemini LLM-as-a-judge."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        limiter: Optional[AsyncRateLimiter] = None,
        model: Optional[str] = None,
    ):
        self.client = client or GeminiClient()
        self.limiter = limiter or AsyncRateLimiter()
        self.model = model or self.client.default_model
        self.rule_filter = FastRuleFilter()
        self.last_audit_metrics = {"remote_batches": 0, "remote_records": 0}

    @staticmethod
    def _content(item: DatasetItem) -> tuple[str, str]:
        prompt, response = item.get_prompt_and_response()
        if not prompt and not response:
            response = item.get_classification_text()
        if not prompt and response:
            prompt = response
        return prompt, response

    async def score_item_async(self, item: DatasetItem) -> DatasetItem:
        """Evaluate a single dataset item and attach QualityScore."""
        prompt, response = self._content(item)

        # 1. Fast heuristic pre-check
        passed_rules, rule_issues = self.rule_filter.evaluate(prompt, response)
        if not passed_rules:
            item.quality_score = QualityScore(
                overall_score=2.0,
                clarity=3.0,
                factual_accuracy=3.0,
                reasoning_depth=1.0,
                instruction_following=2.0,
                is_safe=True,
                issues=rule_issues,
                recommendations="Failed basic heuristic checks: " + "; ".join(rule_issues),
            )
            item.metadata["rule_check_passed"] = False
            return item

        # 2. Gemini LLM-as-a-Judge Evaluation
        eval_prompt = f"""Please evaluate the following training data sample:

[PROMPT / INSTRUCTION]
{prompt}

[RESPONSE]
{response}
"""
        try:
            score: QualityScore = await self.limiter.execute_with_retry(
                lambda: self.client.generate_structured_async(
                    prompt=eval_prompt,
                    response_schema=QualityScore,
                    model=self.model,
                    system_instruction=SCORER_SYSTEM_PROMPT,
                    temperature=0.1,
                )
            )
            item.quality_score = score
            item.metadata["rule_check_passed"] = True
        except ProviderError:
            raise
        except Exception as e:
            item.metadata["scoring_error"] = str(e)
            item.quality_score = QualityScore(
                overall_score=5.0,
                clarity=5.0,
                factual_accuracy=5.0,
                reasoning_depth=5.0,
                instruction_following=5.0,
                is_safe=True,
                issues=[f"Scoring failed: {e}"],
                recommendations="Retry scoring.",
            )

        return item

    async def audit_sample_batch_async(
        self,
        items: List[DatasetItem],
        batch_size: int = 20,
        task_type: Optional[str] = None,
        classes: Optional[List[str]] = None,
    ) -> List[DatasetItem]:
        """Audit a representative sample with many records per structured request."""
        remote_items: List[DatasetItem] = []
        for item in items:
            prompt, response = self._content(item)
            passed_rules, rule_issues = self.rule_filter.evaluate(prompt, response)
            if not passed_rules:
                item.quality_score = QualityScore(
                    overall_score=2.0,
                    clarity=3.0,
                    factual_accuracy=3.0,
                    reasoning_depth=1.0,
                    instruction_following=2.0,
                    is_safe=True,
                    issues=rule_issues,
                    recommendations="Failed basic heuristic checks: " + "; ".join(rule_issues),
                )
                item.metadata["rule_check_passed"] = False
            else:
                remote_items.append(item)

        chunks = [remote_items[index:index + batch_size] for index in range(0, len(remote_items), batch_size)]
        self.last_audit_metrics = {
            "remote_batches": len(chunks),
            "remote_records": len(remote_items),
        }

        async def audit_chunk(chunk: List[DatasetItem]) -> None:
            records = []
            for item in chunk:
                prompt, response = self._content(item)
                records.append({
                    "item_id": item.id,
                    "prompt": prompt[:2500],
                    "response": response[:2500],
                    "labels": item.labels,
                })
            task_context = ""
            if task_type:
                task_context = f"""
These are labeled {task_type} classification records with known classes: {classes or []}.
Judge overall quality by suitability for supervised classification and text/label consistency.
Interpret factual_accuracy as label consistency, reasoning_depth as whether the text contains
enough signal for its label, and instruction_following as schema/label compliance. Do not
penalize a record merely for being short, informal, or lacking an instruction/answer pair.
"""
            audit_prompt = (
                "Audit every record below. Return exactly one entry per item_id, preserve each item_id "
                "verbatim, and apply the scoring rubric independently to each record.\n"
                + task_context
                + "\n"
                + json.dumps(records, ensure_ascii=False)
            )
            try:
                result: QualityAuditBatch = await self.limiter.execute_with_retry(
                    lambda: self.client.generate_structured_async(
                        prompt=audit_prompt,
                        response_schema=QualityAuditBatch,
                        model=self.model,
                        system_instruction=SCORER_SYSTEM_PROMPT,
                        temperature=0.1,
                    )
                )
                by_id = {entry.item_id: entry.score for entry in result.entries}
                for item in chunk:
                    score = by_id.get(item.id)
                    if score is None:
                        item.metadata["scoring_error"] = "Batch audit returned no score for this item_id."
                    else:
                        item.quality_score = score
                        item.metadata["rule_check_passed"] = True
            except ProviderError:
                raise
            except Exception as exc:
                for item in chunk:
                    item.metadata["scoring_error"] = str(exc)

        await asyncio.gather(*(audit_chunk(chunk) for chunk in chunks))
        return items

    async def score_batch_async(
        self,
        items: List[DatasetItem],
        on_progress: Optional[callable] = None,
    ) -> List[DatasetItem]:
        """Score a collection of dataset items concurrently."""
        tasks = []
        for item in items:
            async def wrapped(it=item):
                res = await self.score_item_async(it)
                if on_progress:
                    on_progress()
                return res
            tasks.append(wrapped())

        return await asyncio.gather(*tasks)

    def filter_items(
        self,
        items: List[DatasetItem],
        min_score: float = 7.0,
        drop_failed_rules: bool = True,
    ) -> Tuple[List[DatasetItem], List[DatasetItem]]:
        """Split items into kept (high quality) and dropped (low quality) subsets."""
        kept = []
        dropped = []
        for item in items:
            if drop_failed_rules and item.metadata.get("rule_check_passed") is False:
                dropped.append(item)
                continue
            if item.quality_score and item.quality_score.overall_score >= min_score:
                kept.append(item)
            else:
                dropped.append(item)
        return kept, dropped
