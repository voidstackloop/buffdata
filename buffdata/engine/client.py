import asyncio
import os
from typing import Any, Dict, List, Optional, Type, TypeVar
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

T = TypeVar("T", bound=BaseModel)

class GeminiClient:
    """High-performance client for Google Gemini API models using google-genai."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-3.7-flash",
        embedding_model: str = "text-embedding-004",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.default_model = default_model
        self.embedding_model = embedding_model
        self._client = None
        self._mock_mode = False

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                # If SDK fails to init, will raise during call
                self._client = None
        else:
            self._mock_mode = True

    @property
    def client(self):
        if self._client is None and not self._mock_mode:
            from google import genai
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not set. Please set the environment variable or pass api_key.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        """Generate response adhering strictly to a Pydantic schema using structured output."""
        target_model = model or self.default_model

        if self._mock_mode or not self.api_key:
            # Fallback mock for testing / validation when key not provided
            return self._mock_structured(response_schema)

        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self.client.models.generate_content(
            model=target_model,
            contents=prompt,
            config=config,
        )

        # Parse structured output from text or parsed object
        if hasattr(response, "parsed") and response.parsed is not None:
            return response.parsed
        return response_schema.model_validate_json(response.text)

    async def generate_structured_async(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        """Async wrapper for generate_structured."""
        return await asyncio.to_thread(
            self.generate_structured,
            prompt=prompt,
            response_schema=response_schema,
            model=model,
            system_instruction=system_instruction,
            temperature=temperature,
        )

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate free-form text response."""
        target_model = model or self.default_model
        if self._mock_mode or not self.api_key:
            return f"[Simulated Gemini response to: {prompt[:40]}...]"

        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self.client.models.generate_content(
            model=target_model,
            contents=prompt,
            config=config,
        )
        return response.text

    async def generate_text_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Async wrapper for generate_text."""
        return await asyncio.to_thread(
            self.generate_text,
            prompt=prompt,
            model=model,
            system_instruction=system_instruction,
            temperature=temperature,
        )

    def embed_texts(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate dense embeddings using text-embedding-004."""
        target_model = model or self.embedding_model
        if self._mock_mode or not self.api_key:
            import numpy as np
            return [np.random.randn(768).tolist() for _ in texts]

        results = []
        for text in texts:
            response = self.client.models.embed_content(
                model=target_model,
                contents=text,
            )
            results.append(response.embedding.values)
        return results

    async def embed_texts_async(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Async wrapper for embed_texts."""
        return await asyncio.to_thread(self.embed_texts, texts=texts, model=model)

    def _mock_structured(self, schema_cls: Type[T]) -> T:
        """Generate plausible mock data when offline."""
        from buffdata.models.schemas import QualityScore, RefinementResult, EvolutionResult, PreferenceResult
        if schema_cls == QualityScore:
            return QualityScore(
                overall_score=8.5,
                clarity=9.0,
                factual_accuracy=8.5,
                reasoning_depth=8.0,
                instruction_following=9.0,
                is_safe=True,
                issues=["Minor brevity"],
                recommendations="Expand the reasoning explanation.",
            )
        elif schema_cls == RefinementResult:
            return RefinementResult(
                refined_prompt="Clarified prompt with clear instructions.",
                refined_response="High-quality refined response with step-by-step reasoning and clean formatting.",
                reasoning_added=True,
                formatting_fixed=True,
                artifacts_removed=["Certainly! Here is your answer:"],
                explanation_of_changes="Removed AI preamble and formatted code.",
            )
        elif schema_cls == EvolutionResult:
            return EvolutionResult(
                evolved_prompt="Complexified prompt with multi-step constraints and edge cases.",
                evolved_response="Detailed response answering the evolved prompt with deep analytical rigor.",
                evolution_type="deepen_reasoning",
                complexity_score=8.5,
            )
        elif schema_cls == PreferenceResult:
            return PreferenceResult(
                prompt="Explain how backpropagation works.",
                chosen="Backpropagation computes gradients using the chain rule across network layers...",
                rejected="Backpropagation is just training a model with weights and biases without formulas.",
                rejection_reason="Rejected answer lacks mathematical rigor and omits chain rule explanation.",
            )
        return schema_cls.model_construct()
