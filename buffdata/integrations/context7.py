"""Optional Context7 client for current, version-specific documentation."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class Context7Error(RuntimeError):
    """Raised when Context7 configuration or a request fails."""


class Context7Client:
    base_url = "https://context7.com/api/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        opener: Callable[..., Any] = urlopen,
    ):
        self.api_key = api_key or os.getenv("CONTEXT7_API_KEY")
        self._opener = opener

    def _get(self, path: str, **params: str) -> dict[str, Any]:
        if not self.api_key:
            raise Context7Error("CONTEXT7_API_KEY is required for Context7 documentation lookups.")
        url = f"{self.base_url}{path}?{urlencode(params)}"
        request = Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
        )
        try:
            with self._opener(request, timeout=30) as response:
                body = response.read().decode("utf-8")
                content_type = response.headers.get_content_type()
                if content_type == "application/json":
                    return json.loads(body)
                return {"content": body}
        except Exception as exc:
            raise Context7Error(f"Context7 request failed: {exc}") from exc

    def search_libraries(self, query: str) -> dict[str, Any]:
        """Find Context7 library identifiers by name."""
        return self._get("/libs/search", query=query)

    def get_context(self, library_id: str, query: str) -> dict[str, Any]:
        """Retrieve focused, current documentation for a resolved library id."""
        return self._get("/context", libraryId=library_id, query=query)
