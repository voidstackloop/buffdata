"""Optional Context7 client for current, version-specific documentation."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen, build_opener, HTTPRedirectHandler


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise Context7Error("Context7 redirects are not accepted")


class Context7Error(RuntimeError):
    """Raised when Context7 configuration or a request fails."""


class Context7Client:
    base_url = "https://context7.com/api/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        opener: Optional[Callable[..., Any]] = None,
    ):
        from buffdata.engine.secrets import get_default_secret_resolver
        from buffdata.security.policy import remember_secret
        self.api_key = remember_secret(api_key or get_default_secret_resolver().get("CONTEXT7_API_KEY"))
        self._opener = opener or build_opener(_NoRedirect()).open

    def _get(self, path: str, **params: str) -> dict[str, Any]:
        if not self.api_key:
            raise Context7Error("CONTEXT7_API_KEY is required for Context7 documentation lookups.")
        url = f"{self.base_url}{path}?{urlencode(params)}"
        from buffdata.security.policy import check_network_url, current_context
        if current_context():
            check_network_url(url)
        request = Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
        )
        try:
            with self._opener(request, timeout=30) as response:
                raw = response.read(16 * 1024**2 + 1)
                if len(raw) > 16 * 1024**2:
                    raise Context7Error("Documentation response exceeds size limit")
                body = raw.decode("utf-8")
                content_type = response.headers.get_content_type()
                if content_type == "application/json":
                    return json.loads(body)
                return {"content": body}
        except Exception as exc:
            from buffdata.security.policy import sanitize
            raise Context7Error(sanitize(f"Context7 request failed: {exc}")) from None

    def search_libraries(self, query: str) -> dict[str, Any]:
        """Find Context7 library identifiers by name."""
        return self._get("/libs/search", query=query)

    def get_context(self, library_id: str, query: str) -> dict[str, Any]:
        """Retrieve focused, current documentation for a resolved library id."""
        return self._get("/context", libraryId=library_id, query=query)
