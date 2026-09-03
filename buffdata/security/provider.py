"""Preserve successful requests/responses; fail closed on unsafe SDK exceptions."""
import inspect


class SafeProvider:
    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        value = getattr(self._client, name)
        if not callable(value) or not name.startswith(("generate_", "embed_")):
            return value
        from buffdata.engine.client import ProviderError
        if inspect.iscoroutinefunction(value):
            async def guarded(*args, **kwargs):
                try:
                    return await value(*args, **kwargs)
                except Exception:
                    raise ProviderError("Provider request failed; sensitive exception details omitted") from None
            return guarded
        def guarded(*args, **kwargs):
            try:
                return value(*args, **kwargs)
            except Exception:
                raise ProviderError("Provider request failed; sensitive exception details omitted") from None
        return guarded
