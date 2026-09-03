"""Bounded public HTTP fetch with a fresh pinned resolver at every redirect."""
from urllib.parse import urljoin, urlsplit
import socket

from buffdata.security.policy import SecurityError, check_network_url, current_context


def install_worker_network_guard(context):
    """Irreversible per-child socket guard, including third-party download clients.

    This is defense in depth for trusted Python dependencies, not a native-code sandbox.
    Containers additionally restrict actual egress through an allowlisted proxy.
    """
    import ipaddress
    import os
    import sys
    hosts, addresses = set(), set()
    if context.network != "strict":
        urls = list(context.policy.local_endpoints)
        if context.network == "unrestricted":
            urls += ["https://" + host for host in context.policy.allowed_hosts]
        for url in urls:
            host = urlsplit(url).hostname
            try:
                ips = check_network_url(url)
            except SecurityError:
                continue
            hosts.add(host)
            addresses.update(ips)
        # Proxy endpoint is deployer-controlled; users cannot set worker environments.
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if proxy:
            host = urlsplit(proxy).hostname
            hosts.add(host)
            addresses.update(row[4][0] for row in socket.getaddrinfo(host, urlsplit(proxy).port or 3128))

    def audit(event, args):
        if event == "socket.getaddrinfo":
            host = args[0].decode() if isinstance(args[0], bytes) else args[0]
            if host not in hosts and host not in addresses:
                raise SecurityError("Worker DNS destination is not approved")
        elif event in {"socket.connect", "socket.sendto"}:
            address = args[1] if event == "socket.connect" else args[-1]
            # Local Unix sockets permit OS keyrings; they are not network destinations.
            if isinstance(address, tuple) and address[0] not in addresses:
                raise SecurityError("Worker network destination is not approved")
    sys.addaudithook(audit)


async def fetch_public(url: str, *, max_bytes: int = 16 * 1024**2) -> bytes:
    import aiohttp
    from aiohttp.abc import AbstractResolver

    class PinnedResolver(AbstractResolver):
        def __init__(self, addresses):
            self.addresses = addresses

        async def resolve(self, host, port=0, family=socket.AF_UNSPEC):
            return [dict(hostname=host, host=ip, port=port,
                         family=socket.AF_INET6 if ":" in ip else socket.AF_INET,
                         proto=0, flags=0) for ip in self.addresses]

        async def close(self):
            pass

    for _ in range(6):
        addresses = check_network_url(url, ordinary_import=True)
        connector = aiohttp.TCPConnector(resolver=PinnedResolver(addresses), use_dns_cache=False)
        async with aiohttp.ClientSession(connector=connector, trust_env=False,
                                        timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(url, allow_redirects=False) as response:
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise SecurityError("Redirect has no destination")
                    target = urljoin(url, location)
                    if urlsplit(url).scheme == "https" and urlsplit(target).scheme != "https":
                        raise SecurityError("HTTPS downgrade is forbidden")
                    url = target
                    continue
                response.raise_for_status()
                chunks, size = [], 0
                async for block in response.content.iter_chunked(65536):
                    size += len(block)
                    if size > max_bytes:
                        raise SecurityError("Remote document exceeds the size limit")
                    chunks.append(block)
                return b"".join(chunks)
    raise SecurityError("Too many redirects")
