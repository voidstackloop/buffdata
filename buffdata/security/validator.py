import re
import urllib.parse
import ipaddress
import socket

def is_safe_url(url: str) -> bool:
    """Validate URL to prevent SSRF and local file reads."""
    try:
        from buffdata.security.policy import check_network_url
        check_network_url(url, ordinary_import=True)
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
            
        hostname = parsed.hostname
        if not hostname:
            return False

        # Check if it's an IP address
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                return False
        except ValueError:
            pass # Not an IP, it's a domain name

        # Block localhost/internal domain resolutions
        if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0"):
            return False

        return True
    except Exception:
        return False

class SecretMasker:
    """Mask API keys and tokens in strings."""
    @staticmethod
    def mask(text: str) -> str:
        if not text:
            return text
        text = re.sub(r'(?i)(bearer\s+)[^\s\"\'<>]+', r'\1[REDACTED]', text)
        text = re.sub(r'(?:sk-(?:ant-|proj-)?|ctx7sk-|hf_|AIza)[A-Za-z0-9_-]{8,}', '[REDACTED]', text)
        text = re.sub(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', '[REDACTED]', text)
        text = re.sub(r'(?i)([?&](?:token|key|api_key|access_token|signature)=)[^&\s]+', r'\1[REDACTED]', text)
        # Mask anything that looks like a JWT or long alphanumeric token (very basic heuristic)
        # e.g., AIzaSy... (Google), hf_... (HuggingFace)
        text = re.sub(r'(AIza[0-9A-Za-z-_]{35})', 'AIza***MASKED***', text)
        text = re.sub(r'(hf_[A-Za-z0-9]{34})', 'hf_***MASKED***', text)
        # OpenAI keys
        text = re.sub(r'(sk-[A-Za-z0-9-_]{48})', 'sk-***MASKED***', text)
        text = re.sub(r'(sk-proj-[A-Za-z0-9-_]{48,})', 'sk-proj-***MASKED***', text)
        # Anthropic keys
        text = re.sub(r'(sk-ant-[A-Za-z0-9-_]{48,})', 'sk-ant-***MASKED***', text)
        return text
