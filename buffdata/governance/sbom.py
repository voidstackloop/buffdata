"""Software Bill of Materials generation -- CycloneDX 1.5 JSON.

Built from the packages actually installed in the running environment
(importlib.metadata, stdlib only) rather than shelling out to an external SBOM tool whose
exact output this project has no way to verify in this environment. This lists what's really
installed where buffdata runs, which is what a vendor security review actually wants -- not
a static list transcribed by hand that silently drifts the moment a dependency updates.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distributions
from importlib.metadata import version as pkg_version
from typing import Any

CYCLONEDX_SPEC_VERSION = "1.5"


def _normalize_purl_name(name: str) -> str:
    """PyPI package URLs use the lowercased name with runs of separators collapsed to a
    single hyphen -- e.g. "Pillow_SIMD" -> "pillow-simd"."""
    return re.sub(r"[-_.]+", "-", name).strip("-").lower()


def generate_sbom(*, root_package: str = "buffdata") -> dict[str, Any]:
    """A CycloneDX 1.5 SBOM document (as a plain dict, ready for json.dumps) listing every
    distribution importlib.metadata can see in the current environment except the root
    package itself, which is described in metadata.component instead.
    """
    try:
        root_version = pkg_version(root_package)
    except PackageNotFoundError:
        root_version = "0.0.0"

    components: list[dict[str, str]] = []
    seen: set[str] = set()
    for dist in distributions():
        name = (dist.metadata.get("Name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key == root_package.lower() or key in seen:
            continue
        seen.add(key)
        version = dist.version or "0.0.0"
        components.append({
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{_normalize_purl_name(name)}@{version}",
        })
    components.sort(key=lambda component: component["name"].lower())

    return {
        "bomFormat": "CycloneDX",
        "specVersion": CYCLONEDX_SPEC_VERSION,
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": {"type": "application", "name": root_package, "version": root_version},
        },
        "components": components,
    }
