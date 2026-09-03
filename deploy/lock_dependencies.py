"""Print CPU/Linux constraints from the tested interpreter, without installing anything.

Run under the validated Python 3.12 environment. The CPU torch wheel supplies different
metadata from the CUDA wheel, so CUDA-only distributions are deliberately omitted.
The resulting lock is checked by pip check in the image, not assumed valid from generation.

--seed main (default): the base package + [server] extra, run under the main venv, producing
requirements-cpu.lock. --seed presidio-anonymizer: the isolated venv's own tiny closure
(presidio-anonymizer only -- see docs/dependency-release-blocker.md), run under that separate
venv's interpreter, producing requirements-presidio.lock.
"""
import argparse
from importlib.metadata import distribution
from pathlib import Path
import sysconfig
import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", choices=["main", "presidio-anonymizer"], default="main")
args = parser.parse_args()

definition = tomllib.loads(Path("pyproject.toml").read_text())
project = definition["project"]
if args.seed == "main":
    pending = [Requirement(x) for x in project["dependencies"] + project["optional-dependencies"]["server"]
               + definition["build-system"]["requires"]]
else:
    # No build-system.requires here: this venv never builds buffdata itself (or anything
    # else from source -- presidio-anonymizer/cryptography ship as wheels), so setuptools/
    # wheel aren't installed packages in it and shouldn't be required to be.
    pending = [Requirement(x) for x in project["optional-dependencies"]["presidio-anonymizer"]]
resolved, visited = {}, set()
while pending:
    req = pending.pop()
    name = canonicalize_name(req.name)
    if name.startswith(("nvidia-", "cuda-")) or name == "triton":
        continue
    key = (name, tuple(sorted(req.extras)))
    if key in visited:
        continue
    visited.add(key)
    dist = distribution(name)
    if not req.specifier.contains(dist.version, prereleases=True):
        raise RuntimeError(f"Installed {name} does not meet the declared requirement; validate the environment first")
    resolved[name] = dist.version
    for dependency in dist.requires or []:
        child = Requirement(dependency)
        if child.marker is None or any(child.marker.evaluate({"extra": extra}) for extra in {"", *req.extras}):
            pending.append(child)
print("# Python 3.12 / Linux CPU image; exact versions from the validated environment.")
if args.seed == "main":
    print("# Install torch from the official CPU index first; then install this file with --no-deps.")
else:
    print("# Isolated presidio-anonymizer venv; install this file with --no-deps, nothing else.")
for name, version in sorted(resolved.items()):
    print(f"{name}=={version}")
