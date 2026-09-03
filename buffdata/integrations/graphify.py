"""Local Graphify knowledge-graph integration for repository navigation."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import os
from buffdata.security.policy import check_path, current_context, sanitize, SecurityError
from typing import Optional


class GraphifyError(RuntimeError):
    """Raised when Graphify is not installed or a graph operation fails."""


class GraphifyManager:
    def __init__(self, command: Optional[str] = None):
        self.command = command or shutil.which("graphify")

    def _run(self, *arguments: str, cwd: Path) -> str:
        check_path(cwd, write=arguments[0] == "extract")
        ctx = current_context()
        if arguments[0] == "query" and ctx and ctx.network != "unrestricted":
            raise SecurityError("Graphify query may contact a provider and requires unrestricted execution")
        if not self.command:
            raise GraphifyError(
                "Graphify is not installed. Install the optional dependency with: pip install 'buffdata[productivity]'"
            )
        result = subprocess.run(
            [self.command, *arguments],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=ctx.policy.subprocess_seconds if ctx else 120,
            env={key: value for key, value in os.environ.items() if key in {
                "PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP",
            }},
        )
        if result.returncode:
            message = sanitize((result.stderr or result.stdout).strip())
            raise GraphifyError(message or f"Graphify exited with status {result.returncode}.")
        return result.stdout.strip()

    def build(self, project_path: Path | str) -> tuple[Path, str]:
        """Build an AST-only graph; no source leaves the local machine."""
        project = Path(project_path).resolve()
        output = self._run("extract", str(project), "--code-only", cwd=project)
        graph_path = project / "graphify-out" / "graph.json"
        if not graph_path.exists():
            raise GraphifyError("Graphify completed without creating graphify-out/graph.json.")
        return graph_path, output

    def query(self, question: str, project_path: Path | str) -> str:
        project = Path(project_path).resolve()
        graph_path = project / "graphify-out" / "graph.json"
        if not graph_path.exists():
            raise GraphifyError("No graph found. Run 'buffdata productivity graph-build' first.")
        return self._run("query", question, "--graph", str(graph_path), cwd=project)
