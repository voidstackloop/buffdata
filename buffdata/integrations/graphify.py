"""Local Graphify knowledge-graph integration for repository navigation."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Optional


class GraphifyError(RuntimeError):
    """Raised when Graphify is not installed or a graph operation fails."""


class GraphifyManager:
    def __init__(self, command: Optional[str] = None):
        self.command = command or shutil.which("graphify")

    def _run(self, *arguments: str, cwd: Path) -> str:
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
        )
        if result.returncode:
            message = (result.stderr or result.stdout).strip()
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
