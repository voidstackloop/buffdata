import json
from pathlib import Path

from buffdata.integrations.context7 import Context7Client
from buffdata.integrations.graphify import GraphifyManager


class Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    @property
    def headers(self):
        return type("Headers", (), {"get_content_type": lambda _: "application/json"})()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_context7_uses_authorization_and_query_parameters():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        return Response({"results": []})

    client = Context7Client(api_key="test-key", opener=opener)
    assert client.get_context("/pydantic/pydantic", "model validators") == {"results": []}
    assert "libraryId=%2Fpydantic%2Fpydantic" in seen["url"]
    assert "query=model+validators" in seen["url"]
    assert seen["authorization"] == "Bearer test-key"
    assert seen["timeout"] == 30


def test_graphify_build_and_query_commands(tmp_path: Path, monkeypatch):
    commands = []
    graph = tmp_path / "graphify-out" / "graph.json"

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[1] == "extract":
            graph.parent.mkdir()
            graph.write_text("{}", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("buffdata.integrations.graphify.subprocess.run", fake_run)
    manager = GraphifyManager(command="graphify")
    graph_path, _ = manager.build(tmp_path)
    assert graph_path == graph
    assert manager.query("what calls the pipeline?", tmp_path) == "ok"
    assert commands[0][1:3] == ["extract", str(tmp_path)]
    assert commands[1][1:3] == ["query", "what calls the pipeline?"]
