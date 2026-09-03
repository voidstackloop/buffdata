from pathlib import Path

import yaml


DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


def test_compose_keeps_worker_and_api_off_the_edge_network():
    definition = yaml.safe_load((DEPLOY / "compose.yaml").read_text())
    services = definition["services"]
    assert set(services["api"]["networks"]) == {"control", "database"}
    assert services["worker"]["networks"] == ["control"]
    assert "ports" not in services["api"] and "ports" not in services["worker"]
    assert definition["networks"]["control"]["internal"]
    assert definition["networks"]["database"]["internal"]
    for name in ("api", "worker", "ingress"):
        service = services[name]
        assert service["read_only"] and service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]
    assert "database_url" not in services["worker"]["secrets"]
    assert "provider_secrets" not in services["api"]["secrets"]
    assert all("docker.sock" not in str(v) for s in services.values() for v in s.get("volumes", []))
    assert all("/data" not in v for v in services["ingress"]["volumes"])


def test_nginx_fixed_upstream_and_temporary_paths():
    config = (DEPLOY / "nginx.conf").read_text()
    assert "location /internal/ { return 404; }" in config
    assert "set $api http://api:8000;" in config
    for kind in ("client_body", "proxy", "fastcgi", "uwsgi", "scgi"):
        assert kind + "_temp_path /tmp/" in config
    dockerfile = (DEPLOY / "Dockerfile.team").read_text()
    assert dockerfile.index("COPY buffdata") > dockerfile.index("en_core_web_lg")
    assert "--no-build-isolation" in dockerfile
