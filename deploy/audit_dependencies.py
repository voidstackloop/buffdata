"""Audit the deployment lock without altering the application's interpreter.

Uses an isolated temporary virtualenv; downloads pip-audit and public advisory metadata.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=None,
        help="Defaults to requirements-cpu.lock; pass requirements-presidio.lock to audit "
             "the isolated presidio-anonymizer venv's closure separately.")
    args = parser.parse_args()
    lock = args.lock.resolve() if args.lock else Path(__file__).resolve().with_name("requirements-cpu.lock")
    with tempfile.TemporaryDirectory(prefix="buffdata-dependency-audit-") as temporary:
        environment = venv.EnvBuilder(with_pip=True)
        environment.create(temporary)
        python = Path(temporary) / "bin" / "python"
        subprocess.run([str(python), "-m", "pip", "install", "--quiet", "pip-audit"], check=True)
        result = subprocess.run([str(python), "-m", "pip_audit", "-r", str(lock), "--no-deps",
            "--disable-pip", "--format", "json", "--output", str(args.output.resolve())])
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
