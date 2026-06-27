"""Run provenance and artifact integrity helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from competitor_pricing_ai.config import PipelineConfig
from competitor_pricing_ai.reporting import write_json


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(
    config: PipelineConfig,
    artifacts: dict[str, str | None],
    output_dir: Path,
) -> str:
    artifact_records = {}
    for name, value in artifacts.items():
        if name == "run_manifest":
            continue
        if not value:
            continue
        path = Path(value)
        if path.is_file():
            artifact_records[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    config_json = json.dumps(config.to_dict(), sort_keys=True, default=str).encode("utf-8")
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.root_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=config.root_dir,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        git_commit = None
        git_dirty = None
    manifest = {
        "schema_version": "2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_worktree_dirty": git_dirty,
        "config_sha256": hashlib.sha256(config_json).hexdigest(),
        "artifacts": artifact_records,
        "governance_note": (
            "Only artifacts listed here belong to this training run. Monitoring and handoff "
            "must verify hashes before use."
        ),
    }
    path = output_dir / "run_manifest.json"
    write_json(manifest, path)
    return str(path)


def verify_manifest_artifact(output_dir: Path, artifact_name: str, path: Path) -> None:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError("run_manifest.json is required to prevent stale artifact mixing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("artifacts", {}).get(artifact_name, {}).get("sha256")
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"Artifact integrity check failed for {artifact_name}: {path}")
