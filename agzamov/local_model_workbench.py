"""Local-first llama.cpp identity and capability-vector inspection.

This module is deliberately independent of scored gameplay. It captures an
immutable local runtime/profile receipt and inspects pre-registered Stage E
artifacts without changing their strict scores.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable


LOCAL_PROFILE_RECORD_SCHEMA = "agzamov.local-profile-record.v1"
LOCAL_PROFILE_SCHEMA = "agzamov.local-profile.v1"
LOCAL_CAPABILITY_VECTOR_SCHEMA = "agzamov.local-capability-vector.v1"


class LocalProfileError(RuntimeError):
    """Raised when a local runtime identity cannot be captured safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_snapshot_sha256(profile: dict[str, Any]) -> str:
    encoded = json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _default_fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise LocalProfileError(f"non-object response from {url}")
    return payload


def _default_run_version(binary: Path) -> str:
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LocalProfileError(f"cannot execute llama.cpp binary: {exc}") from exc
    text = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or not text:
        raise LocalProfileError("llama.cpp binary did not return a version")
    return text


def _served_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("data")
    if not isinstance(candidates, list):
        candidates = payload.get("models")
    if not isinstance(candidates, list):
        return []
    return [item for item in candidates if isinstance(item, dict)]


def _model_alias(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("model") or item.get("name") or "")


def _runtime_commit(version: str) -> str:
    match = re.search(r"\(([0-9a-f]{7,40})\)", version, re.IGNORECASE)
    if not match:
        raise LocalProfileError("llama.cpp version does not expose a commit hash")
    return match.group(1).lower()


def capture_llama_cpp_profile(
    *,
    profile_id: str,
    endpoint: str,
    model_alias: str,
    model_file: str | Path,
    runtime_binary: str | Path,
    backend: str,
    context_tokens: int,
    sampling: dict[str, Any],
    reasoning: dict[str, Any],
    response_format: dict[str, Any],
    fetch_json: Callable[[str], dict[str, Any]] = _default_fetch_json,
    run_version: Callable[[Path], str] = _default_run_version,
) -> dict[str, Any]:
    """Capture a deterministic, secret-free llama.cpp profile record."""

    normalized_endpoint = endpoint.rstrip("/")
    if not profile_id.strip():
        raise LocalProfileError("profile_id is required")
    model_path = Path(model_file).expanduser().resolve()
    binary_path = Path(runtime_binary).expanduser().resolve()
    if not model_path.is_file():
        raise LocalProfileError(f"model file not found: {model_path}")
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise LocalProfileError(f"llama.cpp binary is not executable: {binary_path}")

    try:
        health = fetch_json(f"{normalized_endpoint}/health")
        models_payload = fetch_json(f"{normalized_endpoint}/v1/models")
    except LocalProfileError:
        raise
    except Exception as exc:
        raise LocalProfileError(f"llama.cpp endpoint probe failed: {exc}") from exc
    if health.get("status") != "ok":
        raise LocalProfileError("llama.cpp health check did not return status=ok")
    served = next(
        (item for item in _served_models(models_payload) if _model_alias(item) == model_alias),
        None,
    )
    if served is None:
        raise LocalProfileError(f"served model alias not found: {model_alias}")

    version = run_version(binary_path).strip()
    commit = _runtime_commit(version)
    metadata = served.get("meta") or served.get("details") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    profile = {
        "schema_version": LOCAL_PROFILE_SCHEMA,
        "profile_id": profile_id,
        "runtime": {
            "engine": "llama.cpp",
            "endpoint": normalized_endpoint,
            "binary": str(binary_path),
            "version": version,
            "commit": commit,
            "backend": backend,
            "context_tokens": context_tokens,
        },
        "model": {
            "served_alias": model_alias,
            "weight_file": str(model_path),
            "weight_sha256": _sha256_file(model_path),
            "metadata": metadata,
        },
        "sampling": dict(sampling),
        "reasoning": dict(reasoning),
        "response_format": dict(response_format),
    }
    return {
        "schema_version": LOCAL_PROFILE_RECORD_SCHEMA,
        "profile": profile,
        "profile_sha256": profile_snapshot_sha256(profile),
    }


def verify_local_profile_record(
    record: dict[str, Any],
    *,
    fetch_json: Callable[[str], dict[str, Any]] = _default_fetch_json,
    run_version: Callable[[Path], str] = _default_run_version,
) -> dict[str, Any]:
    """Fail closed if a captured local profile no longer matches reality."""

    issues: list[dict[str, str]] = []

    def add(code: str, message: str) -> None:
        issues.append({"code": code, "message": message})

    profile = record.get("profile")
    if record.get("schema_version") != LOCAL_PROFILE_RECORD_SCHEMA or not isinstance(profile, dict):
        add("profile_schema_invalid", "unsupported or missing local profile schema")
        return {"ok": False, "issues": issues}
    if record.get("profile_sha256") != profile_snapshot_sha256(profile):
        add("profile_hash_mismatch", "profile snapshot hash mismatch")

    runtime = profile.get("runtime") or {}
    model = profile.get("model") or {}
    model_path = Path(str(model.get("weight_file") or ""))
    binary_path = Path(str(runtime.get("binary") or ""))
    if not model_path.is_file():
        add("model_file_missing", str(model_path))
    elif _sha256_file(model_path) != model.get("weight_sha256"):
        add("model_weight_hash_mismatch", str(model_path))
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        add("runtime_binary_missing", str(binary_path))
    else:
        try:
            observed_version = run_version(binary_path).strip()
            if observed_version != runtime.get("version"):
                add("runtime_version_mismatch", observed_version)
            elif _runtime_commit(observed_version) != runtime.get("commit"):
                add("runtime_commit_mismatch", observed_version)
        except Exception as exc:
            add("runtime_probe_failed", str(exc))

    endpoint = str(runtime.get("endpoint") or "").rstrip("/")
    try:
        health = fetch_json(f"{endpoint}/health")
        models_payload = fetch_json(f"{endpoint}/v1/models")
        if health.get("status") != "ok":
            add("runtime_health_failed", "status is not ok")
        aliases = {_model_alias(item) for item in _served_models(models_payload)}
        if model.get("served_alias") not in aliases:
            add("served_model_mismatch", str(model.get("served_alias")))
    except Exception as exc:
        add("runtime_endpoint_unavailable", str(exc))
    return {
        "ok": not issues,
        "profile_id": profile.get("profile_id"),
        "profile_sha256": record.get("profile_sha256"),
        "issues": issues,
    }


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def inspect_stage_e_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Return a fail-closed local capability vector for a Stage E artifact set."""

    root = Path(run_dir)
    issues: list[dict[str, str]] = []
    required = ("protocol.json", "protocol.sha256", "summary.json", "artifact-manifest.json")
    if not root.is_dir():
        issues.append(_issue("run_directory_missing", str(root)))
        return {
            "schema_version": LOCAL_CAPABILITY_VECTOR_SCHEMA,
            "ok": False,
            "run_id": root.name,
            "issues": issues,
            "layers": {},
            "conditions": {},
        }
    for name in required:
        if not (root / name).is_file():
            issues.append(_issue("artifact_missing", name))
    if issues:
        return {
            "schema_version": LOCAL_CAPABILITY_VECTOR_SCHEMA,
            "ok": False,
            "run_id": root.name,
            "issues": issues,
            "layers": {},
            "conditions": {},
        }

    try:
        protocol = json.loads((root / "protocol.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        manifest = json.loads((root / "artifact-manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(_issue("artifact_json_invalid", str(exc)))
        return {
            "schema_version": LOCAL_CAPABILITY_VECTOR_SCHEMA,
            "ok": False,
            "run_id": root.name,
            "issues": issues,
            "layers": {},
            "conditions": {},
        }

    if not isinstance(manifest, dict):
        issues.append(_issue("artifact_manifest_invalid", "manifest must be an object"))
    else:
        resolved_root = root.resolve()
        for relative, expected in manifest.items():
            relative_path = Path(str(relative))
            path = root / relative_path
            try:
                path.resolve().relative_to(resolved_root)
                contained = not relative_path.is_absolute() and ".." not in relative_path.parts
            except ValueError:
                contained = False
            if not contained:
                issues.append(_issue("artifact_key_invalid", str(relative)))
            elif not path.is_file():
                issues.append(_issue("artifact_missing", str(relative)))
            elif not isinstance(expected, str) or _sha256_file(path) != expected:
                issues.append(_issue("artifact_hash_mismatch", str(relative)))

    actual_protocol_hash = _sha256_file(root / "protocol.json")
    declared_protocol_hash = (root / "protocol.sha256").read_text().split()[0]
    if actual_protocol_hash != declared_protocol_hash:
        issues.append(_issue("protocol_hash_mismatch", "protocol.json"))
    if summary.get("protocol_sha256") != actual_protocol_hash:
        issues.append(_issue("summary_protocol_mismatch", "summary protocol hash differs"))
    if summary.get("protocol_id") != protocol.get("protocol_id"):
        issues.append(_issue("summary_protocol_mismatch", "protocol IDs differ"))
    expected_calls = len(protocol.get("calls") or [])
    if (
        summary.get("gameplay_started") is not False
        or summary.get("all_calls_completed") is not True
        or summary.get("call_count") != expected_calls
    ):
        issues.append(_issue("stage_e_incomplete", "call cardinality or gameplay flag differs"))
    if issues:
        return {
            "schema_version": LOCAL_CAPABILITY_VECTOR_SCHEMA,
            "ok": False,
            "run_id": root.name,
            "protocol_id": protocol.get("protocol_id"),
            "issues": issues,
            "layers": {},
            "conditions": {},
        }

    strict_conditions = summary.get("by_condition") or {}
    semantic_path = root / "semantic-diagnostic.json"
    semantic = None
    if semantic_path.is_file():
        try:
            semantic = json.loads(semantic_path.read_text())
        except json.JSONDecodeError as exc:
            issues.append(_issue("semantic_diagnostic_invalid", str(exc)))
    semantic_conditions = (semantic or {}).get("by_condition") or {}

    condition_names = sorted(strict_conditions)
    conditions: dict[str, Any] = {}
    total = sum(int(strict_conditions[name].get("total", 0)) for name in condition_names)
    strict_usable = sum(
        int(strict_conditions[name].get("fully_valid", 0)) for name in condition_names
    )
    strict_uci = sum(
        int(strict_conditions[name].get("uci_syntax", 0)) for name in condition_names
    )
    if semantic is not None:
        state_bound = int(semantic.get("state_id_matches_after_wrapper_parse", 0))
        semantic_legal = int(semantic.get("semantic_legal", 0))
        queen_risk = int(semantic.get("semantic_queen_capturable_next_ply", 0))
        semantic_status = "diagnostic_non_scoring"
    else:
        state_bound = sum(
            int(strict_conditions[name].get("state_id_matches", 0)) for name in condition_names
        )
        semantic_legal = sum(
            int(strict_conditions[name].get("legal", 0)) for name in condition_names
        )
        queen_risk = sum(
            int(strict_conditions[name].get("queen_capturable_next_ply", 0))
            for name in condition_names
        )
        semantic_status = "strict_only"

    for name in condition_names:
        strict = strict_conditions[name]
        diagnostic = semantic_conditions.get(name) or {}
        condition_total = int(strict.get("total", 0))
        condition_semantic_legal = int(
            diagnostic.get("semantic_legal", strict.get("legal", 0))
        )
        condition_risk = int(
            diagnostic.get(
                "semantic_queen_capturable_next_ply",
                strict.get("queen_capturable_next_ply", 0),
            )
        )
        conditions[name] = {
            "strict_machine_usable": {
                "passed": int(strict.get("fully_valid", 0)),
                "total": condition_total,
            },
            "semantic_action_legality": {
                "status": semantic_status,
                "passed": condition_semantic_legal,
                "total": condition_total,
            },
            "immediate_tactical_safety": {
                "status": semantic_status,
                "passed": condition_semantic_legal - condition_risk,
                "total": condition_semantic_legal,
            },
        }

    not_measured = {"status": "not_measured", "passed": None, "total": None}
    layers = {
        "response_complete": {
            "status": "measured",
            "passed": total,
            "total": total,
        },
        "state_binding": {
            "status": "measured",
            "passed": state_bound,
            "total": total,
        },
        "strict_serialization": {
            "status": "measured",
            "passed": strict_uci,
            "total": total,
        },
        "strict_machine_usable": {
            "status": "measured",
            "passed": strict_usable,
            "total": total,
        },
        "semantic_action_legality": {
            "status": semantic_status,
            "passed": semantic_legal,
            "total": total,
        },
        "immediate_tactical_safety": {
            "status": semantic_status,
            "passed": semantic_legal - queen_risk,
            "total": semantic_legal,
        },
        "operator_exactness": dict(not_measured),
        "sparse_composition": dict(not_measured),
        "dense_scaling": dict(not_measured),
        "transition_binding": dict(not_measured),
        "multi_step_conversion": dict(not_measured),
    }
    return {
        "schema_version": LOCAL_CAPABILITY_VECTOR_SCHEMA,
        "ok": not issues,
        "run_id": root.name,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": actual_protocol_hash,
        "model_profile": protocol.get("model_profile"),
        "issues": issues,
        "layers": layers,
        "conditions": conditions,
    }
