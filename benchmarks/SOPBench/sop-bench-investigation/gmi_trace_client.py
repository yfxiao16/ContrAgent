"""Shared GMI Cloud API helpers for trace-mining experiments."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://api.gmi-serving.com/v1"


def load_local_env(repo_root: Path) -> None:
    """Load GMI_API_KEY from an ignored repository .env without logging it."""
    if os.environ.get("GMI_API_KEY"):
        return
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != "GMI_API_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            os.environ["GMI_API_KEY"] = value
        return


def request_json(
    path: str,
    key: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "openai-python/1.0.0",
        },
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def generate_completion(
    model_id: str,
    prompt: str,
    key: str,
    timeout: int = 180,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You infer formal temporal contracts from labeled interaction traces."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if model_id.startswith(("openai/gpt-5.4", "openai/gpt-5.6")):
        payload["max_completion_tokens"] = payload.pop("max_tokens")

    for attempt_no in range(1, 3):
        try:
            return request_json("/chat/completions", key, payload, timeout=timeout)
        except (urllib.error.HTTPError, TimeoutError, urllib.error.URLError) as exc:
            retryable = (
                isinstance(exc, urllib.error.HTTPError) and (exc.code == 429 or exc.code >= 500)
            ) or (not isinstance(exc, urllib.error.HTTPError) and isinstance(exc, (TimeoutError, urllib.error.URLError)))
            if attempt_no == 2 or not retryable:
                raise
            delay = 30
            if isinstance(exc, urllib.error.HTTPError) and exc.headers.get("Retry-After", "").isdigit():
                delay = max(delay, int(exc.headers["Retry-After"]))
            time.sleep(delay)
    raise RuntimeError("GMI completion retry loop exited unexpectedly")
