"""Shared pytest configuration: tests exercise the blocking semantics."""

from __future__ import annotations

import os


def pytest_configure(config):  # noqa: ARG001 - pytest hook signature
    os.environ.setdefault("CONTRAGENT_MODE", "enforce")
