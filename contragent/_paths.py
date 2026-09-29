"""Path-resolution helpers used to defend against path traversal.

Used by code paths whose path argument may come from user-controlled
input (YAML config ``include:`` clauses, API request bodies, agent ids
that turn into directory names, …).  CLI entry points where the user
already chose the path freely don't need these — passing
``safe_root=None`` keeps the previous "trust the caller" behavior.

The two helpers here implement the same canonical pattern::

    p = Path(spec).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()
    p.relative_to(safe_root.resolve())  # raises ValueError on escape

…packaged with friendlier error messages so callers get one line.
"""

from __future__ import annotations

from pathlib import Path


class PathEscapeError(ValueError):
    """Raised when a resolved path falls outside the allowed root."""


def safe_resolve(
    spec: str | Path,
    *,
    base_dir: Path | None = None,
    safe_root: Path | None = None,
    allow_absolute: bool = True,
) -> Path:
    """Resolve ``spec`` to an absolute :class:`Path`, optionally
    confining it under ``safe_root``.

    Args:
        spec: The user-provided path. ``~`` is always expanded.
        base_dir: Used to anchor relative paths. If ``None``, relative
            paths resolve against the current working directory.
        safe_root: If provided, the resolved path **must** be a
            descendant of (or equal to) ``safe_root.resolve()`` — any
            ``..`` traversal that escapes triggers
            :class:`PathEscapeError`.
        allow_absolute: If ``False``, absolute ``spec`` values (after
            ``expanduser``) are rejected. Defaults to ``True`` for
            backward-compatibility with CLI callers.

    Returns:
        The fully-resolved absolute :class:`Path`. Existence is **not**
        checked here — that's the caller's job.

    Raises:
        PathEscapeError: If the resolved path escapes ``safe_root``
            (when one is supplied), or if ``allow_absolute=False`` and
            the spec is absolute after ``expanduser``.
    """
    p = Path(spec).expanduser()

    if not allow_absolute and p.is_absolute():
        raise PathEscapeError(f"absolute paths are not allowed here: {spec!r}")

    if not p.is_absolute():
        anchor = (base_dir or Path.cwd()).resolve()
        p = (anchor / p).resolve()
    else:
        p = p.resolve()

    if safe_root is not None:
        root = safe_root.resolve()
        try:
            p.relative_to(root)
        except ValueError as e:
            raise PathEscapeError(
                f"path {spec!r} resolves to {p} which is outside the allowed root {root}"
            ) from e

    return p
