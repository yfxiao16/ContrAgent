"""Tests for ``include:``: pulling a shipped or local library into an agent block."""

from __future__ import annotations

from pathlib import Path

import pytest

from contragent.config import (
    ConfigError,
    _resolve_include_spec,
    bundled_libraries_root,
    config_to_system,
    load_config,
)

BANK = "contragent:sopbench/bank"


class TestResolveIncludeSpec:
    def test_bundled_spec_resolves_under_package(self):
        path = _resolve_include_spec(BANK, Path.cwd())
        assert path.exists()
        assert path.is_relative_to(bundled_libraries_root())

    def test_yaml_suffix_optional(self):
        a = _resolve_include_spec(BANK, Path.cwd())
        b = _resolve_include_spec(BANK + ".yaml", Path.cwd())
        assert a == b

    @pytest.mark.parametrize(
        "spec",
        sorted(
            f"contragent:{p.relative_to(bundled_libraries_root()).with_suffix('')}"
            for p in bundled_libraries_root().rglob("*.yaml")
        ),
    )
    def test_every_shipped_library_is_resolvable_and_loads(self, spec, tmp_path):
        assert _resolve_include_spec(spec, Path.cwd()).exists()
        host = tmp_path / "host.yaml"
        host.write_text(f'version: "1"\nagents:\n  bot:\n    include: ["{spec}"]\n')
        system = config_to_system(load_config(host))
        assert system.contracts

    def test_unknown_bundled_spec_lists_available(self):
        with pytest.raises(ConfigError, match="not found") as exc:
            _resolve_include_spec("contragent:nope/missing", Path.cwd())
        assert BANK in str(exc.value)

    def test_empty_bundled_spec_rejected(self):
        with pytest.raises(ConfigError, match="empty"):
            _resolve_include_spec("contragent:", Path.cwd())

    def test_path_traversal_in_bundled_spec_rejected(self):
        with pytest.raises(ConfigError, match="outside"):
            _resolve_include_spec("contragent:../../pyproject", Path.cwd())

    def test_missing_local_path_named(self, tmp_path):
        with pytest.raises(ConfigError, match="file not found"):
            _resolve_include_spec("no_such.yaml", tmp_path)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


LOCAL_LIB = """\
version: "1"
agents:
  "*":
    contracts:
      - desc: "pay_bill at most twice"
        G: {ltl: "G((Var('count', 'pay_bill') <= 2))"}
"""


class TestIncludeIntoAgent:
    def test_included_then_local_contracts_in_order(self, tmp_path):
        _write(tmp_path / "shared.yaml", LOCAL_LIB)
        host = _write(
            tmp_path / "host.yaml",
            'version: "1"\nagents:\n  bot:\n    include: ["shared.yaml"]\n'
            '    contracts:\n      - desc: "local"\n        G: {ltl: "G(!(called(\'rm\')))"}\n',
        )
        cfg = load_config(host)
        entries = cfg.agents["bot"].contracts
        assert [e.desc for e in entries] == ["pay_bill at most twice", "local"]
        assert entries[0].library_source == "shared.yaml"
        assert entries[1].library_source is None

    def test_pulled_contracts_compile(self, tmp_path):
        _write(tmp_path / "shared.yaml", LOCAL_LIB)
        host = _write(
            tmp_path / "host.yaml", 'version: "1"\nagents:\n  bot:\n    include: ["shared.yaml"]\n'
        )
        system = config_to_system(load_config(host))
        assert len(system.contracts) == 1
        assert system.contracts[0].agent.id == "bot"

    def test_library_must_define_single_wildcard_agent(self, tmp_path):
        _write(tmp_path / "bad.yaml", 'version: "1"\nagents:\n  alice:\n    contracts: []\n')
        host = _write(
            tmp_path / "host.yaml", 'version: "1"\nagents:\n  bot:\n    include: ["bad.yaml"]\n'
        )
        with pytest.raises(ConfigError, match="exactly one"):
            load_config(host)


class TestNestedInclude:
    def test_library_can_include_another_library(self, tmp_path):
        _write(tmp_path / "leaf.yaml", LOCAL_LIB)
        _write(
            tmp_path / "mid.yaml",
            'version: "1"\nagents:\n  "*":\n    include: ["leaf.yaml"]\n'
            '    contracts:\n      - desc: "mid"\n        G: {ltl: "G(!(called(\'drop\')))"}\n',
        )
        host = _write(
            tmp_path / "host.yaml", 'version: "1"\nagents:\n  bot:\n    include: ["mid.yaml"]\n'
        )
        cfg = load_config(host)
        assert [e.desc for e in cfg.agents["bot"].contracts] == ["pay_bill at most twice", "mid"]

    def test_cycle_detected(self, tmp_path):
        _write(
            tmp_path / "a.yaml",
            'version: "1"\nagents:\n  "*":\n    include: ["b.yaml"]\n    contracts: []\n',
        )
        _write(
            tmp_path / "b.yaml",
            'version: "1"\nagents:\n  "*":\n    include: ["a.yaml"]\n    contracts: []\n',
        )
        host = _write(
            tmp_path / "host.yaml", 'version: "1"\nagents:\n  bot:\n    include: ["a.yaml"]\n'
        )
        with pytest.raises(ConfigError, match="cycle"):
            load_config(host)
