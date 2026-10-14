"""Non-executing CPU tests for G2 strict-load Kaggle packaging contracts."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / 'scripts'
    / 'gen5_g2_historical_parent_strict_load_only_diagnostic_d11f355.py'
)
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).with_name(
        'gen5_g2_historical_parent_strict_load_only_diagnostic_d11f355_kaggle_contract_fix.py'
    )
spec = importlib.util.spec_from_file_location('g2_strict_diag', SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _stub_repo(monkeypatch, *, branch='', head='deadbeef', dirty=False):
    def fake_git(_repo, *args):
        if args == ('rev-parse', 'HEAD'):
            return head
        if args == ('branch', '--show-current'):
            return branch
        if args == ('status', '--porcelain'):
            return '?? unexpected.txt' if dirty else ''
        if len(args) == 2 and args[0] == 'rev-parse' and args[1].startswith('HEAD:'):
            return module.SOURCE_BLOBS[args[1].removeprefix('HEAD:')]
        raise AssertionError(f'unexpected git {args}')

    monkeypatch.setattr(module, 'git', fake_git)
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))


def test_exact_detached_head_allowed(monkeypatch, tmp_path):
    _stub_repo(monkeypatch, branch='', head='deadbeef')
    module.check_repository(tmp_path, 'deadbeef')


def test_expected_branch_allowed(monkeypatch, tmp_path):
    _stub_repo(monkeypatch, branch=module.BRANCH, head='deadbeef')
    module.check_repository(tmp_path, 'deadbeef')


def test_unrelated_branch_rejected(monkeypatch, tmp_path):
    _stub_repo(monkeypatch, branch='main', head='deadbeef')
    with pytest.raises(RuntimeError, match='BRANCH_MISMATCH'):
        module.check_repository(tmp_path, 'deadbeef')


def test_detached_wrong_head_rejected(monkeypatch, tmp_path):
    _stub_repo(monkeypatch, branch='', head='a'*40)
    with pytest.raises(RuntimeError, match='HEAD_MISMATCH'):
        module.check_repository(tmp_path, 'b'*40)


def test_detached_dirty_tree_rejected(monkeypatch, tmp_path):
    _stub_repo(monkeypatch, branch='', head='deadbeef', dirty=True)
    with pytest.raises(RuntimeError, match='WORKTREE_DIRTY'):
        module.check_repository(tmp_path, 'deadbeef')


def test_registered_output_scope_accepted(monkeypatch, tmp_path):
    rel = 'reports/reason_router_gen5_g2_strict_load_only_runs/g2-strict-load-9b931dd-r1/strict_load_identity.json'
    monkeypatch.setenv('CONTRAMAMBA_RUN_ARTIFACT_ROOT', rel.rsplit('/', 1)[0])
    module.check_output_destination(tmp_path, tmp_path / rel)


def test_wrong_output_scope_rejected(monkeypatch, tmp_path):
    rel = 'reports/unrelated/g2-strict-load-9b931dd-r1/strict_load_identity.json'
    monkeypatch.setenv('CONTRAMAMBA_RUN_ARTIFACT_ROOT', rel.rsplit('/', 1)[0])
    with pytest.raises(RuntimeError, match='OUTPUT_RUN_SCOPE'):
        module.check_output_destination(tmp_path, tmp_path / rel)


def test_wrong_run_env_rejected(monkeypatch, tmp_path):
    rel = 'reports/reason_router_gen5_g2_strict_load_only_runs/g2-strict-load-9b931dd-r1/strict_load_identity.json'
    monkeypatch.setenv('CONTRAMAMBA_RUN_ARTIFACT_ROOT', 'reports/wrong')
    with pytest.raises(RuntimeError, match='OUTPUT_RUN_ROOT_ENV_MISMATCH'):
        module.check_output_destination(tmp_path, tmp_path / rel)


def test_existing_output_rejected(monkeypatch, tmp_path):
    out = tmp_path / 'outside.json'
    out.write_text('{}')
    with pytest.raises(RuntimeError, match='OUTPUT_COLLISION'):
        module.check_output_destination(tmp_path, out)


def test_external_output_still_allowed(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    module.check_output_destination(repo, tmp_path / 'outside.json')
