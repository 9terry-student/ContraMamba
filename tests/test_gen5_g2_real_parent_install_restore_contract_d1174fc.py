"""CPU-only, stdlib contract tests for the G2 install/restore runner.

These do NOT construct a pretrained model, open checkpoints, or run CUDA.
"""
import importlib.util
from pathlib import Path
import tempfile
import pytest

SCRIPT = (Path(__file__).resolve().parents[1] / 'scripts' /
          'gen5_g2_real_parent_install_restore_only_d1174fc.py')
spec = importlib.util.spec_from_file_location('g2_install_driver', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_import_has_no_torch_dependency():
    assert mod.HELDOUT_SEED == 6201
    assert mod.ARM == 'FULL'


def test_pin_contract():
    assert len(mod.BASE_HEAD) == 40
    assert mod.STRICT_EVIDENCE_BLOB == '012e66e1040dbe7d405d0889f7781c55780a0650'
    assert mod.G1_LOADER_BLOB == '2a47b6278d966e23d4920e51396c91f7f4b184ee'


def test_out_allowed_only_in_registered_root(monkeypatch, tmp_path):
    relative = mod.OUTPUT_PARENT + '/gen5-g2-install-d1174fc-r1'
    out = tmp_path / relative / mod.OUTPUT_BASENAME
    monkeypatch.setenv('CONTRAMAMBA_RUN_ARTIFACT_ROOT', relative)
    assert mod.check_output_path(tmp_path, out) == relative


def test_out_rejects_wrong_environment(monkeypatch, tmp_path):
    relative = mod.OUTPUT_PARENT + '/gen5-g2-install-d1174fc-r1'
    out = tmp_path / relative / mod.OUTPUT_BASENAME
    monkeypatch.setenv('CONTRAMAMBA_RUN_ARTIFACT_ROOT', 'reports/other')
    with pytest.raises(RuntimeError, match='OUTPUT_RUN_ROOT_ENV_MISMATCH'):
        mod.check_output_path(tmp_path, out)


@pytest.mark.parametrize('relative', [
    'reports/other_runs/my-run/install_restore_identity.json',
    'reports/reason_router_gen5_g2_parent_installation_only_runs/my-run/other.json',
    'reports/reason_router_gen5_g2_parent_installation_only_runs/../install_restore_identity.json',
])
def test_out_rejects_scope(monkeypatch, tmp_path, relative):
    monkeypatch.setenv('CONTRAMAMBA_RUN_ARTIFACT_ROOT', 'reports/other')
    with pytest.raises(RuntimeError):
        mod.check_output_path(tmp_path, tmp_path / relative)


def test_out_rejects_external_path(tmp_path):
    with tempfile.TemporaryDirectory() as outside:
        with pytest.raises(RuntimeError, match='OUTPUT_NOT_REPO_LOCAL'):
            mod.check_output_path(tmp_path, Path(outside) / mod.OUTPUT_BASENAME)


def test_out_rejects_collision(monkeypatch, tmp_path):
    relative = mod.OUTPUT_PARENT + '/my-run'
    out = tmp_path / relative / mod.OUTPUT_BASENAME
    out.parent.mkdir(parents=True)
    out.write_text('already exists')
    monkeypatch.setenv('CONTRAMAMBA_RUN_ARTIFACT_ROOT', relative)
    with pytest.raises(RuntimeError, match='OUTPUT_COLLISION'):
        mod.check_output_path(tmp_path, out)


def test_gpu_install_requires_explicit_authorization(tmp_path):
    with pytest.raises(RuntimeError, match='EXPLICIT_GPU_INSTALL_AUTHORIZATION_REQUIRED'):
        mod.installation_only(tmp_path, mod.BASE_HEAD, tmp_path, tmp_path,
                              tmp_path / 'result.json', False)


def test_runner_contains_no_forward_invocation():
    import ast
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        (isinstance(node.func, ast.Attribute) and node.func.attr == 'forward') or
        (isinstance(node.func, ast.Name) and node.func.id == 'forward')
        for node in calls
    )
