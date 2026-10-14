"""CPU-only structural and projection checks for G2 science CUDA preflight.

These tests never construct models, load frozen checkpoints, or use CUDA.
"""
import ast
import importlib.util
from pathlib import Path
import pytest

MODULE = Path(__file__).resolve().parents[1] / 'scripts' / 'gen5_g2_scientific_cuda_two_gpu_preflight_3960eff.py'
SPEC = importlib.util.spec_from_file_location('g2_science_preflight', MODULE)
g2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g2)


def test_complete_shards_and_exact_batch_schedule():
    a=g2.shard_batches(0)
    b=g2.shard_batches(1)
    assert len(a)==13 and len(b)==14
    assert a[0]==(0,32) and a[-1]==(384,416)
    assert b[0]==(416,448) and b[-1]==(832,840)
    assert sum(y-x for x,y in a+b)==840
    assert len({i for x,y in a+b for i in range(x,y)})==840


def test_fixed_seed_arm_order_and_forward_count():
    assert g2.SEEDS==(6201,6202,6203)
    assert g2.ARMS==('ZERO','FULL','SHARED_ONLY','RESIDUAL_ONLY','SHARED_NORMMATCH','RESIDUAL_NORMMATCH')
    assert g2.EXPECTED_CELLS==18
    assert g2.EXPECTED_BACKBONE_FORWARDS_PER_BATCH==22
    assert sum(len(g2.shard_batches(w))*g2.EXPECTED_BACKBONE_FORWARDS_PER_BATCH for w in (0,1))==594


def test_eta_uses_parallel_max_not_sum():
    inp={0:{'benchmark_sec':120.,'setup_sec':45.},1:{'benchmark_sec':150.,'setup_sec':60.}}
    e=g2.estimate_seconds(inp)
    assert e['parallel_wall_lower_sec']==60+150*(13+.25)
    assert e['parallel_wall_conservative_sec']==60+1.3*150*14
    assert e['not_a_measured_full_run'] is True
    assert e['parallel_wall_conservative_sec']>e['parallel_wall_lower_sec']


def test_projection_rejects_nonpositive_bench():
    with pytest.raises(RuntimeError,match='BLOCKED:TIME_INPUT'):
        g2.estimate_seconds({0:{'benchmark_sec':0.,'setup_sec':1.},1:{'benchmark_sec':4.,'setup_sec':2.}})


def test_guard_fails_on_unregistered_or_existing_output(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir()
    folder=repo/'reports'/'reason_router_gen5_g2_science_preflight_runs'/'g2-science-pref-1'
    with pytest.raises(RuntimeError,match='BLOCKED:RUN_ARTIFACT_ROOT_MISMATCH'):
        g2.output_contract(repo,folder)
    monkeypatch.setenv('CONTRAMAMBA_RUN_ARTIFACT_ROOT',folder.relative_to(repo).as_posix())
    assert g2.output_contract(repo,folder)==folder.relative_to(repo).as_posix()
    folder.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='BLOCKED:OUTPUT_DIR_EXISTS'):
        g2.output_contract(repo,folder)


def test_no_full_run_mode_and_cuda_opt_in():
    tree=ast.parse(MODULE.read_text(encoding='utf-8'))
    text=MODULE.read_text(encoding='utf-8')
    assert "choices=('preflight','worker')" in text
    assert "need(args.authorize_gpu_preflight,'GPU_PREFLIGHT_EXPLICIT_AUTHORIZATION')" in text
    assert 'torch.optim' not in text and '.backward(' not in text
    assert '--mode full' not in text
    assert 'FULL_RUN=NOT_EXECUTED' in text
    assert '_historical_forward_from_hidden' in text
    assert 'installed_g2_loaded_parent' in text
    assert 'phase2_active_mask' in text
    assert 'g2_companion_active_mask' in text
    assert 'torch.cuda.synchronize()' in text
    assert len(tree.body)>0


def test_preflight_no_import_torch_at_module_top_level():
    tree=ast.parse(MODULE.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,ast.Import):
            assert not any(alias.name.startswith('torch') for alias in node.names)
        if isinstance(node,ast.ImportFrom):
            assert not str(node.module).startswith('torch')


def test_unchanged_numeric_tolerances():
    assert g2.NUMERIC_LOGITS_ATOL==1e-5
    assert g2.NUMERIC_CE_ATOL==1e-6
    assert g2.DEV_ROWS_SHA=='b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25'
    assert g2.DEV_ENCODING_SHA=='e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51'
