"""CPU-only science-runner provenance, scope, aggregation, and reference checks."""
from __future__ import annotations
import ast
import importlib.util
import math
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
MODULE=ROOT/'scripts'/'gen5_g2_two_t4_full_functional_science_99d1b47.py'
SPEC=importlib.util.spec_from_file_location('g2_full_science',MODULE)
full=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)


def fixture_worker(worker: int):
    start,end=full.pre.SHARDS[worker]
    n=end-start
    labels=[i%3 for i in range(start,end)]
    cells={}
    for seed in full.pre.SEEDS:
        for arm in full.pre.ARMS:
            # Smooth deterministic toy logits, identical ZERO across seeds.
            shift={'ZERO':0.,'FULL':0.4,'SHARED_ONLY':0.2,'RESIDUAL_ONLY':0.15,
                   'SHARED_NORMMATCH':0.3,'RESIDUAL_NORMMATCH':0.35}[arm]
            vals=[];ce=[]
            for target in labels:
                vec=[shift if j==target else 0. for j in range(3)]
                vals.append(vec)
                ce.append(math.log(math.exp(shift)+2)-shift)
            cells[f'{seed}/{arm}']={'logits':vals,'ce_per_row':ce}
    return {'worker':worker,'row_range':[start,end],'row_ids':[f'row-{i}' for i in range(start,end)],
            'labels':labels,'cells':cells}


def test_full_matrix_and_shards_exact():
    assert full.FULL_FORWARD_COUNT == 594
    assert len(full.pre.shard_batches(0))==13
    assert len(full.pre.shard_batches(1))==14
    assert 3*6==18
    assert full.EXPECTED_BASE_HEAD=='99d1b477a84c2a93650239288439832a1d830bfe'


def test_full_aggregation_preserves_840_row_weighting():
    result=full.aggregate([fixture_worker(0),fixture_worker(1)])
    assert result['row_count']==840
    assert result['statistical_inference']=='NOT_ESTABLISHED'
    assert result['phase3b_eligibility']=='BLOCKED_UNCHANGED'
    assert len(result['seed_results'])==3
    for s in ('6201','6202','6203'):
        entry=result['seed_results'][s]
        assert len(entry['arms'])==6
        assert entry['full_gain_vs_zero']>0
        assert entry['arms']['ZERO']['gain_vs_zero_ce']==0
        assert math.isclose(entry['arms']['FULL']['full_gain_retained_fraction'],1.,abs_tol=1e-9)
        assert entry['arms']['SHARED_ONLY']['full_gain_retained_fraction']>0
        assert entry['arms']['RESIDUAL_NORMMATCH']['mean_ce'] < entry['arms']['RESIDUAL_ONLY']['mean_ce']
        assert entry['nonadditivity_mean_centered_logit_l2']>=0


def test_negative_gain_and_zero_full_gain_are_not_faked():
    w0,w1=fixture_worker(0),fixture_worker(1)
    for w in (w0,w1):
        for seed in full.pre.SEEDS:
            w['cells'][f'{seed}/FULL']=w['cells'][f'{seed}/ZERO']
    r=full.aggregate([w0,w1])
    for seed in r['seed_results'].values():
        assert seed['arms']['FULL']['retention_defined'] is False
        assert seed['arms']['SHARED_ONLY']['full_gain_retained_fraction'] is None


def test_reject_duplicate_row_and_broken_cell_coverage():
    w0,w1=fixture_worker(0),fixture_worker(1)
    w1['row_ids'][0]=w0['row_ids'][0]
    with pytest.raises(RuntimeError,match='POPULATION_ROW_IDENTITY'):
        full.aggregate([w0,w1])
    w0,w1=fixture_worker(0),fixture_worker(1)
    w1['cells']['6201/FULL']['logits'].pop()
    with pytest.raises(RuntimeError,match='CELL_COVERAGE'):
        full.aggregate([w0,w1])


def test_no_model_torch_import_on_static_import():
    tree=ast.parse(MODULE.read_text(encoding='utf-8'))
    for n in tree.body:
        if isinstance(n,ast.Import):
            assert not any(x.name.startswith('torch') for x in n.names)
        if isinstance(n,ast.ImportFrom):
            assert not str(n.module).startswith('torch')


def test_prescribed_numerical_path_and_no_training():
    src=MODULE.read_text(encoding='utf-8')
    for require in ('installed_g2_loaded_parent','Phase2Layer22MixerWrapper',
        'phase2_active_mask','g2_companion_active_mask','torch.inference_mode()',
        'pre.NUMERIC_LOGITS_ATOL','pre.NUMERIC_CE_ATOL','torch.cuda.synchronize()',
        "'EVERY_BATCH_ZERO_AND_FULL_PASS'",'parent_parameter_fingerprint'):
        assert require in src
    assert 'torch.optim' not in src and '.backward(' not in src
    assert '--mode preflight' not in src
    assert "'train'" not in src
    assert 'pre.EXPECTED_BACKBONE_FORWARDS_PER_BATCH' in src


def test_output_path_contract(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir()
    path=repo/'reports'/'reason_router_gen5_g2_functional_science_runs'/'gen5-g2-full-abc-r1'
    with pytest.raises(RuntimeError,match='RUN_ARTIFACT_ROOT_MISMATCH'):
        full.output_contract(repo,path)
    monkeypatch.setenv('CONTRAMAMBA_RUN_ARTIFACT_ROOT',path.relative_to(repo).as_posix())
    assert full.output_contract(repo,path)==path.relative_to(repo).as_posix()
    path.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='OUTPUT_DIR_EXISTS'):
        full.output_contract(repo,path)


def test_worker_summary_schema_distinct():
    assert full.SCHEMA_WORKER!=full.SCHEMA_SUMMARY
    assert full.SCHEMA_WORKER.endswith('_WORKER_V1')
    assert full.SCHEMA_SUMMARY.endswith('_SUMMARY_V1')
