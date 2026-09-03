from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import preflight_longterm_o0c_runtime_source_provenance as preflight


class TorchVersion(str):
    pass


MAMBA_PASS = """\
class MambaMixer:
    def forward(self, hidden_states):
        if hidden_states.device.type == "cpu":
            return self.slow_forward(hidden_states)
        else:
            raise RuntimeError("non-cpu backend unresolved")

    def slow_forward(self, hidden_states):
        ssm_state = hidden_states.new_zeros(1)
        conv_state = hidden_states.new_zeros(1)
        collected_hidden_states = []
        for token in hidden_states:
            ssm_state = ssm_state + token
            collected_hidden_states.append(ssm_state)
        return {"hidden_states": collected_hidden_states}
"""

CACHE_PASS = """\
class Cache:
    def update(self, ssm_state):
        self.ssm_state = ssm_state
"""


def write_package(tmp_path: Path, mamba_text: str = MAMBA_PASS, cache_text: str = CACHE_PASS) -> dict[str, Path]:
    root = tmp_path / "site-packages"
    pkg = root / "transformers"
    mamba_dir = pkg / "models" / "mamba"
    mamba_dir.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models").mkdir(exist_ok=True)
    (pkg / "models" / "__init__.py").write_text("", encoding="utf-8")
    (mamba_dir / "__init__.py").write_text("", encoding="utf-8")
    mamba_path = mamba_dir / "modeling_mamba.py"
    cache_path = pkg / "cache_utils.py"
    mamba_path.write_text(mamba_text, encoding="utf-8", newline="")
    cache_path.write_text(cache_text, encoding="utf-8", newline="")
    return {"root": root, "pkg": pkg, "mamba": mamba_path, "cache": cache_path}


class FakeDist:
    version = "5.0.0"

    def __init__(self, package_root: Path, files=None):
        self.package_root = package_root
        self.files = files if files is not None else [Path("transformers/__init__.py")]

    def locate_file(self, path):
        return self.package_root.parent / Path(path)


def spec(path: Path, name: str):
    return importlib.machinery.ModuleSpec(name, loader=None, origin=str(path))


def pkg_spec(path: Path):
    return importlib.machinery.ModuleSpec("transformers", loader=None, origin=str(path / "__init__.py"))


def providers(paths: dict[str, Path], dist_pkg: Path | None = None):
    dist = FakeDist(dist_pkg or paths["pkg"])

    def dist_provider(name: str):
        assert name == "transformers"
        return dist

    def spec_provider(name: str):
        mapping = {
            "transformers": pkg_spec(paths["pkg"]),
            "transformers.models.mamba.modeling_mamba": spec(paths["mamba"], name),
            "transformers.cache_utils": spec(paths["cache"], name),
        }
        return mapping.get(name)

    return dist_provider, spec_provider


def test_exact_expected_runtime_pass_and_str_subclass():
    actual = preflight.runtime_versions(
        version_lookup=lambda name: {"numpy": "2.0.2", "torch": TorchVersion("2.10.0+cpu"), "transformers": "5.0.0"}[name],
        python_version=lambda: "3.12.13",
    )
    assert actual["torch"] == "2.10.0+cpu"
    preflight.check_runtime(
        {"python": "3.12.13", "numpy": "2.0.2", "torch": "2.10.0+cpu", "transformers": "5.0.0"},
        actual,
    )


@pytest.mark.parametrize(
    "bad_key,bad_value",
    [
        ("python", "3.12.12"),
        ("numpy", "2.0.1"),
        ("torch", "2.10.0"),
        ("transformers", "5.0.1"),
    ],
)
def test_wrong_runtime_versions_block(bad_key, bad_value):
    expected = {"python": "3.12.13", "numpy": "2.0.2", "torch": "2.10.0+cpu", "transformers": "5.0.0"}
    actual = dict(expected)
    actual[bad_key] = bad_value
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.check_runtime(expected, actual)
    assert exc.value.status == "BLOCKED_RUNTIME_VERSION_MISMATCH"


def test_runtime_unavailable_blocks():
    def missing(_name):
        raise importlib.metadata.PackageNotFoundError("numpy")

    import importlib.metadata

    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.runtime_versions(version_lookup=missing, python_version=lambda: "3.12.13")
    assert exc.value.status == "BLOCKED_RUNTIME_VERSION_UNAVAILABLE"


def test_valid_reconciled_distribution_import_root(tmp_path):
    paths = write_package(tmp_path)
    (tmp_path / "repo").mkdir()
    dist_provider, spec_provider = providers(paths)
    result = preflight.resolve_transformers_sources(tmp_path / "repo", dist_provider, spec_provider)
    assert result.import_root == paths["pkg"].resolve(strict=True)
    assert result.distribution_root == paths["pkg"].resolve(strict=True)


def test_valid_descendant_and_prefix_trap(tmp_path):
    paths = write_package(tmp_path)
    assert preflight.is_descendant_or_equal(paths["mamba"].resolve(), paths["pkg"].resolve())
    evil = tmp_path / "site-packages" / "transformers_evil" / "models" / "mamba"
    evil.mkdir(parents=True)
    evil_source = evil / "modeling_mamba.py"
    evil_source.write_text(MAMBA_PASS, encoding="utf-8")
    assert not preflight.is_descendant_or_equal(evil_source.resolve(), paths["pkg"].resolve())


def test_dot_segments_normalize_to_descendant(tmp_path):
    paths = write_package(tmp_path)
    dotted = paths["pkg"] / "." / "models" / ".." / "models" / "mamba" / "modeling_mamba.py"
    assert preflight.canonical_path(dotted) == paths["mamba"].resolve(strict=True)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink unsupported")
def test_lexical_inside_resolved_outside_symlink_blocks(tmp_path):
    paths = write_package(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_source = outside / "modeling_mamba.py"
    outside_source.write_text(MAMBA_PASS, encoding="utf-8")
    paths["mamba"].unlink()
    try:
        paths["mamba"].symlink_to(outside_source)
    except OSError:
        pytest.skip("host disallows symlink creation")
    dist_provider, spec_provider = providers(paths)
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.resolve_transformers_sources(tmp_path / "repo", dist_provider, spec_provider)
    assert exc.value.status == "BLOCKED_TRANSFORMERS_SOURCE_SHADOWING"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink unsupported")
def test_lexical_outside_resolved_inside_symlink_can_reconcile(tmp_path):
    paths = write_package(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_source = outside / "modeling_mamba.py"
    try:
        outside_source.symlink_to(paths["mamba"])
    except OSError:
        pytest.skip("host disallows symlink creation")

    def spec_provider(name: str):
        return {
            "transformers": pkg_spec(paths["pkg"]),
            "transformers.models.mamba.modeling_mamba": spec(outside_source, name),
            "transformers.cache_utils": spec(paths["cache"], name),
        }.get(name)

    result = preflight.resolve_transformers_sources(tmp_path / "repo", providers(paths)[0], spec_provider)
    assert result.mamba_source == paths["mamba"].resolve(strict=True)


def test_repo_local_shadow_blocks(tmp_path):
    paths = write_package(tmp_path / "repo")
    dist_provider, spec_provider = providers(paths)
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.resolve_transformers_sources(tmp_path / "repo", dist_provider, spec_provider)
    assert exc.value.status == "BLOCKED_TRANSFORMERS_SOURCE_SHADOWING"


def test_pythonpath_out_of_distribution_shadow_blocks(tmp_path):
    (tmp_path / "repo").mkdir()
    dist_paths = write_package(tmp_path / "dist")
    shadow_paths = write_package(tmp_path / "shadow")
    dist_provider, _ = providers(dist_paths)
    _, shadow_spec_provider = providers(shadow_paths)
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.resolve_transformers_sources(tmp_path / "repo", dist_provider, shadow_spec_provider)
    assert exc.value.status == "BLOCKED_TRANSFORMERS_SOURCE_SHADOWING"


def test_ambiguous_distribution_roots_block(tmp_path):
    (tmp_path / "repo").mkdir()
    paths = write_package(tmp_path)
    other = write_package(tmp_path / "other")

    class AmbiguousDist(FakeDist):
        def __init__(self, package_root: Path):
            super().__init__(
                package_root,
                files=[Path("transformers/__init__.py"), Path("transformers/models/mamba/modeling_mamba.py")],
            )

        def locate_file(self, path):
            if Path(path).parts[-3:] == ("models", "mamba", "modeling_mamba.py"):
                return other["pkg"] / "models" / "mamba" / "modeling_mamba.py"
            return paths["pkg"].parent / Path(path)

    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.resolve_transformers_sources(tmp_path / "repo", lambda _: AmbiguousDist(paths["pkg"]), providers(paths)[1])
    assert exc.value.status == "BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS"


def test_missing_source_and_unresolvable_path_blocks(tmp_path):
    paths = write_package(tmp_path)
    (tmp_path / "repo").mkdir()

    def missing_spec_provider(name: str):
        if name == "transformers.cache_utils":
            return None
        return providers(paths)[1](name)

    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.resolve_transformers_sources(tmp_path / "repo", providers(paths)[0], missing_spec_provider)
    assert exc.value.status == "BLOCKED_SOURCE_FILE_UNRESOLVED"

    paths["mamba"].unlink()
    with pytest.raises(preflight.PreflightBlocked) as exc2:
        preflight.resolve_transformers_sources(tmp_path / "repo", *providers(paths))
    assert exc2.value.status == "BLOCKED_SOURCE_PATH_CANONICALIZATION_FAILED"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows filesystem case reconciliation")
def test_windows_case_reconciliation(tmp_path):
    paths = write_package(tmp_path)
    assert preflight.same_path(Path(str(paths["pkg"]).upper()).resolve(strict=True), paths["pkg"].resolve(strict=True))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX case distinction")
def test_posix_case_distinction(tmp_path):
    lower = tmp_path / "transformers"
    upper = tmp_path / "Transformers"
    lower.mkdir()
    upper.mkdir()
    assert not preflight.same_path(lower.resolve(), upper.resolve())


def test_raw_bytes_identity_no_normalization(tmp_path):
    path = tmp_path / "source.py"
    data = b"alpha\r\nbeta\n# final"
    path.write_bytes(data)
    facts = preflight.raw_source_identity(path, "m")
    assert facts.sha256 == hashlib.sha256(data).hexdigest()
    assert facts.bytes == len(data)
    assert facts.lf_count == 2
    assert facts.cr_count == 1
    assert facts.final_lf is False
    path.write_bytes(data + b"\n")
    assert preflight.raw_source_identity(path, "m").final_lf is True


def test_ast_symbol_binding_and_stable_span(tmp_path):
    paths = write_package(tmp_path)
    mamba = preflight.raw_source_identity(paths["mamba"], preflight.SOURCE_KEYS["mamba"])
    cache = preflight.raw_source_identity(paths["cache"], preflight.SOURCE_KEYS["cache"])
    locations = preflight.bind_symbol_locations(mamba, cache)
    assert set(locations) == set(preflight.SYMBOL_KEYS)
    assert locations["mixer_forward_dispatch"]["qualname"] == "MambaMixer.forward"
    assert locations["mixer_forward_dispatch"]["start_line"] == 2
    assert locations["mixer_forward_dispatch"]["source_sha256"] == mamba.sha256


def test_ast_missing_and_ambiguous_symbols_block(tmp_path):
    paths = write_package(tmp_path, mamba_text="class MambaMixer:\n    pass\n")
    mamba = preflight.raw_source_identity(paths["mamba"], preflight.SOURCE_KEYS["mamba"])
    cache = preflight.raw_source_identity(paths["cache"], preflight.SOURCE_KEYS["cache"])
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.bind_symbol_locations(mamba, cache)
    assert exc.value.status == "BLOCKED_REQUIRED_SYMBOL_UNRESOLVED"

    duplicate = MAMBA_PASS + "\nclass MambaMixer:\n    def slow_forward(self):\n        return None\n"
    paths = write_package(tmp_path / "dup", mamba_text=duplicate)
    mamba = preflight.raw_source_identity(paths["mamba"], preflight.SOURCE_KEYS["mamba"])
    cache = preflight.raw_source_identity(paths["cache"], preflight.SOURCE_KEYS["cache"])
    with pytest.raises(preflight.PreflightBlocked) as exc2:
        preflight.bind_symbol_locations(mamba, cache)
    assert exc2.value.status == "BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS"


def test_parse_and_decode_failures(tmp_path):
    bad_syntax = tmp_path / "bad.py"
    bad_syntax.write_text("def nope(:\n", encoding="utf-8")
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.raw_source_identity(bad_syntax, "m")
    assert exc.value.status == "BLOCKED_SOURCE_DECODE_OR_PARSE_FAILURE"
    bad_utf8 = tmp_path / "bad_utf8.py"
    bad_utf8.write_bytes(b"\xff")
    with pytest.raises(preflight.PreflightBlocked) as exc2:
        preflight.raw_source_identity(bad_utf8, "m")
    assert exc2.value.status == "BLOCKED_SOURCE_DECODE_OR_PARSE_FAILURE"


def test_semantics_and_backend_classifications():
    assert preflight.classify_recurrent_semantics(MAMBA_PASS) == "SOURCE_SUPPORTS_O0C_CONVENTION"
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics("ssm_state = plausible")
    assert exc.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"
    with pytest.raises(preflight.PreflightBlocked) as exc2:
        preflight.classify_recurrent_semantics("ssm_state = pre_consumption")
    assert exc2.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"
    assert preflight.classify_backend(MAMBA_PASS) == "BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN"
    assert preflight.classify_backend("class MambaMixer:\n    def slow_forward(self):\n        pass\n") == "BACKEND_SEQUENTIAL_PRESENT_SELECTION_UNPROVEN"
    assert preflight.classify_backend("x = 1\n") == "BACKEND_STATICALLY_UNRESOLVED"


def test_string_literal_adversary_does_not_prove_recurrent_or_backend():
    text = '"""ssm_state post_consumption hidden cpu_sequential_static"""\n'
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics(text)
    assert exc.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"
    assert preflight.classify_backend(text) == "BACKEND_STATICALLY_UNRESOLVED"


def test_unrelated_function_adversary_requires_review():
    text = """\
class MambaMixer:
    def forward(self, hidden_states):
        return self.slow_forward(hidden_states)

    def slow_forward(self, hidden_states):
        return hidden_states

def unrelated_helper(hidden_states):
    ssm_state = hidden_states.new_zeros(1)
    hidden = []
    for token in hidden_states:
        ssm_state = ssm_state + token
        hidden.append(ssm_state)
    return hidden
"""
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics(text)
    assert exc.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"


def test_unrelated_assignment_adversary_does_not_bind_update():
    text = """\
class MambaMixer:
    def forward(self, hidden_states):
        return self.slow_forward(hidden_states)

    def slow_forward(self, hidden_states):
        ssm_state = hidden_states.new_zeros(1)
        conv_state = hidden_states.new_zeros(1)
        collected_hidden_states = []
        ssm_state_post_consumption_hidden = ssm_state
        for token in hidden_states:
            prior = ssm_state
            collected_hidden_states.append(prior)
        return {"hidden_states": collected_hidden_states}
"""
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics(text)
    assert exc.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"


def test_duplicate_plausible_update_adversary_blocks_ambiguous():
    text = MAMBA_PASS.replace(
        "ssm_state = ssm_state + token\n            collected_hidden_states.append(ssm_state)",
        "ssm_state = ssm_state + token\n            ssm_state = ssm_state + token\n            collected_hidden_states.append(ssm_state)",
    )
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics(text)
    assert exc.value.status == "BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS"


def test_prior_state_readout_adversary_does_not_support():
    text = MAMBA_PASS.replace(
        "ssm_state = ssm_state + token\n            collected_hidden_states.append(ssm_state)",
        "collected_hidden_states.append(ssm_state)\n            ssm_state = ssm_state + token",
    )
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics(text)
    assert exc.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"


def test_self_only_update_adversary_does_not_support():
    text = """\
class MambaMixer:
    def forward(self, hidden_states):
        return self.slow_forward(hidden_states)

    def slow_forward(self, hidden_states):
        ssm_state = hidden_states.new_zeros(1)
        conv_state = hidden_states.new_zeros(1)
        for i in range(len(hidden_states)):
            ssm_state = ssm_state
            output = use(ssm_state)
        return output
"""
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics(text)
    assert exc.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"


def test_constant_overwrite_adversary_does_not_support():
    text = """\
class MambaMixer:
    def forward(self, hidden_states):
        return self.slow_forward(hidden_states)

    def slow_forward(self, hidden_states):
        ssm_state = hidden_states.new_zeros(1)
        conv_state = hidden_states.new_zeros(1)
        for i in range(len(hidden_states)):
            ssm_state = hidden_states.new_zeros(1)
            output = use(ssm_state)
        return output
"""
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics(text)
    assert exc.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"


def test_current_step_only_adversary_does_not_support():
    text = """\
class MambaMixer:
    def forward(self, hidden_states):
        return self.slow_forward(hidden_states)

    def slow_forward(self, hidden_states):
        ssm_state = hidden_states.new_zeros(1)
        conv_state = hidden_states.new_zeros(1)
        for i in range(len(hidden_states)):
            x_t = hidden_states[i]
            ssm_state = project(x_t)
            output = use(ssm_state)
        return output
"""
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics(text)
    assert exc.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"


def test_prior_state_only_adversary_does_not_support():
    text = """\
class MambaMixer:
    def forward(self, hidden_states):
        return self.slow_forward(hidden_states)

    def slow_forward(self, hidden_states):
        ssm_state = hidden_states.new_zeros(1)
        conv_state = hidden_states.new_zeros(1)
        decay = 0.5
        for i in range(len(hidden_states)):
            ssm_state = decay * ssm_state
            output = use(ssm_state)
        return output
"""
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics(text)
    assert exc.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"


def test_direct_current_step_and_prior_state_dependency_supports():
    text = """\
class MambaMixer:
    def forward(self, hidden_states):
        if hidden_states.device.type == "cpu":
            return self.slow_forward(hidden_states)
        else:
            raise RuntimeError("non-cpu backend unresolved")

    def slow_forward(self, hidden_states):
        ssm_state = hidden_states.new_zeros(1)
        conv_state = hidden_states.new_zeros(1)
        decay = 0.5
        for i in range(len(hidden_states)):
            x_t = hidden_states[i]
            ssm_state = decay * ssm_state + project(x_t)
            output = use(ssm_state)
        return output
"""
    assert preflight.classify_recurrent_semantics(text) == "SOURCE_SUPPORTS_O0C_CONVENTION"


def test_transitive_current_step_and_prior_state_dependency_supports():
    text = """\
class MambaMixer:
    def forward(self, hidden_states):
        if hidden_states.device.type == "cpu":
            return self.slow_forward(hidden_states)
        else:
            raise RuntimeError("non-cpu backend unresolved")

    def slow_forward(self, hidden_states):
        ssm_state = hidden_states.new_zeros(1)
        conv_state = hidden_states.new_zeros(1)
        decay = 0.5
        for i in range(len(hidden_states)):
            x_t = hidden_states[i]
            input_term = project(x_t)
            candidate = decay * ssm_state + input_term
            ssm_state = candidate
            output = use(ssm_state)
        return output
"""
    assert preflight.classify_recurrent_semantics(text) == "SOURCE_SUPPORTS_O0C_CONVENTION"


def test_wrong_index_adversary_does_not_support_current_step_dependency():
    text = """\
class MambaMixer:
    def forward(self, hidden_states):
        return self.slow_forward(hidden_states)

    def slow_forward(self, hidden_states):
        ssm_state = hidden_states.new_zeros(1)
        conv_state = hidden_states.new_zeros(1)
        decay = 0.5
        for i in range(len(hidden_states)):
            x_t = hidden_states[0]
            ssm_state = decay * ssm_state + project(x_t)
            output = use(ssm_state)
        return output
"""
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics(text)
    assert exc.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"


def test_unrelated_indexed_tensor_does_not_satisfy_update_dependency():
    text = """\
class MambaMixer:
    def forward(self, hidden_states):
        return self.slow_forward(hidden_states)

    def slow_forward(self, hidden_states):
        ssm_state = hidden_states.new_zeros(1)
        conv_state = hidden_states.new_zeros(1)
        decay = 0.5
        for i in range(len(hidden_states)):
            unrelated = hidden_states[i]
            ssm_state = decay * ssm_state
            output = use(ssm_state)
        return output
"""
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.classify_recurrent_semantics(text)
    assert exc.value.status == "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"


def test_backend_token_sentinel_adversary_does_not_prove_static_cpu_path():
    text = """\
class MambaMixer:
    def forward(self, hidden_states):
        backend = "cpu_sequential_static"
        return self.slow_forward(hidden_states)

    def slow_forward(self, hidden_states):
        pass
"""
    assert preflight.classify_backend(text) == "BACKEND_SEQUENTIAL_PRESENT_SELECTION_UNPROVEN"


def test_backend_optional_path_blocks_static_proof():
    text = MAMBA_PASS.replace(
        "if hidden_states.device.type == \"cpu\":",
        "if use_mamba_kernels:\n            return selective_scan_fn(hidden_states)\n        if hidden_states.device.type == \"cpu\":",
    )
    assert preflight.classify_backend(text) == "BACKEND_ASSOCIATIVE_OR_KERNEL_PATH_MAY_INTERVENE"


def test_build_artifact_blocks_unresolved_backend(tmp_path):
    text = MAMBA_PASS.replace(
        "if hidden_states.device.type == \"cpu\":\n            return self.slow_forward(hidden_states)\n        else:\n            raise RuntimeError(\"non-cpu backend unresolved\")",
        "return self.slow_forward(hidden_states)",
    )
    paths = write_package(tmp_path, mamba_text=text)
    resolution = preflight.SourceResolution(
        paths["pkg"].resolve(),
        paths["pkg"].resolve(),
        paths["mamba"].resolve(),
        paths["cache"].resolve(),
        "5.0.0",
    )
    runtime = {"python": "3.12.13", "numpy": "2.0.2", "torch": "2.10.0+cpu", "transformers": "5.0.0"}
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.build_artifact(runtime, runtime, resolution)
    assert exc.value.status == "BLOCKED_BACKEND_PATH_UNRESOLVED"


def test_artifact_schema_deterministic_serialization_and_collision(tmp_path):
    paths = write_package(tmp_path)
    resolution = preflight.SourceResolution(
        paths["pkg"].resolve(),
        paths["pkg"].resolve(),
        paths["mamba"].resolve(),
        paths["cache"].resolve(),
        "5.0.0",
    )
    runtime = {"python": "3.12.13", "numpy": "2.0.2", "torch": "2.10.0+cpu", "transformers": "5.0.0"}
    artifact = preflight.build_artifact(runtime, runtime, resolution)
    assert set(artifact) == {
        "backend_static_classification",
        "cache_source",
        "expected_runtime",
        "mamba_source",
        "notes",
        "o0c_full_sequence_capture_feasibility",
        "o0c_state_indexing_compatibility",
        "optimized_kernel_availability",
        "preflight_status",
        "runtime",
        "schema_version",
        "source_resolution",
        "symbol_locations",
    }
    assert artifact["preflight_status"] in preflight.STATUSES
    symbol_keys = {"module", "qualname", "source_file_key", "source_sha256", "start_line", "end_line"}
    for location in artifact["symbol_locations"].values():
        assert set(location) == symbol_keys
    data1 = preflight.serialize_artifact(artifact)
    data2 = preflight.serialize_artifact(artifact)
    assert data1 == data2
    assert data1.endswith(b"\n")
    assert b"\n  " in data1
    decoded = data1.decode("utf-8")
    assert list(json.loads(decoded).keys()) == sorted(json.loads(decoded).keys())
    for forbidden in ("timestamp", "hostname", "username", "uuid", "branch"):
        assert forbidden not in decoded.lower()
    with pytest.raises(preflight.PreflightBlocked):
        preflight.serialize_artifact({"x": float("nan")})
    output = tmp_path / "artifact.json"
    preflight.publish_artifact(output, data1)
    assert output.read_bytes() == data1
    with pytest.raises(preflight.PreflightBlocked) as exc:
        preflight.publish_artifact(output, data1)
    assert exc.value.status == "BLOCKED_OUTPUT_COLLISION"


def test_cli_rejects_missing_required_flags():
    with pytest.raises(SystemExit):
        preflight.parse_args(["--output", "x"])


def test_safety_static_scan_production_code():
    source = Path(preflight.__file__).read_text(encoding="utf-8")
    forbidden_patterns = [
        r"from_pretrained\s*\(",
        r"\.forward\s*\(",
        r"huggingface_hub",
        r"hf_hub_download",
        r"snapshot_download",
        r"subprocess",
        r"\bpip\b",
        r"\bconda\b",
        r"\buv\s+pip\b",
        r"\bsite-packages\b.*write",
        r"\bstartswith\s*\(",
    ]
    for pattern in forbidden_patterns:
        assert not re.search(pattern, source)
    ast.parse(source)


def test_no_forbidden_runtime_invocation_by_subprocess(tmp_path):
    output = tmp_path / "blocked.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(preflight.__file__)),
            "--output",
            str(output),
            "--expected-python",
            "0",
            "--expected-numpy",
            "0",
            "--expected-torch",
            "0",
            "--expected-transformers",
            "0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode != 0
    assert "BLOCKED_RUNTIME_VERSION_MISMATCH" in completed.stdout
    assert not output.exists()
