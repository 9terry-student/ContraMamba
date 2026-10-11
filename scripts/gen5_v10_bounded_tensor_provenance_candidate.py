#!/usr/bin/env python3
"""Gen5 v10 Stage A: CPU-only canonical input tensor manifests, no execution.

A verified digest proves equality to the *provided CPU values* only. It never
proves how those tensors were produced, validates a scientific run, or executes
model/gradient/GPU work. No file I/O is performed by this module or its CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from typing import Any

SCHEMA = "GEN5_V10_BOUNDED_INPUT_MANIFEST_V1"
LAYOUT = "C_CONTIGUOUS_CANONICAL_V1"
BINDING_DOMAIN = b"GEN5_V10_TENSOR_BINDING_V1\x00"
MANIFEST_DOMAIN = b"GEN5_V10_INPUT_MANIFEST_V1\x00"
FINE_CLASSES = ("A_TITLE", "A_NAME", "A_ROLE", "A_PREDICATE", "RESIDUAL")
SIDES = ("CLAIM", "EVIDENCE")
ROLE_KEYS = ("CLAIM", "EVIDENCE", "EOS")
TENSOR_KEYS = tuple(sorted((
    "component", "log_a", "survival", "attention_mask",
    *(f"role_masks.{name}" for name in ROLE_KEYS),
    *(f"fine_masks.{side}.{name}" for side in SIDES for name in FINE_CLASSES),
)))
BINDING_FIELDS = frozenset(("key", "shape", "dtype", "layout", "numel",
                            "byte_count", "content_sha256", "device_at_capture",
                            "binding_sha256"))
MANIFEST_FIELDS = frozenset(("schema_version", "geometry", "tensors", "manifest_sha256",
                             "status", "V10_E0_PROVENANCE", "SCIENTIFIC_EXECUTION",
                             "PRODUCTION"))
GEOMETRY_FIELDS = frozenset(("k", "batch_rows", "sequence_length", "state_width",
                             "intermediate_size", "state_size", "synthetic_cpu_geometry"))
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class ProvenanceBlocked(ValueError):
    """Fail closed on invalid input, metadata, or copied content."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ProvenanceBlocked("GEN5_V10_INPUT_MANIFEST_BLOCKED:" + code)


def canonical_json_bytes(obj: Any) -> bytes:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ProvenanceBlocked("CANONICAL_JSON_INVALID") from exc


def _digest(domain: bytes, value: Any) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def _positive_int(value: Any, label: str) -> int:
    require(type(value) is int and value > 0, label + "_POSITIVE_INT")
    return value


def _shape(value: Any, label: str) -> list[int]:
    require(type(value) is list and len(value) > 0, label + "_RANK")
    return [_positive_int(v, label + "_DIM") for v in value]


def _geometry_from_shape(*, k: int, b: int, seq: int, width: int,
                         intermediate_size: Any, state_size: Any,
                         synthetic_cpu_geometry: bool) -> dict[str, Any]:
    require(type(synthetic_cpu_geometry) is bool, "SYNTHETIC_FLAG_TYPE")
    require(k == 4 and b in (8, 32) and seq == 128, "K_BATCH_SEQUENCE")
    _positive_int(intermediate_size, "INTERMEDIATE_SIZE")
    _positive_int(state_size, "STATE_SIZE")
    require(width == intermediate_size * state_size, "STATE_WIDTH")
    require(synthetic_cpu_geometry or width == 24576, "PRODUCTION_WIDTH")
    return {"k": k, "batch_rows": b, "sequence_length": seq,
            "state_width": width, "intermediate_size": intermediate_size,
            "state_size": state_size, "synthetic_cpu_geometry": synthetic_cpu_geometry}


def _flat_tensors(*, component: Any, log_a: Any, survival: Any,
                  attention_mask: Any, role_masks: Any,
                  fine_masks: Any) -> dict[str, Any]:
    require(type(role_masks) is dict and set(role_masks) == set(ROLE_KEYS),
            "ROLE_MASK_KEYS")
    require(type(fine_masks) is dict and set(fine_masks) == set(SIDES),
            "FINE_MASK_SIDES")
    for side in SIDES:
        require(type(fine_masks[side]) is dict and
                set(fine_masks[side]) == set(FINE_CLASSES),
                "FINE_MASK_CLASSES:" + side)
    flattened = {"component": component, "log_a": log_a,
                 "survival": survival, "attention_mask": attention_mask}
    flattened.update({f"role_masks.{side}": role_masks[side] for side in ROLE_KEYS})
    flattened.update({f"fine_masks.{side}.{name}": fine_masks[side][name]
                      for side in SIDES for name in FINE_CLASSES})
    require(set(flattened) == set(TENSOR_KEYS), "TENSOR_KEYS")
    return flattened


def _validate_tensor_inputs(tensors: dict[str, Any], geometry: dict[str, Any]) -> None:
    import torch

    b, seq, w, k = (geometry[n] for n in
                    ("batch_rows", "sequence_length", "state_width", "k"))
    i, s = geometry["intermediate_size"], geometry["state_size"]
    float_specs = {"component": (k, b, seq, w), "log_a": (b, seq, i, s),
                   "survival": (b, seq, w)}
    mask_specs = {"attention_mask": (b, seq)}
    mask_specs.update({key: (b, seq) for key in TENSOR_KEYS
                       if key.startswith(("role_masks.", "fine_masks."))})
    for key in TENSOR_KEYS:
        t = tensors[key]
        require(isinstance(t, torch.Tensor) and t.device.type == "cpu", "CPU_TENSOR:" + key)
        expected_shape = float_specs.get(key, mask_specs.get(key))
        require(tuple(t.shape) == expected_shape, "TENSOR_SHAPE:" + key)
        expected_dtype = torch.float32 if key in float_specs else torch.bool
        require(t.dtype == expected_dtype, "TENSOR_DTYPE:" + key)
        if key in float_specs:
            require(bool(torch.all(torch.isfinite(t)).item()), "NONFINITE:" + key)
    require(bool(torch.all(tensors["log_a"] <= 0).item()), "POSITIVE_LOG_A")
    require(bool(torch.all(tensors["survival"] >= 0).item()), "NEGATIVE_SURVIVAL")

    # Confirm the mask keys encode a real disjoint role partition, not merely
    # a set of individually well-typed but contradictory bit arrays.
    attention = tensors["attention_mask"]
    role = {name: tensors[f"role_masks.{name}"] for name in ROLE_KEYS}
    require(bool(torch.all((role["CLAIM"] | role["EVIDENCE"] | role["EOS"])
                           == attention).item()), "ROLE_PARTITION_MISMATCH")
    require(not bool(torch.any(role["CLAIM"] & role["EVIDENCE"]).item()) and
            not bool(torch.any(role["CLAIM"] & role["EOS"]).item()) and
            not bool(torch.any(role["EVIDENCE"] & role["EOS"]).item()),
            "ROLE_MASK_OVERLAP")
    require(bool(torch.all(role["EOS"].sum(dim=1) == 1).item()), "EOS_COUNT")
    for side in SIDES:
        assigned = torch.zeros_like(attention)
        for name in FINE_CLASSES:
            this = tensors[f"fine_masks.{side}.{name}"]
            require(not bool(torch.any(assigned & this).item()),
                    "FINE_MASK_OVERLAP:" + side)
            assigned |= this
        require(bool(torch.all(assigned == role[side]).item()),
                "FINE_MASK_PARTITION:" + side)


def _iter_tensor_bytes(tensor: Any, *, dtype: str):
    """Stream C-order canonical bytes; never encode PyTorch storage strides.

    For CPU float32 inputs NumPy's explicit '<f4' view guarantees little-endian
    IEEE-754 encoding. For CPU bool, uint8 conversion normalizes truth values to
    exactly 0/1. No Python float-list serialization or GPU access is used.
    """
    import numpy as np

    # .contiguous() produces a fresh copy for noncontiguous tensors without
    # modifying the caller's storage. detach() only prevents autograd tracking.
    arr = tensor.detach().contiguous().numpy()
    if dtype == "float32":
        arr = arr.astype(np.dtype("<f4"), copy=False)
    else:
        arr = arr.astype(np.uint8, copy=False)
    require(arr.flags.c_contiguous, "CANONICAL_NOT_CONTIGUOUS")
    byte_view = memoryview(arr).cast("B")
    for start in range(0, byte_view.nbytes, 1 << 20):
        yield byte_view[start:start + (1 << 20)]


def _tensor_entry(key: str, tensor: Any) -> dict[str, Any]:
    import torch

    dtype = "float32" if tensor.dtype == torch.float32 else "bool"
    size = int(tensor.numel())
    count = size * (4 if dtype == "float32" else 1)
    hasher = hashlib.sha256()
    seen = 0
    for chunk in _iter_tensor_bytes(tensor, dtype=dtype):
        hasher.update(chunk)
        seen += len(chunk)
    require(seen == count, "CONTENT_BYTE_COUNT:" + key)
    entry = {"key": key, "shape": list(tensor.shape), "dtype": dtype,
             "layout": LAYOUT, "numel": size, "byte_count": count,
             "content_sha256": hasher.hexdigest(), "device_at_capture": "cpu"}
    entry["binding_sha256"] = _digest(BINDING_DOMAIN, entry)
    return entry


def manifest_for_cpu_inputs(*, component: Any, log_a: Any, survival: Any,
                            attention_mask: Any, role_masks: Any,
                            fine_masks: Any, shape: Any,
                            synthetic_cpu_geometry: bool = False) -> dict[str, Any]:
    """Hash supplied CPU tensor *values*. Does not authenticate their source."""
    import torch

    require(isinstance(component, torch.Tensor) and component.ndim == 4,
            "COMPONENT_RANK")
    k, b, seq, width = component.shape
    require(hasattr(shape, "intermediate_size") and hasattr(shape, "state_size"),
            "SHAPE_FIELDS")
    geometry = _geometry_from_shape(k=k, b=b, seq=seq, width=width,
                                   intermediate_size=shape.intermediate_size,
                                   state_size=shape.state_size,
                                   synthetic_cpu_geometry=synthetic_cpu_geometry)
    tensors = _flat_tensors(component=component, log_a=log_a, survival=survival,
                            attention_mask=attention_mask, role_masks=role_masks,
                            fine_masks=fine_masks)
    _validate_tensor_inputs(tensors, geometry)
    entries = [_tensor_entry(key, tensors[key]) for key in TENSOR_KEYS]
    base = {"schema_version": SCHEMA, "geometry": geometry, "tensors": entries,
            "status": "CPU_VALUES_ONLY_SOURCE_UNAUTHENTICATED",
            "V10_E0_PROVENANCE": "NOT_EVALUATED",
            "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED", "PRODUCTION": "BLOCKED"}
    base["manifest_sha256"] = _digest(MANIFEST_DOMAIN, base)
    validate_manifest_structure(base)
    return base


def validate_manifest_structure(manifest: Any) -> dict[str, str]:
    """Recheck canonical metadata and hashes (cannot attest tensor bytes alone)."""
    require(type(manifest) is dict and set(manifest) == MANIFEST_FIELDS,
            "MANIFEST_SCHEMA_KEYS")
    require(manifest["schema_version"] == SCHEMA, "MANIFEST_SCHEMA_VERSION")
    for field, literal in (("status", "CPU_VALUES_ONLY_SOURCE_UNAUTHENTICATED"),
                           ("V10_E0_PROVENANCE", "NOT_EVALUATED"),
                           ("SCIENTIFIC_EXECUTION", "NOT_AUTHORIZED"),
                           ("PRODUCTION", "BLOCKED")):
        require(type(manifest[field]) is str and manifest[field] == literal,
                "FORBIDDEN_PROMOTION:" + field)
    geo = manifest["geometry"]
    require(type(geo) is dict and set(geo) == GEOMETRY_FIELDS, "GEOMETRY_SCHEMA")
    for name in ("k", "batch_rows", "sequence_length", "state_width",
                 "intermediate_size", "state_size"):
        _positive_int(geo[name], "GEOMETRY_" + name)
    checked_geo = _geometry_from_shape(k=geo["k"], b=geo["batch_rows"],
                                       seq=geo["sequence_length"], width=geo["state_width"],
                                       intermediate_size=geo["intermediate_size"],
                                       state_size=geo["state_size"],
                                       synthetic_cpu_geometry=geo["synthetic_cpu_geometry"])
    require(geo == checked_geo, "GEOMETRY_MISMATCH")
    entries = manifest["tensors"]
    require(type(entries) is list and len(entries) == len(TENSOR_KEYS), "TENSOR_SET_SIZE")
    require([e.get("key") if type(e) is dict else None for e in entries]
            == list(TENSOR_KEYS), "TENSOR_KEY_ORDER_OR_IDENTITY")
    k, b, seq, width = geo["k"], geo["batch_rows"], geo["sequence_length"], geo["state_width"]
    expected_shapes = {"component": [k, b, seq, width],
                       "log_a": [b, seq, geo["intermediate_size"], geo["state_size"]],
                       "survival": [b, seq, width]}
    for entry in entries:
        require(type(entry) is dict and set(entry) == BINDING_FIELDS,
                "TENSOR_ENTRY_SCHEMA")
        key = entry["key"]
        require(type(key) is str and key in TENSOR_KEYS, "TENSOR_KEY_STRING")
        expected_dtype = "float32" if key in expected_shapes else "bool"
        require(type(entry["dtype"]) is str and entry["dtype"] == expected_dtype,
                "ENTRY_DTYPE:" + key)
        shape = _shape(entry["shape"], "ENTRY_SHAPE:" + key)
        require(shape == expected_shapes.get(key, [b, seq]), "ENTRY_SHAPE:" + key)
        require(entry["layout"] == LAYOUT and type(entry["layout"]) is str,
                "ENTRY_LAYOUT:" + key)
        require(entry["device_at_capture"] == "cpu" and
                type(entry["device_at_capture"]) is str, "ENTRY_DEVICE:" + key)
        n = math.prod(shape)
        require(type(entry["numel"]) is int and entry["numel"] == n,
                "ENTRY_NUMEL:" + key)
        require(type(entry["byte_count"]) is int and
                entry["byte_count"] == n * (4 if expected_dtype == "float32" else 1),
                "ENTRY_BYTE_COUNT:" + key)
        require(type(entry["content_sha256"]) is str and
                HEX64.fullmatch(entry["content_sha256"]) is not None,
                "ENTRY_CONTENT_SHA256:" + key)
        require(type(entry["binding_sha256"]) is str and
                HEX64.fullmatch(entry["binding_sha256"]) is not None,
                "ENTRY_BINDING_SHA256:" + key)
        binding_base = {field: entry[field] for field in entry if field != "binding_sha256"}
        require(_digest(BINDING_DOMAIN, binding_base) == entry["binding_sha256"],
                "ENTRY_BINDING_DIGEST:" + key)
    require(type(manifest["manifest_sha256"]) is str and
            HEX64.fullmatch(manifest["manifest_sha256"]) is not None, "MANIFEST_DIGEST_FORMAT")
    manifest_base = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(_digest(MANIFEST_DOMAIN, manifest_base) == manifest["manifest_sha256"],
            "MANIFEST_DIGEST_MISMATCH")
    return {"status": "STRUCTURAL_HASHES_VALID_CONTENT_UNAUTHENTICATED",
            "manifest_sha256": manifest["manifest_sha256"]}


def verify_cpu_inputs(manifest: Any, *, component: Any, log_a: Any,
                      survival: Any, attention_mask: Any, role_masks: Any,
                      fine_masks: Any, shape: Any,
                      synthetic_cpu_geometry: bool = False) -> dict[str, str]:
    """Rederive every digest directly from provided CPU tensors and compare."""
    validate_manifest_structure(manifest)
    observed = manifest_for_cpu_inputs(
        component=component, log_a=log_a, survival=survival,
        attention_mask=attention_mask, role_masks=role_masks,
        fine_masks=fine_masks, shape=shape,
        synthetic_cpu_geometry=synthetic_cpu_geometry)
    require(observed == manifest, "CPU_TENSOR_CONTENT_OR_METADATA_MISMATCH")
    return {"status": "MATCHES_PROVIDED_CPU_VALUES_NOT_SOURCE_AUTHENTICATED",
            "manifest_sha256": manifest["manifest_sha256"]}


def compare_manifest_identity(first: Any, second: Any) -> dict[str, str]:
    """Compare two canonical manifest identities, without claiming E0 PASS."""
    validate_manifest_structure(first)
    validate_manifest_structure(second)
    require(first == second, "INPUT_MANIFEST_IDENTITY_MISMATCH")
    return {"status": "MATCHED_MANIFEST_IDS_INPUT_ORIGIN_NOT_AUTHENTICATED",
            "manifest_sha256": first["manifest_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", required=True)
    args = parser.parse_args()
    require(args.plan, "STATIC_PLAN_ONLY")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
            "CUDA_VISIBILITY_MUST_BE_DISABLED")
    print("GEN5_V10_BOUNDED_TENSOR_PROVENANCE_CPU_STATIC_PLAN_PASS")
    print("SCHEMA=" + SCHEMA)
    print("TENSOR_KEYS=" + str(len(TENSOR_KEYS)))
    print("GPU=OFF; MODEL_LOAD=NO; SCIENTIFIC_EXECUTION=NO")
    print("V10_E0_PROVENANCE=NOT_EVALUATED;PRODUCTION=BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
