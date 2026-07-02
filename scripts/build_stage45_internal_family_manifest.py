"""Build Stage45-B internal family manifest for controlled data only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage45_internal_family_utils import (  # noqa: E402
    build_family_manifest,
    render_manifest_markdown,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object")
            records.append(value)
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an internal-only Stage45-B transformation-family manifest."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/stage45b_internal_family_manifest.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/stage45b_internal_family_manifest.md"),
    )
    parser.add_argument("--min-family-size", type=int, default=20)
    parser.add_argument("--family-field", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_jsonl(args.data)
    manifest = build_family_manifest(
        records,
        input_jsonl=str(args.data),
        min_family_size=args.min_family_size,
        family_field=args.family_field,
    )
    write_json(args.output_json, manifest)
    write_text(args.output_md, render_manifest_markdown(manifest))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
