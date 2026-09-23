# Deterministic paper-figure provenance and output checks.
import sys
sys.dont_write_bytecode = True

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import build_figures as b


def main():
    sources, data = b.load_data()
    manifest = json.loads(
        (b.OUT / "figure_data_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 2
    assert manifest["scientific_head"] == b.BASE_SCIENCE_HEAD
    assert manifest["scientific_authorities"] == {
        "base_evidence": b.BASE_SCIENCE_HEAD,
        "coordinate_free_geometry_snapshot": b.PAPER_EVIDENCE_HEAD,
        "pair_resampling_robustness": b.ROBUSTNESS_HEAD,
    }
    assert manifest["sources"] == sources.sources
    assert manifest["extracted_values"] == sources.values
    assert manifest["plot_data"] == data
    assert manifest["new_statistics"] is False
    assert manifest["model_execution"] is False

    assert data["behavior"] == 0.008332191656033197
    assert data["metrics"] == {
        "Pearson": 0.617548086733788,
        "Spearman": 0.8324696941077123,
        "Sign agreement": 0.8233333333333334,
    }
    assert data["geometry"]["130M"] == {
        "mu_k2": 0.027899337798707836,
        "lambda1": 0.8706181418918275,
    }

    assert (
        data["coordinate_free_geometry"]["xg2"]["cka_matrix"][0][1]
        == 0.7425463019538905
    )
    assert (
        data["coordinate_free_geometry"]["xg4"]["cka_matrix"][2][4]
        == 0.39271700005130383
    )
    assert (
        data["coordinate_free_geometry"]["xg2"]["rsm_matrix"][0][3]
        == 0.2957302277107003
    )

    cf_meta = manifest["sources"][b.CF_RESULT]
    assert cf_meta["authority_commit"] == b.PAPER_EVIDENCE_HEAD
    objective_meta = manifest["sources"][b.OBJECTIVE_BOOTSTRAP]
    assert objective_meta["authority_commit"] == b.ROBUSTNESS_HEAD
    assert "fig4" in objective_meta["figures"]

    # Fail-closed negative controls.
    mutations = [
        lambda d: d["objectives"]["130M"].update(contra=-1),
        lambda d: d["planes"].update({"790M": ["P3", "P5"]}),
        lambda d: d["pairs"].pop(),
        lambda d: d["causal_supported"].update({"370M": False}),
        lambda d: d["chain_supported"].__setitem__(2, False),
        lambda d: d["core"].update({"130M": 0}),
        lambda d: d["geometry"]["130M"].update(lambda1=0),
        lambda d: d["coordinate_free_geometry"]["xg2"]["cka_matrix"][0].__setitem__(1, 0.1),
    ]
    for mutate in mutations:
        wrong = copy.deepcopy(data)
        mutate(wrong)
        try:
            b.validate_values(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid evidence contract accepted")

    from pypdf import PdfReader
    from PIL import Image

    for i, stem in enumerate(b.STEMS):
        for extension in [".pdf", ".png"]:
            path = b.OUT / (stem + extension)
            assert path.is_file() and path.stat().st_size > 1000, path
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == manifest["outputs"][path.name]
            )

        pdf = PdfReader(b.OUT / (stem + ".pdf"))
        assert len(pdf.pages) == 1
        assert float(pdf.pages[0].mediabox.width) == 504
        assert not pdf.pages[0].images

        text = pdf.pages[0].extract_text()

        if i in [0, 2, 3]:
            for scale in b.SCALES:
                assert scale in text, (stem, scale)

        if i == 0:
            assert "Causal role" in text
            assert "Objective-conditioned" in text

        if i == 1:
            assert "Prospective evidence chain" in text
            assert text.count("Supported") == 4
            for forbidden in ["C_PP3", "D_SPEC", "D_NEC", "D_SUF", "D_BEH"]:
                assert forbidden not in text
            assert "exact statistics are in Appendix" in text

        if i == 2:
            assert text.count("Supported") == 5
            assert "XG2 centered linear CKA" in text
            assert "XG4 centered linear CKA" in text
            assert "Historical endpoint labels" in text
            assert "mu_k2" not in text
            assert "lambda1" not in text
            assert "D_SUF" not in text
            assert "D_DOM" not in text
            assert "D_CORE" not in text

        if i == 3:
            assert "All five scales are shown descriptively" in text
            assert "Separate y-scales" in text

        with Image.open(b.OUT / (stem + ".png")) as img:
            assert img.width == 2800 and img.height >= 1800

    for path in [
        Path(__file__),
        Path(b.__file__),
        b.OUT / "README.md",
        b.OUT / "figure_data_manifest.json",
    ]:
        assert all(
            line.rstrip() == line
            for line in path.read_text(encoding="utf-8").splitlines()
        ), path

    print("PASS: 4 PDFs + 4 PNGs; deterministic source/value binding")
    print("PASS: Figure 2 reader-facing causal chain; exact endpoint notation moved to appendix")
    print("PASS: Figure 3 frozen XG2/XG4 coordinate-free CKA matrices")
    print("PASS: historical scalar geometry remains supporting evidence")
    print("PASS: task +,+,+,-,-; vanilla -,-,-,+,+")
    print("PASS: no new scientific statistics or model execution")

    if "--rebuild" in sys.argv:
        paths = [
            b.OUT / name for name in manifest["outputs"]
        ] + [
            b.OUT / "figure_data_manifest.json",
            b.OUT / "README.md",
        ]
        before = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        }
        subprocess.run(
            [sys.executable, "-B", str(Path(b.__file__))],
            cwd=b.ROOT,
            check=True,
        )
        after = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        }
        assert before == after, "Rebuild was not byte deterministic"
        print("PASS: deterministic rebuild; identical figure/manifest/README bytes")


if __name__ == "__main__":
    main()
