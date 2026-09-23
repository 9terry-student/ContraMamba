"""Narrow deterministic provenance/value/output checks; no statistical testing."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import subprocess
from pathlib import Path

import build_figures as b


def main():
    sources, data = b.load_data()
    manifest = json.loads((b.OUT / "figure_data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scientific_head"] == b.HEAD
    assert manifest["sources"] == sources.sources, "Source hashes/figure assignments changed"
    assert manifest["extracted_values"] == sources.values, "Source locator/value mismatch"
    assert manifest["plot_data"] == data, "Plot data differs from frozen extraction"
    assert list(data["objectives"]) == b.SCALES
    assert data["behavior"] == 0.008332191656033197, "Frozen D_BEH changed"
    assert data["metrics"] == {
        "Pearson": 0.617548086733788,
        "Spearman": 0.8324696941077123,
        "Sign agreement": 0.8233333333333334,
    }, "Frozen 130M metrics changed"
    assert data["geometry"]["130M"] == {
        "mu_k2": 0.027899337798707836,
        "lambda1": 0.8706181418918275,
    }, "Frozen historical 130M geometry changed"
    for key in ["130M.mu_k2", "130M.lambda1"]:
        assert sources.values[key]["source"] == manifest["extracted_values"][key]["source"]
    # Deterministic negative controls prove contracts fail closed.
    import copy
    for mutate in [lambda d: d["objectives"]["130M"].update(contra=-1),
                   lambda d: d["planes"].update({"790M": ["P3", "P5"]}),
                   lambda d: d["pairs"].pop(),
                   lambda d: d["causal_supported"].update({"370M": False}),
                   lambda d: d["core"].update({"130M": 0}),
                   lambda d: d["geometry"]["130M"].update(lambda1=0)]:
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
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["outputs"][path.name]
        pdf = PdfReader(b.OUT / (stem + ".pdf"))
        assert len(pdf.pages) == 1
        assert float(pdf.pages[0].mediabox.width) == 504
        assert not pdf.pages[0].images, "Raster content unexpectedly embedded in vector PDF"
        text = pdf.pages[0].extract_text()
        if i in [0, 2, 3]:
            for scale in b.SCALES:
                assert scale in text, (stem, scale)
        if i == 0:
            assert "Task-margin" in text and "Behavioral" not in text
            assert "Within each scale:" in text and "same frozen native" in text
        if i == 1:
            for value in data["chain_display_means"]:
                assert value in text, (stem, "Missing full frozen mean", value)
            assert text.count("N = 300") == 5  # four cards plus unchanged Panel B
            assert "not magnitude-comparable" in text
            for value in ["0.61755", "0.83247", "0.82333"]:
                assert value in text, (stem, value)
        if i == 2:
            assert "Frozen mean" not in text and text.count("Supported") == 5
            assert "Q(selected restored) - Q(coefficient-matched control)" in text
            assert "historical response-blind 130M" in text
            assert "residual criterion failed" in text
        if i == 3:
            assert "post-primary" in text and "Separate y-scales" in text
        with Image.open(b.OUT / (stem + ".png")) as img:
            assert img.width == 2800 and img.height >= 1800
    for path in [Path(__file__), Path(b.__file__), b.OUT / "README.md", b.OUT / "figure_data_manifest.json"]:
        assert all(line.rstrip() == line for line in path.read_text(encoding="utf-8").splitlines()), path
    print("PASS: 4 PDFs + 4 PNGs; one vector page each, 7-inch width, 400-dpi PNGs")
    print("PASS: all five scales in figures 1/3/4; exact selected/control plane map")
    print("PASS: Contra = +,+,+,-,-; vanilla = -,-,-,+,+")
    print("PASS: 130M Pearson = 0.617548086733788; Spearman = 0.8324696941077123; sign agreement = 0.8233333333333334")
    print("PASS: 300 unfiltered source pairs; pinned source hashes and all output hashes")
    print("PASS: negative controls reject changed signs, plane identities, missing rows, fabricated 130M D_CORE")
    print("PASS: no new scientific statistics computed")
    print("PASS: common Figure 3 causal contrast, five source-bound support statuses, historical 130M geometry; no Fig. 3 mean row")
    if "--rebuild" in sys.argv:
        paths = [b.OUT / name for name in manifest["outputs"]] + [b.OUT / "figure_data_manifest.json", b.OUT / "README.md"]
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        subprocess.run([sys.executable, "-B", str(Path(b.__file__))], cwd=b.ROOT, check=True)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        assert before == after, "Rebuild was not byte deterministic"
        print("PASS: deterministic rebuild; identical bytes for 8 figures, manifest, and README")


if __name__ == "__main__":
    main()
