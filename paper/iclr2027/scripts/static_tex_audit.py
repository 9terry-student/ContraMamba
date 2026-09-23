from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
main = (root / "main.tex").read_text(encoding="utf-8")
appendix = (root / "appendix.tex").read_text(encoding="utf-8")
bib = (root / "references.bib").read_text(encoding="utf-8")
tex = main + "\n" + appendix
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cites = set()
for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", tex):
    cites.update(s.strip() for s in group.split(","))
labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", tex))
figures = re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", tex)
errors = []
if cites - keys:
    errors.append("missing citation keys: " + repr(sorted(cites - keys)))
if refs - labels:
    errors.append("unresolved reference labels: " + repr(sorted(refs - labels)))
if len(labels) != len(re.findall(r"\\label\{([^}]+)\}", tex)):
    errors.append("duplicate labels")
if len(keys) != len(re.findall(r"@\w+\{([^,]+),", bib)):
    errors.append("duplicate bibliography keys")
for figure in figures:
    if not (root / "figures" / figure).is_file():
        errors.append("missing figure: " + figure)
for path in [root / "main.tex", root / "appendix.tex", root / "references.bib"]:
    s = path.read_text(encoding="utf-8")
    for line_no, line in enumerate(s.splitlines(), 1):
        if line.rstrip() != line:
            errors.append(f"{path.name}:{line_no} trailing whitespace")
    nonascii = sorted({hex(ord(c)) for c in s if ord(c) > 127})
    if nonascii:
        errors.append(path.name + " non-ASCII: " + ", ".join(nonascii))
for name, body in [("main", main), ("appendix", appendix)]:
    stack = []
    for idx, ch in enumerate(body):
        if ch in "{}" and (idx == 0 or body[idx - 1] != "\\"):
            if ch == "{":
                stack.append(idx)
            elif stack:
                stack.pop()
            else:
                errors.append(name + " has extra closing brace")
                break
    if stack:
        errors.append(name + " has unclosed brace")
    for env in ["abstract", "equation", "align", "figure", "table", "tabular", "enumerate"]:
        if body.count(r"\begin{" + env + "}") != body.count(r"\end{" + env + "}"):
            errors.append(name + " has unbalanced " + env + " environment")
print("citations:", sorted(cites))
print("labels:", len(labels), "references:", len(refs))
print("figures:", figures)
print("errors:", errors)
raise SystemExit(bool(errors))
