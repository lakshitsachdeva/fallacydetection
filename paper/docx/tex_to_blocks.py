"""Flatten the LaTeX paper into JSON blocks for the Word build (build_docx.js)."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
FIG = PAPER.parent / "figures" / "final"


def read(name):
    return (PAPER / name).read_text(encoding="utf-8")


macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", read("generated.tex")))
src = read("fallacylens_ieee.tex")
for _ in range(3):  # results_body.tex itself \inputs the table files
    src = re.sub(r"\\input\{(\w+)\.tex\}", lambda m: read(m.group(1) + ".tex") if m.group(1) != "generated" else "", src)
src = re.sub(r"(?<!\\)%.*", "", src)

# Citation numbers in order of first appearance (IEEEtran unsorted style), cross-checked with the .bbl.
bbl = read("fallacylens_ieee.bbl")
bib_keys = re.findall(r"\\bibitem\{([^}]+)\}", bbl)
num = {k: i + 1 for i, k in enumerate(bib_keys)}


def cite(m):
    ns = sorted(num[k.strip()] for k in m.group(1).split(","))
    if len(ns) > 2 and ns == list(range(ns[0], ns[-1] + 1)):
        return f"[{ns[0]}]–[{ns[-1]}]"
    return ", ".join(f"[{n}]" for n in ns)


label_names = {}


def inline(t):
    """LaTeX inline -> mini markup: **bold**, //italic//, ``code``."""
    t = re.sub(r"~?\\cite\{([^}]+)\}", lambda m: " " + cite(m), t)
    t = re.sub(r"\\(\w+)\{\}", lambda m: macros.get(m.group(1), m.group(0)), t)
    t = re.sub(r"\\(f|sd|acc|clim|climAcc|dev|off|time|n)([A-Z]\w*)\b", lambda m: macros.get(m.group(1) + m.group(2), m.group(0)), t)
    t = re.sub(r"\\ref\{([^}]+)\}", lambda m: label_names.get(m.group(1), "?"), t)
    t = re.sub(r"\\url\{([^}]+)\}", r"\1", t)
    t = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", t)
    t = re.sub(r"\\emph\{([^{}]*)\}", r"//\1//", t)
    t = re.sub(r"\\texttt\{([^{}]*)\}", r"``\1``", t)
    reps = [(r"\,$\pm$\,", " ± "), (r"$\pm$", "±"), (r"\pm", "±"), ("``", "“"), ("''", "”"), ("---", "—"), ("--", "–"),
            (r"\ldots\ ", "… "), (r"\ldots", "…"), (r"\%", "%"), (r"\&", "&"), ("{,}", ","), (r"\,", " "),
            ("~", " "), (r"\times", "×"), (r"$<$", "<"), (r"\Delta", "Δ"), (r"\ ", " ")]
    for a, b in reps:
        t = t.replace(a, b)
    t = t.replace("“", "``") if False else t
    t = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("{", "").replace("}", "").replace("^", "^").replace("\\", ""), t)
    t = t.replace("10^-3", "10⁻³").replace("5 × 10^-5", "5 × 10⁻⁵").replace("10^−3", "10⁻³")
    t = re.sub(r"\\[a-zA-Z]+\*?", "", t).replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", t).strip()


# Pre-assign figure/table numbers from labels in order.
fig_n = tab_n = 0
for env, body in re.findall(r"\\begin\{(figure\*?|table\*?)\}(.*?)\\end\{\1\}", src, re.S):
    lab = re.search(r"\\label\{([^}]+)\}", body)
    if env.startswith("figure"):
        fig_n += 1
        label_names[lab.group(1)] = str(fig_n)
    else:
        tab_n += 1
        label_names[lab.group(1)] = ["I", "II", "III", "IV"][tab_n - 1]

body = src.split(r"\maketitle", 1)[1].split(r"\bibliographystyle", 1)[0]
blocks = []
fig_n = tab_n = 0
pos = 0
token = re.compile(r"\\begin\{(abstract|IEEEkeywords|itemize|figure\*?|table\*?)\}(.*?)\\end\{\1\}|\\section\*?\{([^}]*)\}|\\subsection\{([^}]*)\}", re.S)


def flush_text(text):
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para:
            blocks.append({"type": "p", "text": inline(para)})


for m in token.finditer(body):
    flush_text(body[pos:m.start()])
    pos = m.end()
    env, inner, sec, sub = m.groups()
    if sec is not None:
        blocks.append({"type": "h1", "text": inline(sec), "numbered": not m.group(0).startswith(r"\section*")})
    elif sub is not None:
        blocks.append({"type": "h2", "text": inline(sub)})
    elif env == "abstract":
        blocks.append({"type": "abstract", "text": inline(inner)})
    elif env == "IEEEkeywords":
        blocks.append({"type": "keywords", "text": inline(inner)})
    elif env == "itemize":
        blocks.append({"type": "list", "items": [inline(i) for i in inner.split(r"\item")[1:]]})
    elif env.startswith("figure"):
        fig_n += 1
        cap = inline(re.search(r"\\caption\{(.*)\}\s*\\label", inner, re.S).group(1))
        img = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", inner)
        path = str((PAPER / img.group(1)).resolve()) if img else str(HERE / f"fig{fig_n}.png")
        blocks.append({"type": "figure", "n": fig_n, "caption": cap, "image": path, "wide": env.endswith("*")})
    else:
        tab_n += 1
        cap = inline(re.search(r"\\caption\{(.*?)\}\s*\\label", inner, re.S).group(1))
        tab = inner.split(r"\toprule", 1)[1].split(r"\bottomrule", 1)[0]
        rows = []
        for line in tab.replace(r"\midrule", "").split(r"\\"):
            line = line.strip()
            if line:
                rows.append([inline(c) for c in line.split("&")])
        blocks.append({"type": "table", "n": label_names[re.search(r"\\label\{([^}]+)\}", inner).group(1)],
                       "caption": cap, "rows": rows, "wide": env.endswith("*")})
flush_text(body[pos:])

# References from the .bbl, in citation order.
refs = []
for item in re.split(r"\\bibitem\{[^}]+\}", bbl)[1:]:
    item = item.split(r"\end{thebibliography}")[0]
    item = item.replace(r"\newblock", "").replace(r"\BIBentryALTinterwordspacing", "").replace(r"\BIBentrySTDinterwordspacing", "")
    item = re.sub(r"\\bibinfo\{[^}]*\}\{([^}]*)\}", r"\1", item)
    item = item.replace('{\\"o}', "ö").replace('{\\"u}', "ü").replace("{\\'e}", "é").replace("{\\'o}", "ó") \
        .replace("{\\~n}", "ñ").replace("{\\^o}", "ô").replace("{\\^A}", "Â").replace("{\\L}", "Ł").replace("\\L", "Ł")
    refs.append(inline(item))

title = inline(re.search(r"\\title\{(.*?)\}\n", src, re.S).group(1))
authors = inline(re.search(r"\\IEEEauthorblockN\{(.*?)\}", src).group(1))
affil = [inline(x.replace(r"\{", "LBRACE").replace(r"\}", "RBRACE")).replace("LBRACE", "{").replace("RBRACE", "}") for x in re.search(r"\\IEEEauthorblockA\{(.*?)\}\n\}", src, re.S).group(1).split(r"\\")]
(HERE / "paper.json").write_text(json.dumps({"title": title, "authors": authors, "affil": affil, "blocks": blocks,
                                             "refs": refs}, ensure_ascii=False, indent=1), encoding="utf-8")
print(len(blocks), "blocks,", len(refs), "refs")
