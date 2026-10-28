"""Builds Paper.docx (Word) from main.tex, so the Word and LaTeX versions say exactly the same thing.
Needs pandoc (on Colab: apt-get install pandoc). Run from the paper folder:
    python word/build_docx.py
Figure, table and equation numbers come from main.aux if a LaTeX compile left one, otherwise from the
order in main.tex (the same order IEEEtran uses).
The numbers come from results_macros.tex and tables/*.tex, so rerun it after  python -m src.pipeline paper."""
import re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; PAPER = HERE.parent
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def labels_from_source():
    """Numbers in order of appearance: equations, figures, tables and subsections of main.tex."""
    src = strip_comments((PAPER / "main.tex").read_text()); out = {}
    src = src[src.index("\\begin{document}"):]
    n = {"eq": 0, "fig": 0, "tab": 0, "prop": 0}; sec = sub = 0
    for m in re.finditer(r"\\section\{|\\subsection\{|\\begin\{(equation|figure\*?|table\*?|proposition)\}|\\label\{([^}]+)\}", src):
        tok = m.group(0)
        if tok == "\\section{":
            sec += 1; sub = 0
        elif tok == "\\subsection{":
            sub += 1
        elif m.group(1):
            kind = {"equation": "eq", "proposition": "prop"}.get(m.group(1), "fig" if m.group(1).startswith("figure") else "tab")
            n[kind] += 1; last = kind
        elif m.group(2):
            name = m.group(2)
            if name.startswith("sec:"):
                out[name] = f"{sec}.{sub}"
            elif name.startswith("tab:"):
                out[name] = ROMAN[n["tab"]]
            elif name.startswith("fig:"):
                out[name] = str(n["fig"])
            elif name.startswith("prop:"):
                out[name] = str(n["prop"])
            else:
                out[name] = str(n["eq"])
    return out


def labels():
    aux_p = PAPER / "main.aux"
    if not aux_p.exists() or aux_p.stat().st_mtime < (PAPER / "main.tex").stat().st_mtime:
        return labels_from_source()             # no compiled LaTeX, or it is older than main.tex
    aux = (PAPER / "main.aux").read_text()
    out = {}
    for name, num, anchor in re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{[^}]*\}\{.*?\}\{([^}]*)\}", aux):
        if name.startswith("tab:"):
            out[name] = ROMAN[int(num)]
        elif name.startswith("sec:"):
            sec = re.search(r"subsection\.(\d+)\.(\d+)", anchor)
            out[name] = f"{sec.group(1)}.{sec.group(2)}" if sec else num   # Word headings are numbered 3.7 etc.
        else:
            out[name] = num
    return out


def strip_comments(t):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in t.splitlines())


def unwrap(t, cmd):
    """Remove a wrapper like \\resizebox{..}{..}{BODY} keeping BODY (balanced braces)."""
    while True:
        i = t.find(cmd)
        if i < 0:
            return t
        j = i + len(cmd); args = []
        while len(args) < 3:
            assert t[j] == "{", t[i:i + 60]
            depth, k = 1, j + 1
            while depth:
                depth += {"{": 1, "}": -1}.get(t[k], 0); k += 1
            args.append(t[j + 1:k - 1]); j = k
        t = t[:i] + args[2] + t[j:]


def reference_doc():
    """Times New Roman 10 pt, IEEE-like headings, 1-inch margins."""
    ref = HERE / "reference.docx"
    if ref.exists():
        return
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    ref.write_bytes(subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], capture_output=True).stdout)
    d = Document(ref)
    for sec in d.sections:
        sec.left_margin = sec.right_margin = Inches(0.9); sec.top_margin = sec.bottom_margin = Inches(0.9)
    for st in d.styles:
        try:
            st.font.name = "Times New Roman"; st.font.color.rgb = RGBColor(0, 0, 0)
            rpr = st.element.get_or_add_rPr(); rf = rpr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts")
            if rf is not None:
                for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                    rf.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}" + a.split(":")[1], "Times New Roman")
        except Exception:
            pass
    sizes = {"Normal": 10, "Body Text": 10, "First Paragraph": 10, "Compact": 10, "Title": 22, "Author": 11,
             "Abstract": 9, "Heading 1": 11, "Heading 2": 10, "Heading 3": 10, "Caption": 8, "Image Caption": 8,
             "Table Caption": 8, "Bibliography": 8}
    for name, sz in sizes.items():
        try:
            st = next(x for x in d.styles if x.name == name); st.font.size = Pt(sz)
            if name.startswith("Heading"):
                st.font.bold = name != "Heading 2"; st.font.italic = name == "Heading 2"
                st.paragraph_format.space_before = Pt(8); st.paragraph_format.space_after = Pt(4)
            if name in ("Body Text", "First Paragraph"):
                st.paragraph_format.space_after = Pt(4); st.paragraph_format.alignment = 3   # justify
        except (KeyError, StopIteration):
            pass
    d.save(ref)


def main():
    reference_doc()
    L = labels()
    src = strip_comments((PAPER / "main.tex").read_text())
    body = src[src.index("\\begin{document}") + len("\\begin{document}"):src.index("\\bibliographystyle")]
    def macro_def(name):
        """Body of \\newcommand{\\NAME}{...} in the preamble (balanced braces)."""
        i = src.index("\\newcommand{\\" + name + "}") + len(name) + 14
        i = src.index("{", i); depth, k = 1, i + 1
        while depth:
            depth += {"{": 1, "}": -1}.get(src[k], 0); k += 1
        return src[i + 1:k - 1].strip().lstrip("%").strip()
    title = " ".join(macro_def("PaperTitle").split())
    front = ("\\textbf{\\emph{Abstract}}---" + macro_def("PaperAbstract") + "\n\n\\textbf{\\emph{Index Terms}}---"
             + macro_def("PaperKeywords") + "\n\n")
    body = body.replace("\\makefront", front).replace("\\title{\\PaperTitle}", "")
    body = body.replace("\\begin{paperproof}", "\\emph{Proof.} ").replace("\\end{paperproof}", " $\\square$")
    body = re.sub(r"\\author\{.*?\n\}\n", "", body, flags=re.S).replace("\\maketitle", "")
    body = re.sub(r"\\title\{.*?\}\n", "", body, count=1, flags=re.S)
    body = body.replace("\\begin{abstract}", "\\textbf{\\emph{Abstract}}---").replace("\\end{abstract}", "")
    body = body.replace("\\begin{IEEEkeywords}", "\\textbf{\\emph{Index Terms}}---").replace("\\end{IEEEkeywords}", "")
    body = body.replace("\\begin{IEEEproof}", "\\emph{Proof.} ").replace("\\end{IEEEproof}", " $\\square$")
    count = iter(range(1, 100))
    body = re.sub(r"\\begin\{proposition\}\s*(\\label\{[^}]+\})?\s*",
                  lambda m: f"\\textbf{{Proposition {next(count)}.}} \\emph{{", body).replace("\\end{proposition}", "}")
    # tables: inline the generated files and drop layout-only commands
    body = re.sub(r"\\input\{(tables/[^}]+)\}", lambda m: (PAPER / (m.group(1) + ".tex")).read_text(), body)
    body = unwrap(body, "\\resizebox")
    body = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", body)
    body = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}|\\footnotesize|\\centering", "", body)
    # figures: TikZ drawings become the rendered PNGs; plots use the PNG copies
    body = body.replace("\\input{fig_pipeline}", "\\includegraphics[width=6.5in]{word/fig_pipeline.png}")
    body = body.replace("\\input{fig_circuits}", "\\includegraphics[width=3.3in]{word/fig_circuits.png}")
    body = re.sub(r"\\includegraphics\[width=\\(textwidth|columnwidth)\]\{figures/([^}]+)\}",
                  lambda m: f"\\includegraphics[width={'6.5' if m.group(1) == 'textwidth' else '3.4'}in]{{figures/{m.group(2)}.png}}", body)
    body = body.replace("figure*", "figure").replace("table*", "table")

    # captions get their numbers in IEEE style
    def cap(m):
        env, inner = m.group(1), m.group(2)
        lab = re.search(r"\\label\{([^}]+)\}", inner)
        n = L.get(lab.group(1), "?") if lab else "?"
        pre = f"Fig. {n}. " if env == "figure" else f"TABLE {n}. "
        return m.group(0).replace("\\caption{", "\\caption{" + pre, 1)
    body = re.sub(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", cap, body, flags=re.S)
    # equation numbers and cross references
    def eqnum(m):
        lab = re.search(r"\\label\{([^}]+)\}", m.group(0)); n = L.get(lab.group(1), "") if lab else ""
        return m.group(0).replace("\\end{equation}", f"\\qquad ({n})\n\\end{{equation}}")
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", eqnum, body, flags=re.S)
    body = re.sub(r"\\eqref\{([^}]+)\}", lambda m: f"({L.get(m.group(1), '?')})", body)
    body = re.sub(r"\\ref\{([^}]+)\}", lambda m: L.get(m.group(1), "?"), body)
    body = re.sub(r"\\label\{[^}]+\}", "", body)
    for a, b in (("\\Bigl", "\\left"), ("\\Bigr", "\\right"), ("\\bigl", "\\left"), ("\\bigr", "\\right")):
        body = body.replace(a, b)

    macros = strip_comments((PAPER / "results_macros.tex").read_text())
    text = strip_comments((PAPER / "results_text.tex").read_text()) if (PAPER / "results_text.tex").exists() else ""
    import yaml
    cfgp = PAPER.parent / "configs" / "config.yaml"
    info = yaml.safe_load(open(cfgp)).get("paper", {}) if cfgp.exists() else {}
    authors = ", ".join(info.get("authors", ["Author One", "Author Two", "Author Three"]))
    affil = ", ".join(x for x in [info.get("department"), info.get("institution"), info.get("city"), info.get("country")] if x) \
        or "Department, University, City, Country"
    ack = " ".join(x for x in [info.get("funding", ""), info.get("ai_disclosure", "")] if x)
    body = body.replace("\\paperack{\\PaperAckText}", "\\section*{Acknowledgment}\n" + ack +
                        " Code, cached LLM answers and the preregistration are available at \\RepoURL.")
    repo = info.get("repo_url", "https://github.com/USERNAME/qkernel-icl")
    body = body.replace("\\input{authors}", "")
    pre = r"""\documentclass{article}
\usepackage{amsmath,amssymb,graphicx,booktabs}
\newcommand{\res}[1]{#1}
""" + macros.replace("\\providecommand{\\res}[1]{#1}", "") + r"""
\newcommand{\fillin}[1]{\textbf{[FILL: #1]}}
""" + text.replace("\\providecommand{\\fillin}[1]{{\\color{red}[#1]}}", "") + r"""
\newcommand{\RepoURL}{\url{""" + repo + r"""}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\ket}[1]{\lvert #1 \rangle}
\DeclareMathOperator{\Tr}{Tr}
\title{""" + title + r"""}
\author{""" + authors + " (" + affil + r""")}
\begin{document}
\maketitle
"""
    tex = pre + body + "\n\\section*{References}\n\\end{document}\n"
    (HERE / "_paper_for_word.tex").write_text(tex)
    cmd = ["pandoc", str(HERE / "_paper_for_word.tex"), "-f", "latex", "-o", str(PAPER / "Paper.docx"),
           "--citeproc", "--bibliography", str(PAPER / "refs.bib"), "--csl", str(HERE / "ieee.csl"),
           "--reference-doc", str(HERE / "reference.docx"), "--number-sections", "--resource-path", str(PAPER)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=PAPER)
    print(r.stderr[-2000:]) if r.returncode or r.stderr else None
    if r.returncode:
        sys.exit("pandoc failed")
    polish(PAPER / "Paper.docx")
    print("wrote", PAPER / "Paper.docx")


def polish(path):
    """Booktabs-like table rules, highlighted placeholders, centred figures, roman section numbers."""
    from docx import Document
    from docx.enum.text import WD_COLOR_INDEX, WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from docx.shared import Pt
    d = Document(path)
    for t in d.tables:
        tblPr = t._tbl.tblPr
        b = OxmlElement("w:tblBorders")
        for edge, sz in (("top", 12), ("bottom", 12)):
            e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "000000"); b.append(e)
        tblPr.append(b)
        jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); tblPr.append(jc)
        if t.rows:                                       # rule under the header row
            for cell in t.rows[0].cells:
                tcPr = cell._tc.get_or_add_tcPr(); bd = OxmlElement("w:tcBorders"); e = OxmlElement("w:bottom")
                e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000"); bd.append(e); tcPr.append(bd)
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    for r in p.runs:
                        r.font.size = Pt(8)
    paras = list(d.paragraphs) + [p for t in d.tables for row in t.rows for c in row.cells for p in c.paragraphs]
    for p in paras:
        inside = False
        for r in p.runs:
            if "[FILL:" in r.text:
                inside = True
            if inside or "XYZ" in r.text:
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            if inside and "]" in r.text.split("[FILL:")[-1]:
                inside = False
        if p._p.xpath(".//pic:pic"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.save(path)


if __name__ == "__main__":
    main()
