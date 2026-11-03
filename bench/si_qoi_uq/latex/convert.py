"""LaTeX source for arXiv from the paper generator's rendered page (artifact/paper.py -> artifact/paper/index.html), so
the HTML page and the PDF cannot drift apart: every number, table and caption comes from the same build.

    python bench/si_qoi_uq/artifact/paper.py         # rebuild the page first
    python -m bench.si_qoi_uq.latex.convert [--preprint]   # -> bench/si_qoi_uq/latex/build/{paper.tex, references.bib, figures/}
    cd bench/si_qoi_uq/latex/build && latexmk -pdf paper.tex

Conversion (pandoc 3, from pypandoc_binary):
  - target: J. Chem. Phys., REVTeX 4.2 (aip, jcp; reprint two-column, or --preprint single-column), AIP's numbered
    superscript citations (aipnum4-2);
  - sections keep the page's literal numbers; figures and tables use LaTeX's counters: each caption's "Figure N." /
    "Table N." label becomes \\label{fig:N} / \\label{tab:N} and every "Figure N" / "Table N" in the text a \\ref, so the
    page's sub-labels ("Table 2c") become plain numbers in the PDF; figures (drawn at the full text width) are figure*;
  - \\( \\) and \\[ \\] maths are replaced by tokens before pandoc and restored as raw LaTeX after it;
  - citations: every bracketed author-year citation ("[Bartok et al. 2018]", "[A 2005; B et al. 2019]", with optional
    ", Sec. x" notes) and every narrative one ("Best et al. [2024]") is mapped to references.bib keys (CITE below) and
    written as \\cite (parenthetical: attached to the preceding word; narrative: "Best et al.\\cite{best2024}"); an unmapped
    citation is an error;
  - figures use the paper-context PDFs (figures.py, set_context("paper"));
  - for pdfLaTeX (arXiv), each non-ASCII character is declared with \\DeclareUnicodeCharacter (UNICODE below); an unmapped one
    is an error.
"""
import pathlib, re, shutil, sys

HERE = pathlib.Path(__file__).parent
PAGE = HERE.parent / "artifact" / "paper" / "index.html"
FIG = HERE.parent / "figures"
BUILD = HERE / "build"

CITE = {
    "Allen 1974": "allen1974", "Cook and Weisberg 1982": "cook1982", "Bartók et al. 2018": "bartok2018",
    "Bigi et al. 2024": "bigi2024", "Drautz 2019": "drautz2019", "Gneiting and Raftery 2007": "gneiting2007",
    "Grossfield and Zuckerman 2009": "grossfield2009", "Lelièvre, Rousset and Stoltz 2010": "lelievre2010",
    "Hoeting et al. 1999": "hoeting1999", "Huber 1967": "huber1967", "White 1982": "white1982",
    "Kleijn and van der Vaart 2012": "kleijn2012", "Müller 2013": "muller2013", "Swinburne and Perez 2024": "swinburne2024",
    "MacKinnon and White 1985": "mackinnon1985", "Musil et al. 2019": "musil2019", "Imbalzano et al. 2021": "imbalzano2021",
    "Kellner and Ceriotti 2024": "kellner2024", "Papadopoulos et al. 2008": "papadopoulos2008", "Lei et al. 2018": "lei2018",
    "Vovk, Gammerman and Shafer 2005": "vovk2005", "Tibshirani et al. 2019": "tibshirani2019", "Barber et al. 2021": "barber2021",
    "Witt et al. 2023": "witt2023", "Simmons, Hudson, Swinburne and Kermode, in prep.": "simmons_inprep",
    "Swinburne, POPSRegression": "popsregression", "ace-jax": "acejax", "kermodegroup demos": "demos",
    # narrative "Author [year]" forms
    "Bell and McCaffrey 2002": "bell2002", "Best et al. 2024": "best2024", "Fisher et al. 2026": "fisher2026",
    "Kass and Raftery 1995": "kass1995", "MacKinnon, Nielsen and Webb 2023": "mackinnon2023",
    "Pustejovsky and Tipton 2018": "pustejovsky2018", "Perez et al. 2025": "perez2025",
    "Maliyov, Grigorev and Swinburne 2025": "maliyov2025", "Ho, Ortner and Wang 2026": "ho2026",
    "Podryabinkin and Shapeev 2017": "podryabinkin2017", "Lysogorskiy et al. 2023": "lysogorskiy2023",
}

UNICODE = {
    "λ": r"\ensuremath{\lambda}", "σ": r"\ensuremath{\sigma}", "θ": r"\ensuremath{\theta}", "Σ": r"\ensuremath{\Sigma}",
    "α": r"\ensuremath{\alpha}", "β": r"\ensuremath{\beta}", "γ": r"\ensuremath{\gamma}", "δ": r"\ensuremath{\delta}",
    "ε": r"\ensuremath{\varepsilon}", "ζ": r"\ensuremath{\zeta}", "η": r"\ensuremath{\eta}", "κ": r"\ensuremath{\kappa}",
    "μ": r"\ensuremath{\mu}", "ν": r"\ensuremath{\nu}", "ρ": r"\ensuremath{\rho}", "τ": r"\ensuremath{\tau}", "φ": r"\ensuremath{\phi}",
    "χ": r"\ensuremath{\chi}", "ψ": r"\ensuremath{\psi}", "ω": r"\ensuremath{\omega}", "Δ": r"\ensuremath{\Delta}",
    "Λ": r"\ensuremath{\Lambda}", "Ψ": r"\ensuremath{\Psi}", "Φ": r"\ensuremath{\Phi}", "Ω": r"\ensuremath{\Omega}",
    "π": r"\ensuremath{\pi}", "ϱ": r"\ensuremath{\varrho}", "ϕ": r"\ensuremath{\phi}",
    "−": r"\ensuremath{-}", "×": r"\ensuremath{\times}", "±": r"\ensuremath{\pm}", "≤": r"\ensuremath{\leq}",
    "≥": r"\ensuremath{\geq}", "≈": r"\ensuremath{\approx}", "≠": r"\ensuremath{\neq}", "→": r"\ensuremath{\to}",
    "←": r"\ensuremath{\leftarrow}", "↔": r"\ensuremath{\leftrightarrow}", "∞": r"\ensuremath{\infty}", "√": r"\ensuremath{\surd}",
    "·": r"\ensuremath{\cdot}", "∼": r"\ensuremath{\sim}", "∝": r"\ensuremath{\propto}", "∈": r"\ensuremath{\in}",
    "⊗": r"\ensuremath{\otimes}", "‖": r"\ensuremath{\|}", "⋆": r"\ensuremath{\star}", "∑": r"\ensuremath{\sum}",
    "≫": r"\ensuremath{\gg}", "≪": r"\ensuremath{\ll}", "≡": r"\ensuremath{\equiv}", "∂": r"\ensuremath{\partial}",
    "⁻": r"\ensuremath{^{-}}", "¹": r"\ensuremath{^{1}}", "²": r"\ensuremath{^{2}}", "³": r"\ensuremath{^{3}}",
    "₀": r"\ensuremath{_{0}}", "₁": r"\ensuremath{_{1}}", "₂": r"\ensuremath{_{2}}", "₃": r"\ensuremath{_{3}}",
    "ₖ": r"\ensuremath{_{k}}", "ᵢ": r"\ensuremath{_{i}}", "ⱼ": r"\ensuremath{_{j}}", "ₒ": r"\ensuremath{_{o}}",
    "–": "--", "—": "---", "‘": "`", "’": "'", "“": "``", "”": "''", "…": r"\ldots{}", "\u00a0": "~", "\u2009": r"\,",
    "\u202f": r"\,", "½": r"\ensuremath{\tfrac12}",
    "°": r"\ensuremath{^\circ}", "Å": r"\AA{}", "é": r"\'e", "è": r"\`e", "ó": r"\'o", "á": r"\'a", "ö": r'\"o',
    "ü": r'\"u', "ä": r'\"a', "ç": r"\c{c}", "ñ": r"\~n", "ž": r"\v{z}", "ė": r"\.e", "Ø": r"\O{}", "ø": r"\o{}",
    "̂": r"\^{}", "̃": r"\~{}", "q̂": r"\ensuremath{\hat q}",
}


def _citations(html, keys):
    """Replace mapped bracketed and narrative citations with tokens (pandoc would escape LaTeX); returns (html, tokens)."""
    tok = {}

    def new(latex, prefix="CITETOKEN"):                                 # NARRTOKEN: narrative, keeps its space
        k = f"{prefix}{len(tok):04d}X"
        tok[k] = latex
        return k

    def cite_item(item):
        item = re.sub(r"\s+", " ", item).strip()
        name, _, post = item.partition(", Sec.")
        post = f"Sec.~{post.strip()}" if post else ""
        if name in CITE:
            return CITE[name], post
        if "; scene" in item and item.split(";")[0] in CITE:             # "kermodegroup demos; scene lj-cr3"
            return CITE[item.split(";")[0]], item.split(";")[1].strip()
        return None, None

    def bracket(m):
        inside = m.group(1)
        whole = re.sub(r"\s+", " ", inside).strip()
        if whole in CITE or whole.startswith("kermodegroup demos"):
            parts = [whole]
        else:
            parts = [p_ for p_ in whole.split("; ")]
        plain, noted = [], []
        for p_ in parts:
            k, post = cite_item(p_)
            if k is None:
                return m.group(0)                                       # not a citation (data, intervals, ...)
            keys.add(k)
            (noted.append((k, post)) if post else plain.append(k))
        latex = (rf"\cite{{{','.join(plain)}}}" if plain else "") + "".join(rf"\cite[{p_}]{{{k}}}" for k, p_ in noted)
        return " " + new(latex)                                         # marker: the space before it is removed
    html = re.sub(r"(?<![\\\w])\[([A-Za-zÀ-ž][^\[\]<>]{1,140}?)\]", bracket, html)

    def narrative(m):
        name = re.sub(r"\s+", " ", m.group(1)).strip() + " " + m.group(2)
        if name not in CITE:
            return m.group(0)
        keys.add(CITE[name])
        return new(re.sub(r"\s+", " ", m.group(1)).strip() + rf"\cite{{{CITE[name]}}}", "NARRTOKEN")
    html = re.sub(r"([A-Z][\w'’\-ö]+(?:,\s+[A-Z][\w'’\-]+)*(?:\s+and\s+[A-Z][\w'’\-]+)?(?:\s+et\s+al\.)?)\s+\[(\d{4})\]",
                  narrative, html)
    return html, tok


def _math(html, tok):
    """MathJax maths -> placeholder tokens, restored as raw LaTeX after pandoc (HTML entities unescaped)."""
    import html as _h

    def new(latex):
        k = f"MATHTOKEN{len(tok):05d}X"
        tok[k] = latex
        return k
    def display(m):
        inner = _h.unescape(m.group(1))
        if re.search(r",\s*\\q?quad", inner):                         # side-by-side equations: one per line in a
            inner = re.sub(r",\s*\\q?quad\s*", lambda _: ",\\\\\n", inner)  # two-column page
            return new("\\begin{gather*}" + inner.strip() + "\n\\end{gather*}")
        if "\n" in inner.strip() and len(inner) > 80:                     # a long equation broken in the source:
            a_, b_ = inner.strip().split("\n", 1)                       # break it there (multline)
            return new("\\begin{multline*}" + a_.strip() + "\\\\\n" + b_.strip() + "\n\\end{multline*}")
        return new("\\[" + inner + "\\]")
    html = re.sub(r"\\\[(.*?)\\\]", display, html, flags=re.S)
    html = re.sub(r"\\\((.*?)\\\)", lambda m: new("\\(" + _h.unescape(m.group(1)) + "\\)"), html, flags=re.S)
    return html


def _tables(tex):
    """pandoc's longtables -> table floats with a tabular, scaled down to the text width only when wider (\\fitwidth):
    the page's tables are all shorter than a page, and some (Table 1) are wider than the text."""
    out, i = [], 0
    while True:
        a = tex.find("\\begin{longtable}", i)
        if a < 0:
            out.append(tex[i:])
            return "".join(out)
        b = tex.index("\\end{longtable}", a) + len("\\end{longtable}")
        t = tex[a:b]
        spec = t[t.index("[]{") + 2:t.index("\n")].strip()
        cap, lab = "", ""
        if "\\caption{" in t:
            c0 = t.index("\\caption{") + len("\\caption{")
            cap = t[c0:t.index("}\\tabularnewline", c0)]
            mm = re.match(r"\s*Table (\w+)\.\s*", cap)
            if mm:
                lab, cap = f"\\label{{tab:{mm.group(1)}}}", cap[mm.end():]
        head = ""
        if "\\endfirsthead" in t:
            h0 = t.index("\\toprule\\noalign{}\n") + len("\\toprule\\noalign{}\n")
            head = t[h0:t.index("\\midrule\\noalign{}\n\\endfirsthead", h0)]
        body = t[t.index("\\endlastfoot\n") + len("\\endlastfoot\n"):t.index("\\end{longtable}")] if "\\endlastfoot" in t else ""
        rule = "\\toprule\n" + (head + "\\midrule\n" if head else "")
        out.append(tex[i:a])
        env = "table*" if len(re.findall(r"[lcr]", spec)) > 5 else "table"     # wide tables span both columns
        out.append(f"\\begin{{{env}}}[htbp]\n\\centering\\small\\setlength{{\\tabcolsep}}{{3pt}}\n"
                   + (f"\\caption{{{cap}}}{lab}\n" if cap else "")
                   + f"\\fitwidth{{\\begin{{tabular}}{spec}\n{rule}{body}\\bottomrule\n\\end{{tabular}}}}\n\\end{{{env}}}")
        i = b


def main():
    import pypandoc
    s = PAGE.read_text()
    title = re.search(r"<h1>(.*?)</h1>", s, re.S).group(1).strip()
    author = re.search(r'<p class="author">(.*?)</p>', s, re.S).group(1).strip()
    affil = re.search(r'<p class="affil">(.*?)</p>', s, re.S).group(1).strip()
    body = s[s.index('<section id="abstract">'):s.index('<section id="refs">')]
    body = re.sub(r'<span class="chip[^"]*">.*?</span>', "", body)
    keys = set()
    body, tok = _citations(body, keys)
    mtok = {}
    body = _math(body, mtok)
    tex = pypandoc.convert_text(body, "latex", format="html", extra_args=["--shift-heading-level-by=-1", "--wrap=preserve"])
    tex = re.sub(r"\s+(CITETOKEN\d+X)", r"\1", tex)               # superscript citations hug the preceding word
    for k, v in list(tok.items()) + list(mtok.items()):
        tex = tex.replace(k, v)
    left = re.findall(r"\[[A-Z][^\]]{0,60}\d{4}\]", tex)
    if left:
        sys.exit(f"unmapped citations: {sorted(set(left))[:10]}")
    # figures: the paper-context PDFs at the full text width (figure*), labelled from the page's caption number
    tex = re.sub(r"\\pandocbounded\{\\includegraphics\[[^\]]*\]\{([^}]+)\.png\}\}",
                 lambda m: rf"\includegraphics[width=\textwidth]{{figures/{m.group(1)}.pdf}}", tex)
    tex = tex.replace("\\begin{figure}", "\\begin{figure*}[tbp]").replace("\\end{figure}", "\\end{figure*}")
    tex = re.sub(r"\\caption\{Figure (\w+)\.\s*(.*?)\}\n(?=\\end\{figure\*\})",
                 lambda m: rf"\caption{{{m.group(2)}}}\label{{fig:{m.group(1)}}}" + "\n", tex, flags=re.S)
    tex = _tables(tex)
    labels = set(re.findall(r"\\label\{((?:fig|tab):\w+)\}", tex))

    def refs(m):
        kind = "fig" if m.group(1).startswith("Figure") else "tab"
        def one(t):
            return rf"\ref{{{kind}:{t}}}" if f"{kind}:{t}" in labels else t
        return m.group(1) + "~" + re.sub(r"\d+[a-z]?", lambda n: one(n.group(0)), m.group(2))
    tex = re.sub(r"\b(Figures?|Tables?) ((?:\d+[a-z]?)(?:(?:, | and | or |--)\d+[a-z]?)*)\b", refs, tex)
    bad = re.findall(r"(?:Figure|Table)s?~(?![^\n]*\\ref)\d", tex)
    if bad:
        sys.exit(f"cross-references without a label: {bad[:5]}")
    # abstract: REVTeX wants it before \\maketitle
    ma = re.search(r"\\section\{Abstract\s*\}\\label\{[^}]*\}\s*(.*?)(?=\\section)", tex, flags=re.S)
    abstract = ma.group(1).strip()
    tex = tex[:ma.start()] + tex[ma.end():]
    used = sorted({c for c in tex + abstract if ord(c) > 127})
    missing = [c for c in used if c not in UNICODE]
    if missing:
        sys.exit("unmapped characters: " + " ".join(f"{c!r} U+{ord(c):04X}" for c in missing))
    uni = "\n".join(rf"\DeclareUnicodeCharacter{{{ord(c):04X}}}{{{UNICODE[c]}}}" for c in used)   # kernel, no package
    layout = "preprint" if "--preprint" in sys.argv else "reprint"
    pre = rf"""\documentclass[aip,jcp,{layout},amsmath,amssymb,floatfix]{{revtex4-2}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern,microtype}}
\usepackage{{graphicx,xcolor}}
\usepackage{{booktabs,array,calc}}
\IfFileExists{{multirow.sty}}{{\usepackage{{multirow}}}}{{}}
\providecommand{{\multirow}}[3]{{#3}}
\usepackage[hidelinks]{{hyperref}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
\newcommand{{\real}}[1]{{#1}}
\setcounter{{secnumdepth}}{{0}}                       % headings carry the page's own numbers
\providecommand{{\lt}}{{<}}\providecommand{{\gt}}{{>}}  % MathJax spellings used in the page
\newsavebox{{\fitbox}}                               % a table at its natural width, or scaled to the line width
\newcommand{{\fitwidth}}[1]{{\sbox{{\fitbox}}{{#1}}\ifdim\wd\fitbox>\linewidth\resizebox{{\linewidth}}{{!}}{{\usebox{{\fitbox}}}}\else\usebox{{\fitbox}}\fi}}
{uni}
\begin{{document}}
\title{{{title}}}
\author{{{author}}}
\affiliation{{{affil}}}
\begin{{abstract}}
{abstract}
\end{{abstract}}
\maketitle
"""
    post = "\n\\bibliography{references}\n\\end{document}\n"
    BUILD.mkdir(exist_ok=True)
    (BUILD / "figures").mkdir(exist_ok=True)
    for f in re.findall(r"figures/([\w\-]+\.pdf)", tex):
        shutil.copy(FIG / f, BUILD / "figures" / f)
    shutil.copy(HERE / "references.bib", BUILD / "references.bib")
    (BUILD / "paper.tex").write_text(pre + tex + post)
    bib = (HERE / "references.bib").read_text()
    unknown = sorted(k for k in keys if "{" + k + "," not in bib)
    print(f"paper.tex: {len(tex)} chars, {len(tok)} citations ({len(keys)} keys), {len(used)} unicode characters mapped"
          + (f"; keys missing from references.bib: {unknown}" if unknown else ""))


if __name__ == "__main__":
    main()
