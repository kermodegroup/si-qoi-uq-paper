r"""Typesetting pass for paper.py: inline math written as text or HTML (λ<sub>F</sub>, K = 200, Beta(k + ½, …)) becomes
\( … \), so the page (MathJax) and the LaTeX source (latex/convert.py) set every symbol as math.  Runs on the assembled
section HTML; existing \( \), \[ \] and every tag except <sub>/<sup> (attributes included) are left alone.  Units
(J/m<sup>2</sup>, Å<sup>3</sup>) stay text.  python bench/si_qoi_uq/artifact/mathify.py --check lists what is left."""
import re

GREEK = {"α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon", "θ": r"\theta", "λ": r"\lambda",
         "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau", "φ": r"\phi",
         "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega", "Γ": r"\Gamma", "Δ": r"\Delta", "Λ": r"\Lambda", "Σ": r"\Sigma",
         "Φ": r"\Phi", "Ψ": r"\Psi"}
GR = "".join(GREEK)
SYM = {"−": "-", "≤": r"\le ", "≥": r"\ge ", "×": r"\times ", "→": r"\to ", "∞": r"\infty", "½": r"\tfrac12", "·": r"\cdot ",
       "±": r"\pm "}
# irregular expressions, written out once (matched on the protected text)
FIXED = [
    ("Beta(k + ½, n − k + ½)", r"\(\mathrm{Beta}(k+\tfrac12,\,n-k+\tfrac12)\)"),
    ("Each count k of n", r"Each count \(k\) of \(n\)"),
    ("Q<sub>k</sub>(λ) = Q̂ + λ(Q<sub>k</sub> − Q̂)", r"\(Q_k(\lambda)=\hat Q+\lambda(Q_k-\hat Q)\)"),
    ("e<sup>T</sup>C<sup>−1</sup>e ≤ χ<sup>2</sup><sub>3</sub>(0.9)", r"\(e^{\mathsf T}C^{-1}e\le\chi^2_3(0.9)\)"),
    ("|ΔF| ≤ σ(χ<sup>2</sup><sub>3</sub>(0.9)/3)<sup>1/2</sup>", r"\(|\Delta F|\le\sigma\,(\chi^2_3(0.9)/3)^{1/2}\)"),
    ("σ = (tr C)<sup>1/2</sup>", r"\(\sigma=(\operatorname{tr}C)^{1/2}\)"),
    ("(N/N<sub>fit</sub>)<sup>β</sup>", r"\((N/N_{\text{fit}})^{\beta}\)"),
    ("N<sub>data</sub> → ∞", r"\(N_{\text{data}}\to\infty\)"),
    ("F<sub>rms</sub> → 0", r"\(F_{\text{rms}}\to0\)"),
    ("P(coverage ≥ 0.95)", r"\(P(\text{coverage}\ge0.95)\)"),
    ("λ = RMS z", r"\(\lambda=\operatorname{RMS}z\)"),
    ("r*<sub>t</sub>", r"\(r^*_t\)"),
    ("DFT − (model + r*)", r"DFT − (model + \(r^*\))"),
    ("correlation order N and total polynomial degree M", r"correlation order \(N\) and total polynomial degree \(M\)"),
    ("bulk modulus B", r"bulk modulus \(B\)"), ("C<sub>44</sub>, B;", r"\(C_{44}\), \(B\);"), ("(x &lt; 1)", r"(\(x&lt;1\))"),
    ("(body order k)", r"(body order \(k\))"), ("BLR a", r"BLR \(a\)"), ("3N forces", r"\(3N\) forces"),
    ("configuration c and observable o", r"configuration \(c\) and observable \(o\)"), ("factor L,", r"factor \(L\),"),
    ("E(V)", r"\(E(V)\)"), ("extrapolated to F → 0", r"extrapolated to \(F_{\text{rms}}\to0\)"), ("d²", r"\(d^2\)"),
    ("log det", r"\(\log\det\)"),
    ("N(0, s<sub>a</sub><sup>2</sup>)", r"\(\mathcal N(0,s_a^2)\)"), ("Γθ", r"\(\Gamma\theta\)"), ("λ²", r"\(\lambda^2\)"),
]
WORDS = [(r"(?<![\w/])σ/tol(?![\w])", lambda m: r"\(\sigma/\text{tol}\)"), (r"(?<![\w*])r\*(?![\w])", lambda m: r"\(r^*\)"),
         (r"q̂", lambda m: r"\(\hat q\)"), (r"Q̂", lambda m: r"\(\hat Q\)"), (r"(?<![\w-])log p(?![\w])", lambda m: r"\(\log p\)"),
         (r"(?<![\w-])([KnNx]) = (\d[\d.]*)", lambda m: r"\(%s=%s\)" % (m.group(1), m.group(2))),
         (r"(?<![\w-])z(?![\w-])", lambda m: r"\(z\)"), (r"(?<![\w.])1σ", lambda m: r"\(1\sigma\)"),
         (r"\|ΔF\|", lambda m: r"\(|\Delta F|\)"), (r"(?<=\x00)n(?=\x00)", lambda m: r"\(n\)")]   # a table header "n"


def tex(s):
    """Text of a sub/superscript or symbol as TeX."""
    s = "".join(GREEK.get(c, SYM.get(c, c)) for c in s)
    return s


def script(c):
    c = c.strip()
    if re.fullmatch(r"[A-Za-z]{2,}", c):
        return r"{\text{%s}}" % c
    t = tex(c)
    return t if len(t) == 1 else "{%s}" % t


BASE = rf"((?<![\w\\])[{GR}]|(?<![\w/\\])[A-Za-z])((?:<su[bp]>[^<]*</su[bp]>)+)"   # a Latin base after "/" is a unit


def _base(m):
    b, scr = m.group(1), m.group(2)
    out = GREEK.get(b, b)
    for kind, c in re.findall(r"<(su[bp])>([^<]*)</su[bp]>", scr):
        out += ("_" if kind == "sub" else "^") + script(c)
    return r"\(%s\)" % out


def _merge(s):
    r"""Join neighbouring math pieces: \(a\) = \(b\), \(a\)/\(b\), \(a\) = 0.3, \(\lambda\)<sup>2</sup>."""
    for _ in range(4):
        s = re.sub(r"\\\)\s*([=≤≥<>])\s*\\\(", lambda m: tex(m.group(1)), s)
        s = re.sub(r"\\\)([+/−\-])\\\(", lambda m: tex(m.group(1)), s)
        s = re.sub(r"\\\)\s*([=≤≥])\s*([−\-]?\d[\d.]*)(?![\d.]*\s*[×%])", lambda m: tex(m.group(1)) + tex(m.group(2)) + r"\)", s)
        s = re.sub(r"\\\)\\\(", "", s)
    return s


def mathify(html):
    keep = []

    def protect(m):
        keep.append(m.group(0))
        return f"\x00{len(keep) - 1}\x00"
    s = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]", protect, html, flags=re.S)
    s = re.sub(r"<(?!/?su[bp]>)[^>]*>", protect, s)
    for a, b in FIXED:
        s = s.replace(a, protect(re.match(".*", b, flags=re.S)))
    for a, b in WORDS:
        s = re.sub(a, lambda m, b=b: protect(re.match(".*", b(m), flags=re.S)), s)
    s = re.sub(r"(\d+(?:\.\d+)?) × 10<sup>([−\-]?\d+)</sup>", lambda m: r"\(%s\times10^{%s}\)" % (m.group(1), tex(m.group(2))), s)
    s = re.sub(r"(?<![\w.])10<sup>([−\-]?\d+)</sup>", lambda m: r"\(10^{%s}\)" % tex(m.group(1)), s)
    s = re.sub(BASE, _base, s)
    s = re.sub(rf"(?<![\w\\])([{GR}])(?![\w-]|-Sn)", lambda m: r"\(%s\)" % GREEK[m.group(1)], s)
    s = re.sub(rf"(?<![\w\\])([{GR}])(?=-[a-z])(?!-Sn)", lambda m: r"\(%s\)" % GREEK[m.group(1)], s)
    for _ in range(3):
        s = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))] if keep[int(m.group(1))].startswith((r"\(", r"\[")) or
                   not keep[int(m.group(1))].startswith("<") else m.group(0), s)
    s = _merge(s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)
    return s


def leftovers(html):
    """Greek letters and sub/superscripts still outside math (units excepted)."""
    s = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]", " ", html, flags=re.S)
    s = re.sub(r"<(?!/?su[bp]>)[^>]*>", " ", s)
    return [" ".join(s[max(0, m.start() - 40): m.start() + 40].split()) for m in
            re.finditer(rf"[{GR}](?!-Sn)|(?<![/Å])\w<su[bp]>", s)]


if __name__ == "__main__":
    import pathlib, sys
    h = (pathlib.Path(__file__).parent / "paper" / "index.html").read_text()
    h = h[h.index("<section"): h.index('id="refs"')]
    for x in leftovers(h):
        print(x)
