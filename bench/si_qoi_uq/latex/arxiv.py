"""arXiv submission package from the LaTeX build: paper.tex, its BibTeX output paper.bbl (arXiv does not run BibTeX) and
the figures it includes, as build/arxiv/ and build/si-qoi-uq-arxiv.tar.gz; then a clean-room pdfLaTeX compile of that
directory alone (no BibTeX), which must give the same page count as build/paper.pdf.

    python -m bench.si_qoi_uq.latex.convert && (cd bench/si_qoi_uq/latex/build && latexmk -pdf paper.tex)
    python -m bench.si_qoi_uq.latex.arxiv
"""
import html, re, shutil, subprocess, tarfile, tempfile, pathlib

BUILD = pathlib.Path(__file__).resolve().parent / "build"
PAGE = pathlib.Path(__file__).resolve().parents[1] / "artifact" / "paper" / "index.html"
CATEGORIES = "cond-mat.mtrl-sci (primary); cross-lists physics.comp-ph, stat.ML"


def metadata(tex, n_pages):
    """The submission form's fields: title and authors from paper.tex, the abstract as plain text from the page (arXiv
    takes no markup; the limit is 1920 characters), and the comments line."""
    title = " ".join(re.search(r"\\title\{(.+?)\}\n", tex, re.S).group(1).split())
    authors = ", ".join(re.findall(r"\\author\{([^}]+)\}", tex))
    s = PAGE.read_text()
    j = s.find('id="abstract"')
    a = s[s.find("<p>", j):s.find("</p>", j)]
    a = " ".join(html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"\\[()]", "", a))).split())
    if len(a) > 1920:
        raise SystemExit(f"abstract {len(a)} characters, over arXiv's 1920")
    nf, nt = len(re.findall(r"\\begin\{figure", tex)), len(re.findall(r"\\begin\{table", tex))
    return (f"Title: {title}\n\nAuthors: {authors}\n\nCategories: {CATEGORIES}\n\n"
            f"Comments: {n_pages} pages, {nf} figures, {nt} tables\n\nAbstract ({len(a)} characters):\n{a}\n")


def pages(pdf):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", out).group(1))


def main():
    tex = (BUILD / "paper.tex").read_text()
    figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)))
    out = BUILD / "arxiv"
    shutil.rmtree(out, ignore_errors=True)
    (out / "figures").mkdir(parents=True)
    for f in ["paper.tex", "paper.bbl", *figs]:
        shutil.copy2(BUILD / f, out / f)
    tgz = BUILD / "si-qoi-uq-arxiv.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                t.add(f, arcname=str(f.relative_to(out)))
    with tempfile.TemporaryDirectory() as d:      # compile from the tarball alone, as arXiv will
        with tarfile.open(tgz) as t:
            t.extractall(d)
        for _ in range(3):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "paper.tex"], cwd=d, capture_output=True, text=True)
        log = (pathlib.Path(d) / "paper.log").read_text(errors="replace")
        bad = [l for l in log.splitlines() if re.search(r"undefined|^! ", l)]
        n, ref = pages(pathlib.Path(d) / "paper.pdf"), pages(BUILD / "paper.pdf")
    size = tgz.stat().st_size
    print(f"{tgz}: {len(figs)} figures, {size / 1024:.0f} KiB; clean compile exit {r.returncode}, {n} pages (build {ref})")
    for l in bad:
        print("  ", l)
    if r.returncode or n != ref or bad:
        raise SystemExit("arXiv package check failed")
    (BUILD / "arxiv_metadata.txt").write_text(metadata(tex, n))
    print(f"{BUILD / 'arxiv_metadata.txt'}")


if __name__ == "__main__":
    main()
