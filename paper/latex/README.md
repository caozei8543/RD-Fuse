# Editable paper reconstruction

`manuscript.tex` is an editable LaTeX **reconstruction based on the uploaded manuscript PDF**. It was not obtained from the original Word or LaTeX project that generated `../Manuscript.pdf`. It contains the research sections, numbered equations, typeset tables, embedded paper figures, references and biographies. Recompilation may change page breaks, fonts and placement. Use the original author-side Word/LaTeX source whenever the journal requests source matching the final PDF.

A XeLaTeX build can be run from this directory:

```bash
xelatex -interaction=nonstopmode -halt-on-error manuscript.tex
xelatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

No regenerated preview PDF is bundled, to avoid confusion with the original PDF. The manuscript and cover letter are in `paper/` and are preserved byte-for-byte.
