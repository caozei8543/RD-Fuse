"""Mechanical parity checks for user-PDF and editable LaTeX reconstruction.

PASS means numeric, structural, figure and reference coverage checks passed;
it does NOT assert pixel-identical layout or the existence of original author DOCX.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SUBMITTED_PDF_SHA256 = 'a0d356ec5659cd0c6b5c12362e161284f5402470fc41675f38bf377d33b3dce3'
EXPECTED_COVER_LETTER_SHA256 = '45c218f72ad1e72298995e050a903fa4486f2dbcf5acf249c82543647ab94cb1'
NUMERIC_STRINGS = ['90.1', '62.7', '94.8', '62.2', '19.8', '17.6',
                   '41.6', '50.3', '23.9', '56.8', '14.8', '37.2',
                   '3024', '336', '840', '12025', '3463', '10822', '1203',
                   '3407', '2026']


def audit(root: Path = ROOT) -> dict:
    base = root/'paper'
    tex = base/'latex'/'manuscript.tex'
    pdf = base/'Manuscript.pdf'
    letter = base/'Cover_Letter.pdf'
    errors=[]
    data = tex.read_text(encoding='utf8') if tex.is_file() else ''
    try:
        import fitz
        p = fitz.open(pdf)
        pdftext = ''.join(page.get_text() for page in p)
        pages = len(p)
    except Exception as exc:
        pdftext = '';pages=0;errors.append('PDF unreadable: '+repr(exc))
    pdfnums = {s: bool(re.search(r'(?<![0-9])'+re.escape(s)+r'(?![0-9])', pdftext.replace(',',''))) for s in NUMERIC_STRINGS}
    texnums = {s: bool(re.search(r'(?<![0-9])'+re.escape(s)+r'(?![0-9])', data.replace(',',''))) for s in NUMERIC_STRINGS}
    for s in NUMERIC_STRINGS:
        if pdfnums[s] and not texnums[s]:
            errors.append(f'PDF declared numeric token {s} absent from TeX')
    tags=sorted(set(int(n) for n in re.findall(r'\\tag\{([0-9]+)\}', data)))
    if tags!=list(range(1,13)):
        errors.append(f'Expected equations 1-12, got {tags}')
    fig_present=all((base/'latex'/'figures'/f'fig{i}.jpeg').is_file() for i in range(1,9))
    if not fig_present: errors.append('One or more figures 1-8 missing')
    figs=sorted(set(int(n) for n in re.findall(r'\\label\{fig:([0-9]+)\}',data)))
    if figs!=list(range(1,9)): errors.append(f'Expected 8 figures referenced, got {figs}')
    refs=sorted(set(int(n) for n in re.findall(r'\\bibitem\{ref([0-9]+)\}',data)))
    if refs!=list(range(1,35)): errors.append('34 reference entries not present')
    tables=len(re.findall(r'\\begin\{table\*?\}',data))
    if tables!=4: errors.append(f'Expected four manuscript tables, got {tables}')
    if pages!=10: errors.append(f'Supplied final PDF expected 10 pages, got {pages}')
    if not letter.is_file(): errors.append('Submitted cover letter PDF missing')
    if pdf.is_file() and hashlib.sha256(pdf.read_bytes()).hexdigest()!=EXPECTED_SUBMITTED_PDF_SHA256:
        errors.append('Submitted PDF differs from the final uploaded 10-page PDF')
    if letter.is_file() and hashlib.sha256(letter.read_bytes()).hexdigest()!=EXPECTED_COVER_LETTER_SHA256:
        errors.append('Cover letter differs from the final uploaded PDF')
    for name in ('Baojian Ren','Tao Cao','Zhengyang Zhang','Jing Wu','Xinglin Zhang','Shuchen Bai'):
        if name not in data: errors.append(f'Author name absent: {name}')
    return {'passed':not errors,'errors':errors,
            'pdf_pages':pages,            'submitted_pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf.is_file() else None,
            'cover_letter_pdf_sha256':hashlib.sha256(letter.read_bytes()).hexdigest() if letter.is_file() else None,
            'equations':tags,'figures':figs,'references':len(refs),'tables':tables,
            'numeric_token_coverage':{'pdf':pdfnums,'latex_source':texnums},
            'not_verified':['original author source availability','pixel-identical typesetting',
                            'independent reproduction of scientific measurements']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=None)
    args=p.parse_args()
    out=audit()
    if args.out:
        path=ROOT/args.out;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps(out,indent=2))
    return 0 if out['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
