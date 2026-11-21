.PHONY: test smoke check source-compile release

test:
	python -m pytest -q

smoke:
	bash scripts/run_smoke.sh

check:
	python -m scripts.audit_paper_source --out ''
	python -m scripts.validate_latency_profile configs/demo_profile.json

source-compile:
	cd paper/latex && xelatex -interaction=nonstopmode -halt-on-error manuscript.tex && xelatex -interaction=nonstopmode -halt-on-error manuscript.tex

release:
	python -m scripts.make_release --out ../RD-Fuse_IEEE_Access_Clean.zip
