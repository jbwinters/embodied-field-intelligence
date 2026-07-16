.PHONY: all clean

all: efi_paper.pdf

efi_paper.pdf: efi_paper.tex
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf efi_paper.tex; \
	else \
		pdflatex -interaction=nonstopmode efi_paper.tex && \
		pdflatex -interaction=nonstopmode efi_paper.tex; \
	fi

clean:
	rm -f efi_paper.aux efi_paper.log efi_paper.out efi_paper.toc \
	      efi_paper.fls efi_paper.fdb_latexmk efi_paper.bbl efi_paper.blg
