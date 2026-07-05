.PHONY: all clean

all: efi_paper.pdf

efi_paper.pdf: efi_paper.tex
	latexmk -pdf efi_paper.tex

clean:
	latexmk -C efi_paper.tex
