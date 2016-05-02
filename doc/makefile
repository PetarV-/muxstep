muxstep-suppl.pdf : muxstep-suppl.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make $<

.PHONY : clean
clean :
	rm -f *.dvi *.log *.bak *.aux *.bbl *.blg *.idx *.ps *.eps *.pdf *.toc *.out *~ &> /dev/null
