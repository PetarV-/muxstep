PROJECT=muxstep-suppl
TEX=pdflatex
BUILDTEX=$(TEX) $(PROJECT).tex

.PHONY : all
all :
	$(BUILDTEX) &> /dev/null
	$(BUILDTEX) &> /dev/null
	$(BUILDTEX) &> /dev/null

.PHONY : clean
clean :
	rm -f *.dvi *.log *.bak *.aux *.bbl *.blg *.idx *.ps *.eps *.pdf *.toc *.out *~ &> /dev/null
