all:
	pdflatex main
	pdflatex main
	bibtex -min-crossrefs=1000 main
	pdflatex main
	pdflatex main

clean:
	rm -f *.aux *.bbl *.blg *.log eval/*.log *.dvi *.backup *~ *.out
