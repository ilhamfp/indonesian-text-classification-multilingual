all: clean install

install:
	mkdir -p output
	mkdir -p build
	cd src && latexmk -pdf -bibtex -outdir=../build thesis.tex
	mv build/thesis.pdf output/ta_2.pdf

install_pdflatex:
	mkdir -p output
	mkdir -p build
	cd src && pdflatex -output-directory=../build thesis.tex 
	mv build/thesis.pdf output

clean:
	rm -f output/* build/*
	find . -iname "*~" -exec rm '{}' ';'
