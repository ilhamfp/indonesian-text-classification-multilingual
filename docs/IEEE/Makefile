all: clean install

install:
	mkdir -p output
	mkdir -p build
	latexmk -pdf -bibtex -outdir=build conference_101719.tex
	mv build/conference_101719.pdf output

install_pdflatex:
	mkdir -p output
	mkdir -p build
	cd src && pdflatex -output-directory=../build thesis.tex 
	mv build/thesis.pdf output

clean:
	rm -f output/* build/*
	find . -iname "*~" -exec rm '{}' ';'
