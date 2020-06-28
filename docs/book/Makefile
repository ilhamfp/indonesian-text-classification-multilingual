all: clean install

install:
	mkdir -p output
	mkdir -p build
	cd src && latexmk -pdf -bibtex -outdir=../build thesis.tex
	mv build/thesis.pdf output/ta.pdf

clean:
	rm -f output/* build/*
	find . -iname "*~" -exec rm '{}' ';'
