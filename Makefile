.PHONY: icons docs

icons:
	cd acid/gui/res/; pyside6-rcc icons.qrc -o icons.py

docs:
	pip install -r docs/requirements.txt
	cd docs && make clean && make html
	cp docs/index.md README.md
	perl -0777pi -e 's/^```{toctree}\s?(.*?\n)*```//gm' README.md
	perl -pi -e "s/]\(_static\//(docs\/_static\//g" README.md
	sed -i 's/<img src="_static\//<img src="docs\/_static\//g' README.md
