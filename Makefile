.PHONY: icons docs dist

icons:
	cd acid/gui/res/; pyside6-rcc icons.qrc -o icons.py

docs:
	pip install -r docs/requirements.txt
	cd docs && make clean && make html
	cp docs/index.md README.md
	perl -0777pi -e 's/^```{toctree}\s?(.*?\n)*```//gm' README.md
	perl -pi -e "s/]\(_static\//(docs\/_static\//g" README.md
	sed -i 's/<img src="_static\//<img src="docs\/_static\//g' README.md

dist:
	pip install --upgrade twine build pipreqs
	rm -rf dist/ acid_chess-egg-info/
	mkdir -p acid_chess-egg-info/
	pipreqs ./ --force --savepath /tmp/requirements.txt
	cat /tmp/requirements.txt
	read -p "Copy requirements to pyproject.toml. Press Enter to afterwards to continue" </dev/tty
	python -m build
	rm -f /tmp/requirements.txt
	#python -m twine upload --repository acid-chess dist/*
