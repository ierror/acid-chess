.PHONY: icons docs

icons:
	cd acid/gui/res/; pyside6-rcc icons.qrc -o icons.py

docs:
	cd docs; make html
	cp docs/index.rst README.rst
	sed -i "s/.. image:: _static\//.. image:: docs\/_static\//g" README.rst
	sed -i 's/<img src="_static\//<img src="docs\/_static\//g' README.rst