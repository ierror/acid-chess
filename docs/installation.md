Installation
============

## Prerequisites

- [python](https://www.python.org/doc/)* >= 3.9
- [libcairo](https://cairographics.org/download/)
- modern hardware, preferably NVIDIA GPU or Mac M`[0-9]+` is recommended!

\* For people who have little or no experience with Python I recommend to use [Anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html#regular-installation) instead of the official python interpreter.

For conda users
```bash
conda install cairo
```

For macOS users
```bash
brew install cairo
```

For Debian / Ubuntu users
```bash
apt install libcairo
```

# Known bugs and limitations
- after switching cameras you will see an "Image capture failed: timed out waiting for a preview frame" error in the logs. Workaroud: Select camara you want to use and restart the app
- Linux and Windows aarch64 are not supported at the moment. [PySide6 needs support first]([https://bugreports.qt.io/browse/PYSIDE-1595])

## Installation via PyPI

```!!! ```

Installation via PyPI is not yet possible!

[Waiting for PyPI to increase the file upload limit](https://github.com/pypi/support/issues/2692).

Use Installation via git for now

```!!!```

For users who only want to use ACID Chess rather than develop for it.

see [Prerequisites](#prerequisites)

### Additional prerequisites

- [pip](https://pip.pypa.io/en/stable/installation/)


### Install the acid-chess package

```bash
pip3 install acid-chess  # or pip install acid-chess  
```

### Run acid-chess

```bash
acid-chess
```

## Installation via git

For users who want to develop for ACID Chess.

see [Prerequisites](#prerequisites)

### Additional prerequisites

- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [git-lfs](https://git-lfs.com)
- [pipenv](https://pipenv.pypa.io/en/latest/index.html)
- [pre-commit](https://pre-commit.com/#install) (if you want to commit)

### Clone the repo

```bash
git clone git@github.com:ierror/acid-chess.git
cd acid-chess
```

### git-lfs Setup

```bash
git lfs install
git lfs pull
```

### Install python requirements

```bash
pipenv install --dev
```

### Run acid-chess

```bash
pipenv run bin/acid-chess
```

### pre-commit

If you want to commit, we recommend to install the hooks pre-commit hooks.

```bash
pre-commit install
```