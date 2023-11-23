Installation
============

## Prerequisites

- [python](https://www.python.org/doc/)* >= 3.9

For people who have little or no experience with Python I recommend to use [Anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html#regular-installation) instead of the official python interpreter.

- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [git-lfs](https://git-lfs.com)
- [pipenv](https://pipenv.pypa.io/en/latest/index.html)
- [pre-commit](https://pre-commit.com/#install) (if you want to commit)
- modern hardware, preferably NVIDIA GPU or Mac M`[0-9]+` is recommended!
- [libcairo](https://cairographics.org/download/)

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
pipenv install
```

or if you want to develop for ACID CHESS

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

# Known bugs and limitations
- after switching cameras you will see an "Image capture failed: timed out waiting for a preview frame" error in the logs. Workaroud: Select camara you want to use and restart the app
- Linux and Windows aarch64 are not supported at the moment. [PySide6 needs support first]([https://bugreports.qt.io/browse/PYSIDE-1595])
