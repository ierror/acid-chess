from pathlib import Path

import acid

PROJECT_ROOT = (Path(__file__).parent / ".." / "..").parent.resolve()

# program name and version
PROGRAM_ID = "acid-chess"
PROGRAM_NAME = "ACID Chess"
PROGRAM_VERSION = acid.__version__
PROGRAM_SITE = "https://github.com/ierror/acid-chess"
BOT_NAME = "ACID"

# path
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DATA_DIR = DATA_DIR / "training"

NOTEBOOKS_TMP_DIR = PROJECT_ROOT / "notebooks" / "tmp"
NOTEBOOKS_LOGS_DIR = NOTEBOOKS_TMP_DIR / "logs"
NOTEBOOKS_CHECKPOINTS_DIR = NOTEBOOKS_TMP_DIR / "checkpoints"

GUI_DIR = PROJECT_ROOT / "acid" / "gui"
GUI_RES_DIR = GUI_DIR / "res"

ENGINES_DIR = DATA_DIR / "engines"
