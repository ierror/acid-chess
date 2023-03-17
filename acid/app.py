import os

# Workaround for bus error on M2
# https://stackoverflow.com/questions/73072612/why-does-np-linalg-solve-raise-bus-error-when-running-on-its-own-thread-mac-m1/75317069#75317069
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import signal
import sys
from multiprocessing import freeze_support

import qdarktheme
from PySide6.QtGui import QIcon
from qdarktheme.qtpy.QtWidgets import QApplication

from acid.gui.main_window import MainWindow


def start():
    freeze_support()

    qdarktheme.enable_hi_dpi()
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(":/icon.png"))
    qdarktheme.setup_theme("dark", corner_shape="sharp")

    window = MainWindow()
    app.aboutToQuit.connect(window.close)
    signal.signal(signal.SIGINT, window.close)
    sys.exit(app.exec())
