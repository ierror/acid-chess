from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QFile, Signal
from PySide6.QtWidgets import QFileDialog, QPlainTextEdit


class ButtonOpensFileDialog:
    value_raw = None
    value_display = None

    def __init__(
        self,
        parent,
        button_elm,
        dialog_method,
        dialog_kwargs,
        target_obj,
        target_attr,
        display_filter=None,
        default_text="choose",
    ):
        self.parent = parent
        self.button_elm = button_elm
        self.dialog_method = dialog_method
        self.display_filter = display_filter
        self.dialog_kwargs = dialog_kwargs
        self.target_obj = target_obj
        self.target_attr = target_attr
        self.default_text = default_text
        button_elm.clicked.connect(self.open_dialog)

    def set_value(self, value):
        if self.display_filter and value is not None:
            self.value_display = self.display_filter(value)
        else:
            self.value_display = value
        self.button_elm.setText(self.value_display or self.default_text)

    def open_dialog(self):
        (path_selected, _) = getattr(QFileDialog, self.dialog_method)(self.parent, **self.dialog_kwargs)
        if path_selected:
            self.value_raw = path_selected
            self.set_value(self.value_raw)
            setattr(self.target_obj, self.target_attr, self.value_raw)


class QPlainTextEditFocusSignaled(QPlainTextEdit):
    editingFinished = Signal()
    receivedFocus = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self._changed = False
        self.setTabChangesFocus(True)
        self.textChanged.connect(self._handle_text_changed)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.receivedFocus.emit()

    def focusOutEvent(self, event):
        if self._changed:
            self.editingFinished.emit()
        super().focusOutEvent(event)

    def _handle_text_changed(self):
        self._changed = True


class PictureLabelFitToParent(QtWidgets.QLabel):
    def __init__(self, text=None, pixmap=None):
        super().__init__()
        self._pixmap = None
        text is not None and self.setText(text)
        pixmap is not None and self.setPixmap(pixmap)

    def setPixmap(self, pixmap: QtGui.QPixmap):
        self._pixmap = pixmap
        self.repaint()

    def paintEvent(self, event: QtGui.QPaintEvent):
        super().paintEvent(event)
        if self._pixmap is not None:
            image_width, image_height = self._pixmap.width(), self._pixmap.height()
            label_width, label_height = self.width(), self.height()
            ratio = min(label_width / image_width, label_height / image_height)
            new_width, new_height = int(image_width * ratio), int(image_height * ratio)
            new_pixmap = self._pixmap.scaledToWidth(new_width, QtCore.Qt.TransformationMode.SmoothTransformation)
            x, y = abs(new_width - label_width) // 2, abs(new_height - label_height) // 2
            QtGui.QPainter(self).drawPixmap(x, y, new_pixmap)
