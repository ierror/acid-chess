from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from typing import Iterable, Union

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QPlainTextEdit, QPushButton, QSlider, QSpinBox, QWidget

from .widgets import ButtonOpensFileDialog, QPlainTextEditFocusSignaled


def _patched__setattr__(self, name, value):
    self._ReactiveBase__setattr__orig(name, value)
    for qtr_instance in self.qtr_instances.get(name, []):
        qtr_instance.attr_changed()


def _connect_ui_elm(ui_elm, callback):
    # listen on ui element for data flow to instance
    if isinstance(ui_elm, QCheckBox):
        ui_elm.stateChanged.connect(callback)
    elif isinstance(ui_elm, QLineEdit):
        ui_elm.editingFinished.connect(callback)
    elif isinstance(ui_elm, QComboBox):
        ui_elm.currentIndexChanged.connect(callback)
    elif isinstance(ui_elm, QSlider):
        ui_elm.valueChanged.connect(callback)
    elif isinstance(ui_elm, QSpinBox):
        ui_elm.valueChanged.connect(callback)
    elif isinstance(ui_elm, ButtonOpensFileDialog):
        pass
    elif isinstance(ui_elm, QPlainTextEditFocusSignaled):
        ui_elm.editingFinished.connect(callback)
    elif isinstance(ui_elm, QPushButton):
        ui_elm.clicked.connect(callback)
    else:
        raise NotImplementedError(type(ui_elm))


@dataclass
class ReactiveBase(metaclass=ABCMeta):
    obj: object
    attr: str
    ui_elm: QWidget

    def __post_init__(self):
        if not hasattr(self.obj, "qtr_instances"):
            # patch __setattr__
            type(self.obj).__setattr__orig = copy(type(self.obj).__setattr__)
            type(self.obj).__setattr__ = _patched__setattr__
            self.obj.qtr_instances = defaultdict(list)
        self.obj.qtr_instances[self.attr].append(self)

        # propagate initial value
        self.attr_changed()

    @abstractmethod
    def ui_elm_changed(self):
        pass

    @abstractmethod
    def attr_changed(self):
        pass


@dataclass
class ReactiveAttrSynced(ReactiveBase):
    def __post_init__(self):
        _connect_ui_elm(self.ui_elm, self.ui_elm_changed)
        super().__post_init__()

    @Slot()
    def ui_elm_changed(self):
        if isinstance(self.ui_elm, QCheckBox):
            value = self.ui_elm.isChecked()
        elif isinstance(self.ui_elm, QLineEdit):
            value = self.ui_elm.text()
        elif isinstance(self.ui_elm, QComboBox):
            value = self.ui_elm.currentIndex()
        elif isinstance(self.ui_elm, QSlider):
            value = self.ui_elm.value()
        elif isinstance(self.ui_elm, QSpinBox):
            value = self.ui_elm.value()
        elif isinstance(self.ui_elm, QPlainTextEdit):
            value = self.ui_elm.toPlainText()
        elif isinstance(self.ui_elm, QPushButton):
            value = not getattr(self.obj, self.attr)
        else:
            raise NotImplementedError(type(self.ui_elm))
        setattr(self.obj, self.attr, value)

    @Slot()
    def attr_changed(self):
        value = getattr(self.obj, self.attr)
        if isinstance(self.ui_elm, QCheckBox):
            if value == "true":
                value = True
            elif value == "false":
                value = False
            self.ui_elm.setChecked(value)
        elif isinstance(self.ui_elm, QLineEdit):
            self.ui_elm.setText(value)
        elif isinstance(self.ui_elm, QComboBox):
            if value is not None:
                self.ui_elm.setCurrentIndex(value)
        elif isinstance(self.ui_elm, QSlider):
            self.ui_elm.setValue(value)
        elif isinstance(self.ui_elm, QSpinBox):
            self.ui_elm.setValue(value)
        elif isinstance(self.ui_elm, ButtonOpensFileDialog):
            self.ui_elm.set_value(value)
        elif isinstance(self.ui_elm, QPlainTextEdit):
            self.ui_elm.setPlainText(value)
        elif isinstance(self.ui_elm, QPushButton):
            pass
        else:
            raise NotImplementedError(type(self.ui_elm))


@dataclass
class ReactiveAttrPresence(ReactiveBase):
    visible_for: Union[Iterable, bool]

    @Slot()
    def ui_elm_changed(self):
        pass

    @Slot()
    def attr_changed(self):
        value = getattr(self.obj, self.attr)
        if isinstance(self.visible_for, Iterable):
            self.ui_elm.setVisible(value in self.visible_for)
        elif isinstance(self.visible_for, bool):
            self.ui_elm.setVisible(bool(value) == self.visible_for)
        else:
            raise NotImplementedError(type(self.ui_elm))


@dataclass
class ReactiveAttrEnableDisable(ReactiveBase):
    enable_for: Union[Iterable, bool]

    @Slot()
    def ui_elm_changed(self):
        pass

    @Slot()
    def attr_changed(self):
        value = getattr(self.obj, self.attr)
        if isinstance(self.enable_for, Iterable):
            self.ui_elm.setEnabled(value in self.enable_for)
        elif isinstance(self.enable_for, bool):
            self.ui_elm.setEnabled(bool(value) == self.enable_for)
        else:
            raise NotImplementedError(type(self.ui_elm))


@dataclass
class ReactiveAttrToolTip(ReactiveBase):
    @Slot()
    def ui_elm_changed(self):
        pass

    @Slot()
    def attr_changed(self):
        value = getattr(self.obj, self.attr)
        self.ui_elm.setToolTip(str(value))


@dataclass
class ReactiveAttrIcon(ReactiveBase):
    values_to_icon: dict

    @Slot()
    def ui_elm_changed(self):
        pass

    @Slot()
    def attr_changed(self):
        for values, icon in self.values_to_icon.items():
            value = getattr(self.obj, self.attr)
            if value in values:
                self.ui_elm.setIcon(icon)
