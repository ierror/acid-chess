from abc import ABCMeta, abstractmethod

from PySide6.QtCore import QSettings

from ... import conf


class QTSettingsSyncedDataclassMixin(metaclass=ABCMeta):
    _qt_settings = QSettings(conf.PROGRAM_ID, conf.PROGRAM_ID)

    @property
    @abstractmethod
    def _qt_settings_synced(self):
        pass

    def __init__(self):
        # read data from qt settings
        for attr in self._qt_settings_synced:
            try:
                value = self._qt_settings.value(attr)
                if value is not None:
                    setattr(self, attr, value)
            except EOFError:
                pass
        super().__init__()

    @property
    def data(self):
        data = self.__dict__
        data = {key: data[key] for key in data if not key.startswith("_") and not key.startswith("qtr_instances")}
        return data

    def persist(self):
        # update qt settings
        for key, value in self.data.items():
            if key in self._qt_settings_synced:
                if type(value) not in (int, float, complex, str, bool, list, tuple, set, type(None)):
                    value = str(value)
                self._qt_settings.setValue(key, value)
