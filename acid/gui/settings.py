from dataclasses import dataclass
from pathlib import Path

from acid.gui.qt.settings import QTSettingsSyncedDataclassMixin


@dataclass
class Settings(QTSettingsSyncedDataclassMixin):
    save_games_dir: Path = None
    collect_training_data: bool = True
    collect_training_data_dir: Path = None
    collect_training_data_threshold_perc: int = 99
    visual_debug_delay: bool = False
    sound_muted: bool = False

    _qt_settings_synced = [
        "save_games_dir",
        "collect_training_data",
        "collect_training_data_threshold_perc",
        "collect_training_data_dir",
        "sound_muted",
    ]

    def __init__(self):
        super().__init__()

    def __setattr__(self, name, value):
        if name == "save_games_dir":
            value = Path(value)
        elif name == "collect_training_data_dir":
            value = Path(value)
        super().__setattr__(name, value)
        self.persist()
