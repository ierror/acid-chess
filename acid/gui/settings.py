from dataclasses import dataclass
from pathlib import Path

import chess

from acid.gui.qt.settings import QTSettingsSyncedDataclassMixin


@dataclass
class Settings(QTSettingsSyncedDataclassMixin):
    save_games_dir: Path = None
    collect_training_data: bool = True
    collect_training_data_dir: Path = None
    collect_training_data_threshold_perc: int = 99
    visual_debug_delay: bool = False
    sound_muted: bool = False
    lichess_access_token: str = None
    lichess_access_token_expires: int = None
    lichess_color_idx: int = 0
    lichess_time_m: int = 10
    lichess_increment_s: int = 5
    lichess_is_rated: bool = True

    _qt_settings_synced = [
        "save_games_dir",
        "collect_training_data",
        "collect_training_data_threshold_perc",
        "collect_training_data_dir",
        "sound_muted",
        "lichess_access_token",
        "lichess_access_token_expires",
        "lichess_color_idx",
        "lichess_time_m",
        "lichess_increment_s",
        "lichess_is_rated",
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
