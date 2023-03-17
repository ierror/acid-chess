import io
import json
from dataclasses import dataclass
from datetime import datetime

import chess.pgn
from dateutil.parser import parse as dateutil_parser

from .gui.qt.settings import QTSettingsSyncedDataclassMixin
from .utils.timezone import datetime_local


@dataclass
class Game(QTSettingsSyncedDataclassMixin):
    program_name: str
    program_version: str
    opponent_type_idx: int = 0
    opening_book_path: str = None
    engine_idx: int = 0
    camera_idx: int = 0
    engine_time_s: int = 30
    engine_editor_text: str = None
    board_fen: str = None
    timestamp: datetime = datetime_local()
    winner: bool = None
    a1_corner: tuple = None

    # pgn attrs start with pgn_
    # see https://en.wikipedia.org/wiki/Portable_Game_Notation
    pgn_event: str = None
    pgn_site: str = None
    pgn_date: str = None
    pgn_round: int = None
    pgn_white: str = None
    pgn_black: str = None
    pgn_result: str = None

    _qt_settings_synced = [
        "opponent_type_idx",
        "opening_book_path",
        "engine_time_s",
        "engine_idx",
        "camera_idx",
        "engine_editor_text",
        "pgn_event",
        "pgn_site",
        "pgn_white",
        "pgn_black",
    ]

    _pgn = None
    _save_dir = None
    _move_stack = None
    _flush_to_disc = False

    def __init__(self, logger, **kwargs):
        self._logger = logger
        self.pgn_date = f"{datetime_local().today():%Y.%m.%d}"
        self.pgn_round = 1
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        super().__init__()

    def __setattr__(self, name, value):
        if name == "timestamp":
            if isinstance(value, str):
                value = dateutil_parser(value)
        if name == "winner":
            if value == chess.WHITE:
                self.pgn_result = "1-0"
            else:
                self.pgn_result = "0-1"

        super().__setattr__(name, value)
        if name in self._qt_settings_synced:
            self.persist()

    @property
    def pgn_path(self):
        return self._save_dir / "game.pgn"

    @property
    def pgn(self):
        return self._pgn

    @property
    def json_state_path(self):
        return self._save_dir / "state.json"

    def enable_flush_to_disc(self):
        self._flush_to_disc = True
        self.persist()

    def set_save_dir(self, save_dir):
        self._save_dir = save_dir
        return self

    def load(self):
        # load state
        with io.open(self.json_state_path) as fh:
            for key, value in json.load(fh).items():
                setattr(self, key, value)
        self._logger.append(f"{self.json_state_path} loaded")

        # load pgn
        if self.pgn_path.exists():
            self._pgn = chess.pgn.read_game(io.open(self.pgn_path))
            self._logger.append(f"{self.pgn_path} loaded")
        else:
            self._logger.append(f"{self.pgn_path} does not exist, not loaded")

    def update(self, move_stack=None, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        if move_stack is not None:
            self._move_stack = move_stack
        self.persist()

    def persist(self):
        if self._save_dir and self._flush_to_disc:
            self._save_dir.mkdir(parents=True, exist_ok=True)

            # persist state
            with io.open(self.json_state_path, "w") as fh:
                data = self.data
                data["timestamp"] = str(self.timestamp)
                json.dump(data, fh, indent=4)

            # persist pgn
            self._pgn = chess.pgn.Game()
            self.pgn.headers["ProgramName"] = self.program_name
            self.pgn.headers["ProgramVersion"] = self.program_version

            for key, value in self.__dict__.items():
                if key.startswith("pgn_") and value is not None:
                    key = key.replace("pgn_", "").title()
                    self.pgn.headers[key] = str(value)

            if self._move_stack:
                self.pgn.add_line(self._move_stack)

            print(self.pgn, file=io.open(self.pgn_path, "w"), end="\n\n")

        super().persist()
