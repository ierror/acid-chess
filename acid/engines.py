import re
from dataclasses import dataclass
from functools import cached_property
from multiprocessing import cpu_count
from pathlib import Path
from typing import Iterable

import chess.engine

from . import conf


@dataclass
class Engine:
    name: str
    binary: str
    help: str
    weights_file_path: Path = None
    options: Iterable[str] = ()

    option_parse_re = re.compile("^setoption name (?P<option_name>.*) value (?P<option_value>.*)$")

    @property
    def banner(self):
        return "\n".join(
            [
                f"# This is a comment. Lines starting with # are ignored.",
                f"# Make sure the '{self.binary}' binary is installed and can be found in PATH.",
                f"# You can make changes here to configure {self.name}",
                f"# {self.help}",
            ]
        )

    @property
    def editor_text(self):
        options = "\n".join(self.options)
        return f"{self.banner}\n\n{options}"

    @cached_property
    def uci(self):
        return chess.engine.SimpleEngine.popen_uci(self.binary)

    def play(self, board, engine_time_s):
        return self.uci.play(board, chess.engine.Limit(time=engine_time_s))

    def quit(self):
        try:
            self.uci.quit()
        except chess.engine.EngineTerminatedError:
            pass


engines = [
    Engine(
        "stockfish",
        "stockfish",
        help="See https://github.com/official-stockfish/Stockfish#the-uci-protocol-and-available-options for UCI options available in Stockfish",
        options=[
            f"# The number of CPU threads used for searching a position.",
            f"# For best performance, set this equal to the number of CPU cores available.",
            f"setoption name Threads value {cpu_count() - 2}",
            f"",
            f"# The size of the hash table in MB. It is recommended to set Hash after setting Threads.",
            f"setoption name Hash value 256",
            f"",
            f"# Lower the Skill Level in order to make Stockfish play weaker (see also UCI_LimitStrength)."
            f"# Internally, MultiPV is enabled, and with a certain probability depending on the Skill Level a weaker move will be played.",
            f"setoption name Skill Level value 20",
        ],
    ),
    Engine(
        "Maia 1100",
        "lc0",
        help="See https://lczero.org/play/configuration/flags/ for UCI options available in lc0",
        weights_file_path=conf.ENGINES_DIR / "maia" / "maia-1100.pb",
        options=[
            f"# The number of CPU threads used for searching a position.",
            f"# For best performance, set this equal to the number of CPU cores available.",
            f"setoption name Threads value {cpu_count() - 2}",
        ],
    ),
    Engine(
        "Maia 1500",
        "lc0",
        help="See https://lczero.org/play/configuration/flags/ for UCI options available in lc0",
        weights_file_path=conf.ENGINES_DIR / "maia" / "maia-1500.pb",
        options=[
            f"# The number of CPU threads used for searching a position.",
            f"# For best performance, set this equal to the number of CPU cores available.",
            f"setoption name Threads value {cpu_count() - 2}",
        ],
    ),
    Engine(
        "Maia 1900",
        "lc0",
        help="See https://lczero.org/play/configuration/flags/ for UCI options available in lc0",
        weights_file_path=conf.ENGINES_DIR / "maia" / "maia-1900.pb",
        options=[
            f"# The number of CPU threads used for searching a position.",
            f"# For best performance, set this equal to the number of CPU cores available.",
            f"setoption name Threads value {cpu_count() - 2}",
        ],
    ),
]
