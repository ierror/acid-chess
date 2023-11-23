import tempfile
from copy import copy
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from io import BytesIO
from multiprocessing import Lock
from pathlib import Path
from random import choice
from time import sleep, time
from uuid import uuid4

import berserk
import chess.engine
import chess.pgn
import chess.polyglot
import chess.svg
import cv2
import qimage2ndarray
from cairosvg import svg2png
from gtts import gTTS
from imutils.perspective import four_point_transform
from PIL import Image
from PIL.ImageQt import ImageQt
from playsound import playsound
from PySide6.QtCore import QFile, QStandardPaths, QThreadPool, QTimer, Slot
from PySide6.QtGui import QFont, QFontDatabase, QIcon, QImage, QKeySequence, QPixmap, QShortcut, Qt
from PySide6.QtMultimedia import QCamera, QImageCapture, QMediaCaptureSession, QMediaDevices
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QVBoxLayout

from acid import conf
from acid.board import Board, Detector, Square
from acid.engines import Engine, engines
from acid.game import Game
from acid.gui.logs import Logger
from acid.gui.online.lichess import AuthBrowserWindow
from acid.gui.qt.reactive import (
    ReactiveAttrEnableDisable,
    ReactiveAttrIcon,
    ReactiveAttrPresence,
    ReactiveAttrSynced,
    ReactiveAttrToolTip,
)
from acid.gui.qt.widgets import ButtonOpensFileDialog, PictureLabelFitToParent, QPlainTextEditFocusSignaled
from acid.gui.settings import Settings
from acid.gui.state import BoardDetectorState, GameMode, GameState
from acid.gui.threads import Worker

from acid.gui.res import icons  # isort:skip

UI_MAPPINGS = {
    "opponent": {
        "computer_black": "Computer - Black",
        "computer_white": "Computer - White",
        "human": "Player over the board",
        "lichess": "Player on Lichess",
    },
    "engine": {engine_idx: engine.name for (engine_idx, engine) in enumerate(engines)},
    "colors": {
        chess.BLACK: "Black",
        chess.WHITE: "White",
    },
}

TIMER_DEFAULT_MS = 50


@dataclass
class Feedback:
    window: QMainWindow
    message: str
    show_message: bool = True
    log_message: bool = True
    speak_message_overwrite: str = None
    speak_message: bool = False

    def __post_init__(self):
        self.window.feedback_last = self
        self.window.provide_feedback(self)


class MainWindow(QMainWindow):
    ui = None
    camera_capture = None
    debug_images_buffer = []
    rendered_image = None
    frame_num = 0
    detector_result = None
    close_requested = False
    current_frame = None
    feedback_last = None
    opening_book_reader = None
    game_state = GameState.NULL
    board_detector_state = BoardDetectorState.NULL
    labels = {}
    pause_text = None

    _capture_session = None
    _game_save_dir = None
    _engine = None
    _camera = None
    _camera_switch_mutex = Lock()

    board_detector = Detector(debug_images_buffer)
    settings = Settings()
    logger = Logger()
    board = Board()
    game = Game(
        logger=logger, program_name=conf.PROGRAM_NAME, program_version=conf.PROGRAM_VERSION, pgn_site=conf.PROGRAM_SITE
    )

    white_time = None
    black_time = None
    last_tick_s = None

    lichess_game = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_ui()
        self.ui.setFocus()
        self.ui.setWindowIcon(QIcon(":/icon.png"))
        self.ui.setWindowTitle(f"{conf.PROGRAM_NAME} :: {conf.PROGRAM_VERSION}")
        self.log("System initialized 🥳")

        # replace board_debug widget by resizeable picture label
        self.board_debug = PictureLabelFitToParent()
        self.ui.boardDebug.setContentsMargins(0, 0, 0, 0)
        lay = QVBoxLayout(self.ui.boardDebug)
        lay.setContentsMargins(0, 0, 0, 0)
        self.board_debug.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.board_debug)

        # replace board_rendered by resizeable picture label
        self.board_rendered = PictureLabelFitToParent()
        self.ui.boardRendered.setContentsMargins(0, 0, 0, 0)
        lay = QVBoxLayout(self.ui.boardRendered)
        lay.setContentsMargins(0, 0, 0, 0)
        self.board_rendered.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.board_rendered)

        # wire tap things
        self.ui.comboBoxOpponent.removeItem(0)
        self.ui.comboBoxOpponent.insertItems(0, UI_MAPPINGS["opponent"].values())
        ReactiveAttrSynced(self.ui.comboBoxOpponent, self.game, "opponent_type_idx")

        self.ui.comboBoxEngine.removeItem(0)
        self.ui.comboBoxEngine.insertItems(0, UI_MAPPINGS["engine"].values())
        ReactiveAttrSynced(self.ui.comboBoxEngine, self.game, "engine_idx")

        self.ui.comboBoxCamera.removeItem(0)
        self.ui.comboBoxCamera.insertItems(0, [c.description() for c in self.cameras])
        ReactiveAttrSynced(self.ui.comboBoxCamera, self.game, "camera_idx")

        # opening book
        documents_dir = Path(QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation))
        dialog_kwargs = {
            "caption": "Select opening book file (polyglot format)",
            "dir": str(documents_dir),
            "filter": "*",
        }
        opening_book = ButtonOpensFileDialog(
            self,
            self.ui.pushButtonOpeningBook,
            "getOpenFileName",
            dialog_kwargs,
            self.game,
            "opening_book_path",
            lambda text: Path(text).name,
        )
        ReactiveAttrPresence(self.ui.pushButtonOpeningBookRemove, [[self.game, "opening_book_path", True]])

        # reactive bind ui elements to game state items
        ReactiveAttrSynced(self.ui.lineEditPlayerWhite, self.game, "pgn_white")
        ReactiveAttrSynced(self.ui.lineEditPlayerBlack, self.game, "pgn_black")
        ReactiveAttrSynced(self.ui.lineEditEventName, self.game, "pgn_event")
        ReactiveAttrSynced(self.ui.spinBoxEngineTime, self.game, "engine_time_s")
        ReactiveAttrSynced(opening_book, self.game, "opening_book_path")

        ReactiveAttrSynced(self.ui.plainTextEditEngineOptions, self.game, "engine_editor_text")

        ReactiveAttrToolTip(self.ui.pushButtonOpeningBook, self.game, "opening_book_path")

        # reactive bind ui elements to gui settings
        ReactiveAttrSynced(self.ui.checkBoxCollectTrainingData, self.settings, "collect_training_data")
        ReactiveAttrSynced(
            self.ui.spinBoxCollectTrainingDataThreshold, self.settings, "collect_training_data_threshold_perc"
        )
        ReactiveAttrSynced(self.ui.checkBoxVisualDebugDelay, self.settings, "visual_debug_delay")

        ReactiveAttrSynced(self.ui.pushButtonMuteUnmute, self.settings, "sound_muted")
        values_to_icon = {
            (True,): QIcon(":/sound-off.svg"),
            (False,): QIcon(":/sound-high.svg"),
        }
        ReactiveAttrIcon(self.ui.pushButtonMuteUnmute, self.settings, "sound_muted", values_to_icon=values_to_icon)

        # disable some ui elements when game has started
        ReactiveAttrEnableDisable(
            self.ui.comboBoxOpponent, self, "game_state", enable_for=[GameState.NULL, GameState.FINISHED]
        )

        # set save games dir default path
        if self.settings.save_games_dir is None:
            game_save_location_def = documents_dir / conf.PROGRAM_NAME / "Games"
            self.settings.save_games_dir = game_save_location_def
        self.settings.save_games_dir.mkdir(parents=True, exist_ok=True)

        # set collect training data dir default path
        if self.settings.collect_training_data_dir is None:
            collect_training_data_dir_def = documents_dir / conf.PROGRAM_NAME / "TrainingData"
            self.settings.collect_training_data_dir = collect_training_data_dir_def

        # some options for computer / human play based on selected opponent
        for ui_elm in [
            self.ui.labelPlayerWhite,
            self.ui.lineEditPlayerWhite,
            self.ui.labelPlayerBlack,
            self.ui.lineEditPlayerBlack,
            self.ui.labelEventName,
            self.ui.lineEditEventName,
            self.ui.labelSaveGamesTo,
            self.ui.pushButtonSaveGamesTo,
        ]:
            ReactiveAttrPresence(ui_elm, [[self.game, "opponent_type_idx", (0, 1, 2)]])

        ReactiveAttrSynced(self.ui.labelGameplayPlayerWhite, self.game, "pgn_white")
        ReactiveAttrSynced(self.ui.labelGameplayPlayerBlack, self.game, "pgn_black")

        # show engine options
        for ui_elm in [
            self.ui.labelEngine,
            self.ui.comboBoxEngine,
            self.ui.labelEngineTime,
            self.ui.spinBoxEngineTime,
            self.ui.labelEngineConfig,
            self.ui.plainTextEditEngineOptions,
            self.ui.labelOpeningBook,
            self.ui.pushButtonOpeningBookRemove,
            self.ui.pushButtonOpeningBook,
        ]:
            ReactiveAttrPresence(ui_elm, [[self.game, "opponent_type_idx", (0, 1)]])

        ReactiveAttrToolTip(self.ui.pushButtonSaveGamesTo, self.settings, "save_games_dir")
        ReactiveAttrToolTip(self.ui.pushButtonCollectTrainingDataSaveTo, self.settings, "collect_training_data_dir")

        re_detect_visible_for = (BoardDetectorState.DETECTED,)
        ReactiveAttrPresence(self.ui.pushButtonReDetectCorners, [[self, "board_detector_state", re_detect_visible_for]])
        self.ui.pushButtonReDetectCorners.setVisible(False)

        values_to_icon = {
            (GameState.NULL, GameState.FINISHED, GameState.PAUSED): QIcon(":/play.svg"),
            (GameState.RUNNING,): QIcon(":/pause.svg"),
        }
        ReactiveAttrIcon(self.ui.pushButtonStartPause, self, "game_state", values_to_icon=values_to_icon)

        self.ui.comboBoxOpponent.currentIndexChanged.connect(self.action_opponent_changed)
        self.ui.comboBoxEngine.currentIndexChanged.connect(self.action_engine_changed)
        self.ui.comboBoxCamera.currentIndexChanged.connect(self.action_camera_changed)
        self.ui.pushButtonOpeningBook.clicked.connect(self.action_opening_book_selected)
        self.ui.pushButtonOpeningBookRemove.clicked.connect(lambda _: self.game.update(opening_book_path=None))
        self.ui.pushButtonSaveGamesTo.clicked.connect(self.action_save_games_to)
        self.ui.pushButtonCollectTrainingDataSaveTo.clicked.connect(self.action_collect_training_data_to)
        self.ui.pushButtonStartPause.clicked.connect(self.action_start_pause)
        self.ui.pushButtonMoveUndo.clicked.connect(self.action_move_undo)
        self.ui.plainTextEditEngineOptions.editingFinished.connect(self.action_engine_configured)
        QShortcut(QKeySequence(Qt.ALT | Qt.Key_Z), self.ui).activated.connect(self.action_move_undo)

        # show lichess options - auth
        self.ui.comboBoxLichessColor.removeItem(0)
        self.ui.comboBoxLichessColor.insertItems(0, UI_MAPPINGS["colors"].values())

        for ui_elm in [
            self.ui.pushButtonLichessLogin,
            self.ui.labelLichessUsernamePasswordHint,
        ]:
            ReactiveAttrPresence(
                ui_elm, [[self.game, "opponent_type_idx", 3], [self.settings, "lichess_access_token", False]]
            )

        for ui_elm in [
            self.ui.labelLichessColor,
            self.ui.comboBoxLichessColor,
            self.ui.labelLichessTime,
            self.ui.spinBoxLichessTime,
            self.ui.labelLichessIncrement,
            self.ui.spinBoxLichessIncrement,
            self.ui.labelLichessRated,
            self.ui.checkBoxLichessRated,
        ]:
            ReactiveAttrPresence(
                ui_elm, [[self.game, "opponent_type_idx", 3], [self.settings, "lichess_access_token", True]]
            )

        ReactiveAttrPresence(
            self.ui.pushButtonLichessLogout,
            [[self.game, "opponent_type_idx", 3], [self.settings, "lichess_access_token", True]],
        )

        ReactiveAttrSynced(self.ui.comboBoxLichessColor, self.settings, "lichess_color_idx")
        ReactiveAttrSynced(self.ui.spinBoxLichessTime, self.settings, "lichess_time_m")
        ReactiveAttrSynced(self.ui.spinBoxLichessIncrement, self.settings, "lichess_increment_s")
        ReactiveAttrSynced(self.ui.checkBoxLichessRated, self.settings, "lichess_is_rated")

        self.ui.pushButtonLichessLogin.clicked.connect(self.action_lichess_login)
        self.ui.comboBoxLichessColor.currentIndexChanged.connect(self.action_lichess_color_changed)
        self.ui.pushButtonLichessLogout.clicked.connect(self.action_lichess_logout)

        # monospace fonts
        mono_font = QFontDatabase.font("Menlo", "regular", 13)
        mono_font.setStyleHint(QFont.StyleHint.Monospace)
        self.ui.plainTextLogbox.setFont(mono_font)

        mono_font = QFontDatabase.font("Menlo", "regular", 13)
        mono_font.setStyleHint(QFont.StyleHint.Monospace)
        self.ui.plainTextEditEngineOptions.setFont(mono_font)

        self.game.set_save_dir(self.game_save_dir)
        self.log(f"Saving games to {self._game_save_dir.parent}")
        if self.settings.collect_training_data:
            self.log(f"Saving training data to {self.settings.collect_training_data_dir}")

        # ensure initial state
        self.action_opening_book_selected()
        self.action_camera_changed()
        self.update_board_rendering()

        # timers
        self.update_ui_timer = QTimer()
        self.update_ui_timer.timeout.connect(self.update_ui)
        self.update_ui_timer.start(TIMER_DEFAULT_MS)

        self.vis_debug_timer_detector = QTimer()
        self.vis_debug_timer_detector.timeout.connect(self.update_ui_detector)
        self.vis_debug_timer_detector.start(TIMER_DEFAULT_MS)

        self.threadpool = QThreadPool()

        # frame grabber worker
        fg_worker = Worker(self.game_loop)
        self.threadpool.start(fg_worker)

        # lichess game state streaming worker
        lichess_worker = Worker(self.lichess_game_state_worker)
        self.threadpool.start(lichess_worker)

    def load_ui(self):
        loader = QUiLoader()
        path = Path(__file__).resolve().parent / "app.ui"
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        loader.registerCustomWidget(QPlainTextEditFocusSignaled)
        self.ui = loader.load(ui_file)
        self.ui.showMaximized()
        ui_file.close()

    def log(self, msg, timestamp=None, stdout_only=False, is_error=False):
        if is_error:
            msg = f"Error: {msg}"
        self.logger.append(msg, stdout_only, timestamp)

    def show_alert(self, message):
        message = str(message)
        self.log(message, is_error=True)
        QMessageBox.information(None, "boing", message, QMessageBox.Ok)

    @property
    def engine_color(self):
        return list(UI_MAPPINGS["opponent"].keys())[self.game.opponent_type_idx]

    @property
    def lichess_color(self):
        return list(UI_MAPPINGS["colors"].keys())[self.settings.lichess_color_idx]

    @property
    def engine(self):
        if self.game.engine_idx is not None:
            return engines[self.game.engine_idx]

    @cached_property
    def cameras(self):
        # uses a cached property to ensure always the same order of cameras for this session
        return QMediaDevices.videoInputs()

    @cached_property
    def lichess_api(self):
        if not self.settings.lichess_access_token:
            raise NotImplementedError()
        return berserk.Client(berserk.TokenSession(self.settings.lichess_access_token))

    @property
    def game_save_dir(self):
        # construct game save dir e.g. /foo/bar/Games/2023-01-01_test_erei-001/
        if self._game_save_dir:
            return self._game_save_dir

        # date
        game_name = self.game.pgn_date.replace(".", "-")

        # event name
        event_name = self.game.pgn_event or ""
        event_name = event_name.replace("  ", " ").replace(" ", "_").lower()
        event_name = event_name.strip()
        if event_name:
            game_name = f"{game_name}_{event_name}"

        # find next free round
        while True:
            self._game_save_dir = self.settings.save_games_dir
            self._game_save_dir = self._game_save_dir / f"{game_name}-{self.game.pgn_round:03d}"
            if (self._game_save_dir / "state.json").exists():
                self.game.update(pgn_round=self.game.pgn_round + 1)
            else:
                break

        self.game.set_save_dir(self._game_save_dir)
        return self._game_save_dir

    @property
    def opponent(self):
        return list(UI_MAPPINGS["opponent"].keys())[self.game.opponent_type_idx]

    def validate_lichess_access_token(self):
        # token already set and still valid?
        if self.settings.lichess_access_token:
            if (
                not self.settings.lichess_access_token_expires
                or self.settings.lichess_access_token_expires < time() + 24 * 3600
            ):
                # no longer valid
                self.settings.lichess_access_token = None
                self.settings.lichess_access_token_expires = None

    @Slot()
    def action_camera_changed(self):
        available_cameras = QMediaDevices.videoInputs()
        if len(available_cameras) == 0:
            self.log("No cameras available", is_error=True)
            return
        elif self.game.camera_idx > len(available_cameras) - 1:
            self.game.camera_idx = 0

        with self._camera_switch_mutex:
            if self._camera:
                self._camera.stop()

            self._camera = QCamera(available_cameras[self.game.camera_idx])
            self._camera.errorOccurred.connect(self.on_camera_error)
            self.camera_capture = QImageCapture(self._camera)
            self.camera_capture.imageCaptured.connect(self.on_image_captured)
            self.camera_capture.errorOccurred.connect(self.on_capture_error)

            self._capture_session = QMediaCaptureSession()
            self._capture_session.setImageCapture(self.camera_capture)
            self._capture_session.setCamera(self._camera)
            self._camera.start()

    @Slot(int, QImage)
    def on_image_captured(self, id, image):
        image = copy(qimage2ndarray.rgb_view(image))
        self.current_frame = image

    @Slot(int, QImageCapture.Error, str)
    def on_capture_error(self, id, error, error_string):
        self.log(error_string, is_error=True)

    @Slot(QCamera.Error, str)
    def on_camera_error(self, error, error_string):
        self.log(error_string, is_error=True)

    @Slot()
    def action_start_pause(self):
        if self.game_state in (GameState.NULL, GameState.PAUSED, GameState.FINISHED):
            # clear, to start with fresh debug images
            self.debug_images_buffer.clear()
            self.game.enable_flush_to_disc()

            # start detection of not already detected
            if self.board_detector_state != BoardDetectorState.DETECTED:
                self.board_detector_state = BoardDetectorState.RUNNING_CORNER_DETECTION
            if self.game_state in (GameState.NULL, GameState.FINISHED):
                self.log("game started")
            else:
                if self.feedback_last:
                    self.feedback_last.show_message = True
                    self.provide_feedback(self.feedback_last)
                self.log("game resumed")
            self.game_state = GameState.RUNNING
        elif self.game_state == GameState.RUNNING:
            if self.opponent == "lichess":
                Feedback(self, "You can't pause a lichess game", False, True)
            else:
                self.game_state = GameState.PAUSED
                self.pause_text = "paused"
                self.log("game paused")

    @Slot()
    def action_move_undo(self):
        self.game_state = GameState.PAUSED
        self.feedback_last = None
        self.pause_text = "Resume the game when you and the board are ready"
        try:
            self.board.pop()
            self.update_board_rendering()
        except IndexError:
            pass

    @Slot()
    def action_opponent_changed(self):
        self.update_board_rendering()

        # set player name to COMPUTER_NAME in case of bot games
        if self.opponent == "computer_black":
            self.game.pgn_black = conf.COMPUTER_NAME
            self.game.pgn_white = None
        elif self.opponent == "computer_white":
            self.game.pgn_white = conf.COMPUTER_NAME
            self.game.pgn_black = None
        elif self.opponent == "lichess":
            self.validate_lichess_access_token()
            # "hack" to force lichess_access_token to trigger ui changes
            self.settings.lichess_access_token = self.settings.lichess_access_token
        else:
            self.game.engine_idx = None

    @Slot()
    def action_load_existing_game(self):
        # not used atm
        (json_game_path, _) = QFileDialog.getOpenFileName(
            self,
            "Select json file",
            str(self.settings.save_games_dir),
            "*.json",
        )
        if json_game_path:
            # force re-creation of "cached" property
            self._game_save_dir = None
            self.game.set_save_dir(Path(json_game_path).parent)
            self.game.load()
            self.board.reset()
            for move in self.game.pgn.mainline_moves():
                self.board.push(move)
            self.update_board_rendering()

    @Slot()
    def action_save_games_to(self):
        dirname = QFileDialog.getExistingDirectory(
            self, "Select directory where to save games", str(self.settings.save_games_dir)
        )
        if dirname:
            self.settings.save_games_dir = dirname
            # force re-creation of "cached" property
            self._game_save_dir = None

    @Slot()
    def action_collect_training_data_to(self):
        dirname = QFileDialog.getExistingDirectory(
            self, "Select directory where to save training data", str(self.settings.collect_training_data_dir)
        )
        if dirname:
            self.settings.collect_training_data_dir = dirname

    @Slot()
    def action_opening_book_selected(self):
        try:
            self.opening_book_reader = None
            if self.game.opening_book_path:
                self.opening_book_reader = chess.polyglot.open_reader(self.game.opening_book_path)
        except OSError as e:
            self.game.opening_book_path = None
            self.show_alert(str(e))

    @Slot()
    def action_engine_changed(self):
        self.game.engine_editor_text = self.engine.editor_text
        self.action_engine_configured()

    @Slot()
    def action_engine_configured(self):
        options = {}
        for line in self.game.engine_editor_text.splitlines():
            line = line.strip()
            line = line.replace("  ", " ")
            if not line or line.startswith("#"):
                continue
            option = Engine.option_parse_re.match(line)
            if option is None:
                self.show_alert(f"Wrong syntax on line '{line}' - Use 'setoption name <option> value <value>'")
                return
            else:
                options[option["option_name"]] = option["option_value"]
        if options:
            try:
                self.engine.uci.configure(options)
                self.log("Engine successfully configured")
            except chess.engine.EngineError as e:
                self.show_alert(e)

    @Slot()
    def action_lichess_login(self):
        # get new token
        def save_token(token):
            self.settings.lichess_access_token = token["access_token"]
            self.settings.lichess_access_token_expires = token["expires_at"]

        auth_browser = AuthBrowserWindow(save_token)
        auth_browser.show()

    @Slot()
    def action_lichess_logout(self):
        self.settings.lichess_access_token = None
        self.settings.lichess_access_token_expires = None

    @Slot()
    def action_lichess_color_changed(self):
        self.update_board_rendering()

    def update_board_rendering(self):
        last_move = None
        if self.board.move_stack:
            last_move = self.board.move_stack[-1]

        rendered_image_out = BytesIO()

        orientation = chess.BLACK
        if self.opponent == "lichess":
            if self.lichess_color == chess.WHITE:
                orientation = chess.WHITE
        else:
            if self.game.opponent_type_idx == chess.WHITE:
                orientation = chess.WHITE

        svg_board = chess.svg.board(self.board, lastmove=last_move, orientation=orientation)
        svg2png(bytestring=svg_board, write_to=rendered_image_out, output_width=2048, output_height=2048, dpi=200)
        self.rendered_image = Image.open(rendered_image_out)

    @Slot()
    def update_ui_detector(self):
        try:
            image = self.debug_images_buffer.pop(0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = qimage2ndarray.array2qimage(image)
            pixmap = QPixmap(image)
            self.board_debug.setPixmap(pixmap)
        except IndexError:
            pass

        self.vis_debug_timer_detector.setInterval(max(int(self.settings.visual_debug_delay) * 1000, TIMER_DEFAULT_MS))

    @Slot()
    def update_ui(self):
        # update rendered_image
        if self.rendered_image:
            image = ImageQt(self.rendered_image)
            pixmap = QPixmap(image)
            self.board_rendered.setPixmap(pixmap)

        # update logs
        if self.logger.has_fresh_data:
            self.ui.plainTextLogbox.setPlainText("\n".join(self.logger.get_entries_reversed()))
            self.logger.mark_consumed()

        # update labels
        for label, text in self.labels.items():
            getattr(self.ui, label).setText(text)

        self.update_clock_times()

    def update_clock_times(self, wtime=None, btime=None):
        if wtime:
            self.white_time = wtime
            self.black_time = btime
            self.last_tick_s = time()
        elif self.white_time:
            if not self.board or self.game_state not in [GameState.RUNNING]:
                return
            if self.board.turn == chess.WHITE:
                self.white_time = self.white_time - timedelta(seconds=(time() - self.last_tick_s))
            else:
                self.black_time = self.black_time - timedelta(seconds=(time() - self.last_tick_s))
            self.last_tick_s = time()

        if self.white_time:
            if self.white_time.hour:
                self.labels["labelGameplayTimeWhite"] = self.white_time.strftime("%H:%M:%S")
            else:
                self.labels["labelGameplayTimeWhite"] = self.white_time.strftime("%M:%S")

        if self.black_time:
            if self.black_time.hour:
                self.labels["labelGameplayTimeBlack"] = self.black_time.strftime("%H:%M:%S")
            else:
                self.labels["labelGameplayTimeBlack"] = self.black_time.strftime("%M:%S")

    def provide_feedback(self, feedback):
        if feedback.show_message:
            self.labels["labelStatus"] = feedback.message
        if feedback.log_message:
            if self.feedback_last and self.feedback_last.message != feedback.message:
                self.log(feedback.message)
        if (feedback.speak_message or feedback.speak_message_overwrite) and not self.settings.sound_muted:
            tts = gTTS(text=(feedback.speak_message_overwrite or feedback.message), lang="en", tld="us", slow=True)
            with tempfile.NamedTemporaryFile() as fp:
                tts.save(fp.name)
                playsound(fp.name, block=False)

    @Slot()
    def lichess_game_state_worker(self, *args, **kwargs):
        while True:
            if not self.lichess_game:
                if self.close_requested:
                    break
                sleep(1)
                continue

            stop_streaming = False
            for game_data in self.lichess_api.board.stream_game_state(self.lichess_game["gameId"]):
                print(game_data)

                if self.close_requested:
                    Feedback(self, "Resigning...")
                    self.lichess_api.board.resign_game(self.lichess_game["gameId"])
                    stop_streaming = True
                    break

                game_state = None
                if game_data["type"] == "gameFull":
                    game_state = game_data["state"]
                elif game_data["type"] == "gameState":
                    if game_data["status"] == "aborted":
                        Feedback(self, "Game aborted", speak_message=True)
                        stop_streaming = True
                        break
                    elif game_data["status"] == "outoftime":
                        Feedback(self, f"Out of time, {game_data['winner']} wins", speak_message=True)
                        stop_streaming = True
                        break
                    elif game_data["status"] == "resign":
                        if game_data["winner"] == chess.COLOR_NAMES[self.lichess_color]:
                            Feedback(self, f"Opponent resigned", speak_message=True)
                        else:
                            Feedback(self, f"You resigned", speak_message=True)
                        stop_streaming = True
                        break

                    game_state = game_data
                    self.update_clock_times(game_state["wtime"], game_state["btime"])
                elif game_data["type"] == "opponentGone":
                    if game_data["gone"] is True:
                        Feedback(
                            self,
                            f"Opponent gone, you can claim victory in {game_data['claimWinInSeconds']} seconds",
                            speak_message=True,
                        )
                        if game_data["claimWinInSeconds"] == 0:
                            self.lichess_api.board.claim_victory(self.lichess_game["gameId"])

                if game_state:
                    if not game_state["moves"]:
                        continue
                    move = chess.Move.from_uci(game_state["moves"].split(" ")[-1])
                    self.lichess_game["move"] = move

            if stop_streaming:
                self.lichess_game["ended"] = True
                break

    @Slot()
    def game_loop(self, *args, **kwargs):
        board_edges = None
        board_warped = None
        squares_coords = None
        last_move = None
        validation_count = 0
        validation_count_needed_initial = 4
        its_opponents_turn = True
        board_image_saved = False

        print(self.opponent)
        if self.opponent in ["computer_white", "computer_black"]:
            game_mode = GameMode.COMPUTER
        elif self.opponent == "lichess":
            game_mode = GameMode.LICHESS
        elif self.opponent == "human":
            game_mode = GameMode.HUMAN
        else:
            raise NotImplementedError(f"No idea what game type to choose for opponent={self.opponent}")

        while True:
            if self.close_requested:
                break

            if self.current_frame is None:
                with self._camera_switch_mutex:
                    if self._camera.isActive():
                        self.camera_capture.capture()
                continue

            self.frame_num += 1
            if self.frame_num % 3 != 0:
                continue

            frame = copy(self.current_frame)
            validation_count_needed = validation_count_needed_initial

            with self._camera_switch_mutex:
                if self._camera.isActive():
                    self.camera_capture.capture()

            if self.frame_num % 10 == 0:
                self.log(f"Frame nr. {self.frame_num}", stdout_only=True)
            if self.board_detector_state == BoardDetectorState.NULL:
                self.debug_images_buffer.append(frame)
                sleep(0.1)
                continue

            # detect board corners
            if self.board_detector_state == BoardDetectorState.RUNNING_CORNER_DETECTION:
                self.labels["labelStatus"] = "Board corner detection"
                if self.close_requested:
                    break

                result = self.board_detector.detect_board_corners(frame)
                if result.success:
                    # board detected!
                    board_edges = result.detected_obj
                    board_warped = result.image
                    self.board_detector_state = BoardDetectorState.RUNNING_SQUARE_DETECTION
                    self.log("detect corners: success")
                else:
                    self.log("detect corners: failed, next try...")

                # save board image for training data
                if not board_image_saved and self.settings.collect_training_data:
                    train_im_path = self.settings.collect_training_data_dir / "boards"
                    train_im_path.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(train_im_path / f"{uuid4()}.jpg"), frame)
                    board_image_saved = True

            # detect squares
            if self.board_detector_state == BoardDetectorState.RUNNING_SQUARE_DETECTION:
                self.labels["labelStatus"] = "Square detection"
                squares_coords = None

                # try with different hough line thresholds
                for hough_lines_threshold in reversed(range(30, 160, 10)):
                    if self.close_requested:
                        break

                    result = self.board_detector.detect_squares(board_warped, hough_lines_threshold)
                    if result.success:
                        # squares detected!
                        squares_coords = result.detected_obj
                        self.board_detector_state = BoardDetectorState.DETECTED
                        self.log(f"detect squares: success, HoughLinesP threshold used={hough_lines_threshold}")
                        if not self.settings.visual_debug_delay:
                            # no need to show old debug images...
                            self.debug_images_buffer.clear()
                        break
                    else:
                        self.log(f"detect squares: failed, HoughLinesP threshold used={hough_lines_threshold}")

                # detection failed
                if not squares_coords:
                    # failed, back to corner detection
                    self.log("detect squares: failed, next try...")
                    self.board_detector_state = BoardDetectorState.RUNNING_CORNER_DETECTION

            if self.board_detector_state != BoardDetectorState.DETECTED:
                continue

            # board detected!
            image = self.board_detector.prepare_image(copy(frame))
            image = four_point_transform(image, board_edges)
            image_4p_transformed = copy(image)
            squares = self.board_detector.cut_squares(image, squares_coords)
            if squares is None:
                # cutting was not successful
                self.board_detector_state = BoardDetectorState.RUNNING_CORNER_DETECTION
                continue

            if self.board.a1_corner is None:
                board_orientation_result = self.board.update_squares(squares)
                self.game.update(a1_corner=self.board.a1_corner)
                self.log(board_orientation_result)

            squares.sort(self.board.a1_corner)
            for s in squares.get_flat():
                image = cv2.circle(image, (int(s.pt1.x), int(s.pt1.y)), 4, (255, 0, 255), -1)
                image = cv2.circle(image, (int(s.pt2.x), int(s.pt2.y)), 4, (255, 0, 255), -1)
                debug_text = str(round(s.cl_probability))

                if s.cl == Square.CL_EMPTY:
                    image = cv2.putText(
                        image,
                        debug_text,
                        (int(s.pt1.x + 25), int(s.pt2.y - 25)),
                        cv2.FONT_ITALIC,
                        1.0,
                        (255, 255, 0),
                        2,
                    )
                elif s.cl == Square.CL_BLACK:
                    image = cv2.putText(
                        image,
                        debug_text,
                        (int(s.pt1.x + 25), int(s.pt2.y - 25)),
                        cv2.FONT_ITALIC,
                        1.0,
                        (112, 128, 144),
                        2,
                    )
                else:
                    image = cv2.putText(
                        image,
                        debug_text,
                        (int(s.pt1.x + 25), int(s.pt2.y - 25)),
                        cv2.FONT_ITALIC,
                        1.0,
                        (255, 255, 255),
                        2,
                    )
            self.debug_images_buffer.append(image)

            if self.game_state == GameState.PAUSED:
                self.labels["labelStatus"] = self.pause_text
                sleep(0.1)
                continue

            # create a lichess game
            if game_mode == GameMode.LICHESS and not self.lichess_game:
                # start a new lichess game
                color = chess.COLOR_NAMES[self.lichess_color]
                Feedback(
                    self,
                    message=f"Waiting for game to start ({color}) {self.settings.lichess_time_m}+{self.settings.lichess_increment_s}",
                )
                self.lichess_api.board.seek(
                    self.settings.lichess_time_m,
                    self.settings.lichess_increment_s,
                    rated=self.settings.lichess_is_rated,
                    color=color,
                )

                games = self.lichess_api.games.get_ongoing()
                if len(games) == 0:
                    Feedback(self, message=f"Started lichess game not found")
                    break
                elif len(games) > 1:
                    Feedback(self, message=f"Starting multiple lichess games is not yet supported")
                    break

                Feedback(self, message=f"Game is ready", speak_message=True)

                self.lichess_game = games[0]
                self.lichess_game["move"] = None
                self.lichess_game["ended"] = False

                # get metadata
                for game_data in self.lichess_api.board.stream_game_state(self.lichess_game["gameId"]):
                    if game_data["type"] == "gameFull":
                        self.labels["labelGameplayPlayerWhite"] = (
                            f"<b>{game_data['white']['title'] or ''}</b> "
                            f"{game_data['white']['name']} "
                            f"<font color='grey'>({game_data['white']['rating']})</font>"
                        )

                        self.labels["labelGameplayPlayerBlack"] = (
                            f"<b>{game_data['black']['title'] or ''}</b> "
                            f"{game_data['black']['name']} "
                            f"<font color='grey'>({game_data['black']['rating']})</font>"
                        )
                        break

                its_opponents_turn = False
                if self.lichess_color == chess.BLACK:
                    its_opponents_turn = True

            move = None
            if game_mode == GameMode.COMPUTER and its_opponents_turn:
                # make engine move
                self.labels["labelStatus"] = "Computer is thinking"

                engine_move = None
                move_type = ""

                # opening book move available?
                if self.opening_book_reader:
                    entries = list(self.opening_book_reader.find_all(self.board))
                    if entries:
                        entry = choice(entries)
                        engine_move = entry.move
                        move_type = "Opening Book"

                # fallback to engine move
                if not engine_move:
                    engine_move = self.engine.play(self.board, self.game.engine_time_s).move
                    move_type = "Engine"

                # was the game paused in the meantime?
                if self.game_state == GameState.PAUSED:
                    continue

                from_square = chess.SQUARE_NAMES[engine_move.from_square]
                to_square = chess.SQUARE_NAMES[engine_move.to_square]
                Feedback(
                    self,
                    message=f"Waiting for {chess.COLOR_NAMES[self.board.turn]} to move: {from_square} to {to_square}",
                    speak_message_overwrite=f"{from_square} to {to_square}",
                    log_message=True,
                )
                self.log(f"{move_type} move: {from_square} to {to_square}")
                its_opponents_turn = False
            elif game_mode == GameMode.LICHESS and its_opponents_turn is True:
                break_game_loop = False
                while True:
                    if self.lichess_game["ended"]:
                        self.lichess_game = None
                        break_game_loop = True
                        self.game_state = GameState.FINISHED
                        break
                    if self.lichess_game["move"]:
                        if not self.board.move_stack or self.board.move_stack[-1] != self.lichess_game["move"]:
                            move = self.lichess_game["move"]
                            last_move = move
                            self.lichess_game["move"] = None
                            from_square = chess.SQUARE_NAMES[move.from_square]
                            to_square = chess.SQUARE_NAMES[move.to_square]
                            validation_count = validation_count_needed * 10
                            Feedback(
                                self,
                                message=f"Move {chess.COLOR_NAMES[self.board.turn]} from {from_square} to {to_square}",
                                speak_message_overwrite=f"{from_square} to {to_square}",
                                log_message=True,
                            )
                            break
                    else:
                        sleep(0.5)

                if break_game_loop:
                    break

            if game_mode in [GameMode.COMPUTER, GameMode.HUMAN] or (
                game_mode == GameMode.LICHESS and its_opponents_turn is False
            ):
                # board diff
                move = self.board.diff(squares)
                if self.board.move_stack and self.board.move_stack[-1] == move:
                    continue

            if move is None:
                continue

            if move != last_move:
                last_move = move
                validation_count = 0
                continue
            else:
                # castling is harder to detect => more checks needed here
                if self.board.has_castling_rights(chess.BLACK) or self.board.has_castling_rights(chess.WHITE):
                    # king or rook moved?
                    squares_to_diff = list(squares.get_flat())
                    king_square = self.board.king(self.board.turn)
                    king_moved = squares_to_diff[king_square].cl == Square.CL_EMPTY
                    rook_moved = False
                    rook_squares = list(self.board.pieces(chess.ROOK, self.board.turn))
                    for square in rook_squares:
                        if squares_to_diff[square].cl == Square.CL_EMPTY:
                            rook_moved = True
                            break
                    if king_moved or rook_moved:
                        validation_count_needed *= 3

                validation_count += 1

            if validation_count < validation_count_needed:
                continue

            # moved!
            if game_mode == GameMode.LICHESS and not its_opponents_turn:
                self.lichess_api.board.make_move(self.lichess_game["gameId"], move.uci())

            self.board.push(move)
            self.board.update_squares(squares)

            if game_mode == "computer" and self.board.turn != self.engine_color:
                Feedback(self, message="Waiting for your move")
            elif game_mode == GameMode.HUMAN:
                Feedback(self, message=f"Waiting for {chess.COLOR_NAMES[self.board.turn]} to move")

            last_move = None
            validation_count = 0

            if game_mode == GameMode.COMPUTER:
                if self.board.turn != self.engine_color:
                    its_opponents_turn = True
                else:
                    its_opponents_turn = False
            elif game_mode == GameMode.LICHESS:
                if self.board.turn == self.lichess_color:
                    its_opponents_turn = False
                else:
                    its_opponents_turn = True

            self.game.update(
                move_stack=self.board.move_stack, board_fen=self.board.board_fen(), a1_corner=self.board.a1_corner
            )
            if not self.settings.sound_muted:
                playsound(conf.GUI_RES_DIR.joinpath("move.wav"))
            self.update_board_rendering()

            # game over?
            if game_mode in [GameMode.COMPUTER, GameMode.HUMAN]:
                outcome = self.board.outcome(claim_draw=True)
                if outcome is not None:
                    if outcome.termination == chess.Termination.CHECKMATE:
                        Feedback(self, message="Game over, checkmate!", speak_message=True)
                    elif outcome.termination == chess.Termination.STALEMATE:
                        Feedback(self, message="Game over, stalemate!", speak_message=True)
                    elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
                        Feedback(self, message="Game over, insufficient material!", speak_message=True)
                    elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                        Feedback(self, message="Game over, fivefold repetition!", speak_message=True)
                    elif outcome.termination == chess.Termination.THREEFOLD_REPETITION:
                        Feedback(self, message="Game over, threefold repetition!", speak_message=True)
                    elif outcome.termination == chess.Termination.VARIANT_WIN:
                        Feedback(self, message="Game over, variant win!", speak_message=True)
                    elif outcome.termination == chess.Termination.VARIANT_LOSS:
                        Feedback(self, message="Game over, variant loss!", speak_message=True)
                    elif outcome.termination == chess.Termination.VARIANT_DRAW:
                        Feedback(self, message="Game over, variant draw!", speak_message=True)

                    self.game.update(move_stack=self.board.move_stack, winner=outcome.winner)
                    self.game_state = GameState.FINISHED
                    break

            # check?
            if game_mode in [GameMode.COMPUTER, GameMode.LICHESS]:
                if self.board.is_check():
                    Feedback(self, message="check!", speak_message=True)

            # collect squares training data
            if self.settings.collect_training_data:
                squares = self.board_detector.cut_squares(image_4p_transformed, squares_coords)
                if squares is None:
                    continue
                squares.sort(self.board.a1_corner)
                for square in squares.get_flat():
                    if square.cl_probability < self.settings.collect_training_data_threshold_perc:
                        train_im_path = self.settings.collect_training_data_dir / "squares" / square.cl_readable
                        train_im_path.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(train_im_path / f"{uuid4()}.jpg"), square.image)

    def close(self, *args):
        if self.close_requested:
            return

        self.log("close requested, cleaning things up...")
        self.close_requested = True
        if self._camera and self._camera.isActive():
            self._camera.stop()
        if self.engine:
            self.engine.quit()
        self.update_ui_timer.stop()
        self.vis_debug_timer_detector.stop()
        self.threadpool.waitForDone()
        QApplication.exit(0)
