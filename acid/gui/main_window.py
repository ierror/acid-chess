import tempfile
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from multiprocessing import Lock
from pathlib import Path
from random import choice
from time import sleep
from uuid import uuid4

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
from PySide6.QtCore import QFile, QSize, QStandardPaths, QThreadPool, QTimer, Slot
from PySide6.QtGui import QFont, QFontDatabase, QIcon, QImage, QKeySequence, QPixmap, QShortcut, Qt
from PySide6.QtMultimedia import QCamera, QImageCapture, QMediaCaptureSession, QMediaDevices
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QVBoxLayout

from acid import conf
from acid.board import Board, Detector, Square
from acid.engines import Engine, engines
from acid.game import Game
from acid.gui.logs import Logger
from acid.gui.qt.reactive import (
    ReactiveAttrEnableDisable,
    ReactiveAttrIcon,
    ReactiveAttrPresence,
    ReactiveAttrSynced,
    ReactiveAttrToolTip,
)
from acid.gui.qt.widgets import ButtonOpensFileDialog, PictureLabelFitToParent, QPlainTextEditFocusSignaled
from acid.gui.settings import Settings
from acid.gui.state import BoardDetectorState, GameState
from acid.gui.threads import Worker

from acid.gui.res import icons  # isort:skip

UI_MAPPINGS = {
    "opponent": {chess.BLACK: "Bot Black", chess.WHITE: "Bot WHITE", "human": "Human (all colors)"},
    "engine": {engine_idx: engine.name for (engine_idx, engine) in enumerate(engines)},
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
        ReactiveAttrSynced(self.game, "opponent_type_idx", self.ui.comboBoxOpponent)

        self.ui.comboBoxEngine.removeItem(0)
        self.ui.comboBoxEngine.insertItems(0, UI_MAPPINGS["engine"].values())
        ReactiveAttrSynced(self.game, "engine_idx", self.ui.comboBoxEngine)

        self.ui.comboBoxCamera.removeItem(0)
        self.ui.comboBoxCamera.insertItems(0, [c.description() for c in self.cameras])
        ReactiveAttrSynced(self.game, "camera_idx", self.ui.comboBoxCamera)

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
        ReactiveAttrPresence(self.game, "opening_book_path", self.ui.pushButtonOpeningBookRemove, visible_for=True)

        # reactive bind ui elements to game state items
        ReactiveAttrSynced(self.game, "pgn_white", self.ui.lineEditPlayerWhite)
        ReactiveAttrSynced(self.game, "pgn_black", self.ui.lineEditPlayerBlack)
        ReactiveAttrSynced(self.game, "pgn_event", self.ui.lineEditEventName)
        ReactiveAttrSynced(self.game, "engine_time_s", self.ui.spinBoxEngineTime)
        ReactiveAttrSynced(self.game, "opening_book_path", opening_book)

        ReactiveAttrSynced(self.game, "engine_editor_text", self.ui.plainTextEditEngineOptions)

        ReactiveAttrToolTip(self.game, "opening_book_path", self.ui.pushButtonOpeningBook)

        # reactive bind ui elements to gui settings
        ReactiveAttrSynced(self.settings, "collect_training_data", self.ui.checkBoxCollectTrainingData)
        ReactiveAttrSynced(
            self.settings, "collect_training_data_threshold_perc", self.ui.spinBoxCollectTrainingDataThreshold
        )
        ReactiveAttrSynced(self.settings, "visual_debug_delay", self.ui.checkBoxVisualDebugDelay)

        ReactiveAttrSynced(self.settings, "sound_muted", self.ui.pushButtonMuteUnmute)
        values_to_icon = {
            (True,): QIcon(":/sound-off.svg"),
            (False,): QIcon(":/sound-high.svg"),
        }
        ReactiveAttrIcon(self.settings, "sound_muted", self.ui.pushButtonMuteUnmute, values_to_icon=values_to_icon)

        # disable some ui elements based when game has started
        ReactiveAttrEnableDisable(
            self, "game_state", self.ui.comboBoxOpponent, enable_for=[GameState.NULL, GameState.FINISHED]
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

        # show engine options based on selected  opponent
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
            ReactiveAttrPresence(self.game, "opponent_type_idx", ui_elm, visible_for=(0, 1))

        ReactiveAttrToolTip(self.settings, "save_games_dir", self.ui.pushButtonSaveGamesTo)
        ReactiveAttrToolTip(self.settings, "collect_training_data_dir", self.ui.pushButtonCollectTrainingDataSaveTo)

        re_detect_visible_for = (BoardDetectorState.DETECTED,)
        ReactiveAttrPresence(
            self, "board_detector_state", self.ui.pushButtonReDetectCorners, visible_for=re_detect_visible_for
        )
        self.ui.pushButtonReDetectCorners.setVisible(False)

        values_to_icon = {
            (GameState.NULL, GameState.FINISHED, GameState.PAUSED): QIcon(":/play.svg"),
            (GameState.RUNNING,): QIcon(":/pause.svg"),
        }
        ReactiveAttrIcon(self, "game_state", self.ui.pushButtonStartPause, values_to_icon=values_to_icon)

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

        # framegrabber worker
        self.threadpool = QThreadPool()
        fg_worker = Worker(self.gameloop)
        self.threadpool.start(fg_worker)

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
    def engine(self):
        if self.game.engine_idx is not None:
            return engines[self.game.engine_idx]

    @cached_property
    def cameras(self):
        # uses a cached property to ensure always the same order of cameras for this session
        return QMediaDevices.videoInputs()

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
                    self.provide_feedback(self.feedback_last)
                self.log("game resumed")
            self.game_state = GameState.RUNNING
        elif self.game_state == GameState.RUNNING:
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
        opponent_idx = self.game.opponent_type_idx
        opponent = list(UI_MAPPINGS["opponent"].keys())[opponent_idx]
        self.update_board_rendering()

        # set player name to BOT_NAME in case of bot games
        if opponent == chess.BLACK:
            self.game.pgn_black = conf.BOT_NAME
            self.game.pgn_white = None
        elif opponent == chess.WHITE:
            self.game.pgn_white = conf.BOT_NAME
            self.game.pgn_black = None
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

    def update_board_rendering(self):
        lastmove = None
        if self.board.move_stack:
            lastmove = self.board.move_stack[-1]

        rendered_image_out = BytesIO()
        orientation = chess.WHITE
        if self.game.opponent_type_idx == chess.WHITE:
            orientation = chess.BLACK

        svg_board = chess.svg.board(self.board, lastmove=lastmove, orientation=orientation)
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

    def provide_feedback(self, feedback):
        if feedback.show_message:
            self.labels["labelStatus"] = feedback.message
        if feedback.log_message:
            if self.feedback_last and self.feedback_last.message != feedback.message:
                self.log(feedback.message)
        if (feedback.speak_message or feedback.speak_message_overwrite) and not self.settings.sound_muted:
            tts = gTTS(text=(feedback.speak_message_overwrite or feedback.message), lang="en", tld="us")
            with tempfile.NamedTemporaryFile() as fp:
                tts.save(fp.name)
                playsound(fp.name, block=False)

    @Slot()
    def gameloop(self, *args, **kwargs):
        board_edges = None
        board_warped = None
        squares_coords = None
        last_move = None
        validation_count = 0
        engine_run_todo = True
        board_image_saved = False

        while True:
            if self.close_requested:
                break

            if self.current_frame is None:
                with self._camera_switch_mutex:
                    if self._camera.isActive():
                        self.camera_capture.capture()
                continue

            self.frame_num += 1
            frame = copy(self.current_frame)

            with self._camera_switch_mutex:
                if self._camera.isActive():
                    self.camera_capture.capture()

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

            # engine move
            if self.board.turn == self.engine_color and engine_run_todo:
                self.labels["labelStatus"] = "Bot is thinking"

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
                if self.game_state != GameState.PAUSED:
                    from_square = chess.SQUARE_NAMES[engine_move.from_square]
                    to_square = chess.SQUARE_NAMES[engine_move.to_square]
                    Feedback(
                        self,
                        message=f"Waiting for {chess.COLOR_NAMES[self.board.turn]} to move: {from_square} to {to_square}",
                        speak_message_overwrite=f"{from_square} to {to_square}",
                        log_message=False,
                    )
                    self.log(f"{move_type} move: {from_square} to {to_square}")
                    engine_run_todo = False

            if not self.engine or (self.engine and self.board.turn != self.engine_color):
                Feedback(self, message=f"Waiting for {chess.COLOR_NAMES[self.board.turn]} to move")

            move = self.board.diff(squares)
            if move is None:
                continue

            # validate move with last frame
            validation_count_needed = 2

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

            if self.board.turn == self.engine_color:
                engine_run_todo = True

            # moved!
            self.board.push(move)
            self.board.update_squares(squares)
            last_move = None
            validation_count = 0

            self.game.update(
                move_stack=self.board.move_stack, board_fen=self.board.board_fen(), a1_corner=self.board.a1_corner
            )
            if not self.settings.sound_muted:
                playsound(conf.GUI_RES_DIR.joinpath("move.wav"))
            self.update_board_rendering()

            # Game over?
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
            if self.engine and self.board.turn != self.engine_color:
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
