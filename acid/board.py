import math
from dataclasses import dataclass
from statistics import mean

import chess
import cv2
import numpy as np
import torch.nn
from cv2.mat_wrapper import Mat
from imutils import auto_canny
from imutils.perspective import four_point_transform, order_points
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from .geometry import Line, Point
from .torch.setup import BoardModelSetup, SquareModelSetup, get_trainer
from .utils.cv2 import put_text, resize_and_pad
from .utils.timezone import datetime_local


@dataclass
class Result:
    message: str
    image: object = None
    success: bool = True
    detected_obj: object = None
    timestamp: object = None

    def __post_init__(self):
        self.timestamp = datetime_local()


class Detector:
    image = None
    image_size = (BoardModelSetup.image_size[0] * 4, BoardModelSetup.image_size[1] * 4)

    def __init__(self, debug_images_buffer):
        self.debug_images_buffer = debug_images_buffer

    def prepare_image(self, image):
        height, width, *_ = image.shape
        if width == self.image_size[0] and height == self.image_size[1]:
            # image already in the right shape
            return image
        return resize_and_pad(image, self.image_size)

    def report_progress(self, message, debug_image):
        # draw message on debug image
        debug_image = cv2.resize(debug_image, (1280, 720))
        put_text(debug_image, message, pos=(10, 10))
        self.debug_images_buffer.append(debug_image)

    @torch.no_grad()
    def detect_board_corners(self, image):
        self.image = self.prepare_image(image)
        self.image_debug = self.image.copy()

        # resize to fit to the detector model
        image_detector = self.image.copy()
        image_detector = cv2.resize(image_detector, BoardModelSetup.image_size)
        image_detector = cv2.cvtColor(image_detector, cv2.COLOR_BGR2RGB)
        self.report_progress("detect board corners: resized board", image_detector)

        # transform
        image_detector = BoardModelSetup.transforms["val"](image=image_detector)["image"]
        self.report_progress("detect board corners: transformed board", np.asarray((to_pil_image(image_detector))))

        # create model and predict mask of board
        model = BoardModelSetup.model
        model.eval()
        prediction = model(image_detector.unsqueeze(0))

        try:
            mask = prediction[0].detach().numpy()
            mask[mask > 0] = 0
            mask[mask < 0] = 1
        except IndexError:
            return Result("detect board corners: masking failed", self.image_debug, success=False)

        gray = cv2.cvtColor(mask[0, :, :], cv2.COLOR_GRAY2BGR)
        gray = (255 / gray.max() * (gray - gray.min())).astype(np.uint8)
        _, gray = cv2.threshold(gray, 110, 1, cv2.THRESH_BINARY)

        # resize back to working size and extract contours from mask image
        gray = self.prepare_image(gray)
        contours, _hierarchy = cv2.findContours(gray[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Result("detect board corners: bord contour extraction failed", self.image_debug, success=False)

        contour = max(contours, key=cv2.contourArea)
        contour_len = cv2.arcLength(contour, True)
        board_edges = cv2.approxPolyDP(contour, 0.05 * contour_len, True)
        board_edges = board_edges.reshape(-1, 2)
        if board_edges is None or len(board_edges) != 4:
            return Result(
                "detect board corners: bord contour extraction failed after approxPolyDP",
                self.image_debug,
                success=False,
            )

        # first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        board_edges = order_points(board_edges)
        board_edges[0] = (board_edges[0][0] - 18, board_edges[0][1] - 18)
        board_edges[1] = (board_edges[1][0] + 18, board_edges[1][1] - 18)
        board_edges[2] = (board_edges[2][0] + 18, board_edges[2][1] + 18)
        board_edges[3] = (board_edges[3][0] - 18, board_edges[3][1] + 18)

        for e in board_edges:
            cv2.circle(self.image_debug, (int(e[0]), int(e[1])), 25, (255, 0, 255), -1)
        self.report_progress("detect board corners: detected edges", self.image_debug)

        try:
            warped = four_point_transform(self.image, board_edges)
            self.image_debug = warped.copy()
            self.report_progress("detect board corners: img warped", self.image_debug)
        except ValueError:
            return Result("detect board corners: four_point_transform failed", self.image_debug, success=False)

        return Result("detect board corners: board warped", image=warped, detected_obj=board_edges)

    def detect_squares(self, warped, hough_lines_threshold):
        if warped is None or not warped.any():
            return Result("detect squares: warped can't be empty", warped, success=False)

        image_h, image_w = warped.shape[0:2]

        image = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        self.report_progress("detect squares: img grayed", image)

        image = cv2.medianBlur(image, 17)
        self.report_progress("detect squares: blurred", image)

        image = auto_canny(image)
        self.report_progress("detect squares: auto canny", image)

        lines = cv2.HoughLinesP(
            image,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_lines_threshold,
            minLineLength=min(image_w, image_h) * 0.5,
            maxLineGap=max(image_w, image_h),
        )
        if lines is None or len(lines) < 18:
            return Result(f"detect squares: len(lines) < 18", image, success=False)

        lines_horizontal = []
        lines_vertical = []

        for line in lines:
            line = Line(Point(line[0][0], line[0][1]), Point(line[0][2], line[0][3]))
            if line.direction == Line.HORIZONTAL:
                lines_horizontal.append(line)
            else:
                lines_vertical.append(line)

        if not lines_horizontal or not lines_vertical:
            return Result("detect squares: no v or h lines", image, success=False)

        # sort lines
        lines_horizontal.sort(key=lambda l: l.p1.y + l.p2.y)
        lines_vertical.sort(key=lambda l: l.p1.x + l.p2.x)

        # remove duplicates
        for lines, image_size in [(lines_horizontal, image_h), ([lines_vertical, image_w])]:
            line_prev = None
            for line in list(lines):
                if line_prev is not None:
                    distance = line.distance(line_prev)
                    # filter by distance
                    if distance < image_size / 11:
                        lines.remove(line)
                        continue
                    else:
                        line_prev = line
                else:
                    line_prev = line

        # remove outliers by slope
        for lines in [lines_horizontal, lines_vertical]:
            if len(lines) == 9:
                continue
            m_mean = mean([abs(line.slope) for line in lines])
            for idx, line in enumerate(lines):
                if abs(m_mean - abs(line.slope)) > m_mean * 0.05:
                    lines.pop(idx)

        # take middle line and walk to bottom and top
        middle_line_index = len(lines_horizontal) // 2
        lines_horizontal_final = [lines_horizontal[middle_line_index]]
        for index in range(1, len(lines_horizontal) // 2 + 1):
            for up_down in (-1, 1):
                line_index = middle_line_index + index * up_down
                if line_index < 0 or line_index == len(lines_horizontal):
                    continue
                line = lines_horizontal[line_index]
                lines_horizontal_final.append(line)
                lines_horizontal_final.sort(key=lambda l: l.p1.y + l.p2.y)
                if len(lines_horizontal_final) == 9:
                    break
            if len(lines_horizontal_final) == 9:
                break

        for line in lines_horizontal_final:
            cv2.line(
                self.image_debug,
                (int(line.p1.x), int(line.p1.y)),
                (int(line.p2.x), int(line.p2.y)),
                (255, 255, 0),
                4,
                cv2.LINE_AA,
            )
        self.report_progress("detect squares: horizontal lines", self.image_debug)

        if len(lines_horizontal_final) != 9:
            return Result("detect squares: len(lines_horizontal_final) != 9", image, success=False)

        # take middle line and walk to left and right
        middle_line_index = len(lines_vertical) // 2
        lines_vertical_final = [lines_vertical[middle_line_index]]
        for index in range(1, len(lines_vertical) // 2 + 1):
            for left_right in (-1, 1):
                line_index = middle_line_index + index * left_right
                if line_index < 0 or line_index == len(lines_vertical):
                    continue

                line = lines_vertical[line_index]
                lines_vertical_final.append(line)
                lines_vertical_final.sort(key=lambda l: l.p1.x + l.p2.x)
                if len(lines_vertical_final) == 9:
                    break
            if len(lines_vertical_final) == 9:
                break

        for line in lines_vertical_final:
            cv2.line(
                self.image_debug,
                (int(line.p1.x), int(line.p1.y)),
                (int(line.p2.x), int(line.p2.y)),
                (255, 0, 0),
                4,
                cv2.LINE_AA,
            )
        self.report_progress("detect squares: vertical lines", self.image_debug)

        if len(lines_vertical_final) != 9:
            return Result("detect squares: len(lines_vertical_final) != 9", image, success=False)

        square_corners = []
        for h_line in lines_horizontal_final:
            for v_line in lines_vertical_final:
                intersect = h_line.intersection(v_line)
                if intersect is None:
                    continue

                is_duplicate = False
                for d in square_corners:
                    try:
                        if math.sqrt((d.x - intersect.x) ** 2 + (d.y - intersect.y) ** 2) < 50:
                            is_duplicate = True
                            break
                    except ValueError:
                        square_corners.append(intersect)
                        break
                if not is_duplicate:
                    square_corners.append(intersect)
                else:
                    pass

        for sc in square_corners:
            cv2.circle(self.image_debug, (int(sc.x), int(sc.y)), 5, (0, 0, 255), -1)

        self.report_progress("detect squares: horizontal lines: squares corners", self.image_debug)

        if len(square_corners) != 81:
            return Result(
                f"detect squares: squares not detected, len(square_corners) != 81 ({len(square_corners)})",
                self.image_debug,
                success=False,
            )

        # sort square coords from top to bottom, followed by left to right
        square_coords = [[] for _ in range(0, 9)]
        square_corners.sort(key=lambda p: p.y)
        row_index = 0

        for c in range(0, 81):
            if c > 0 and c % 9 == 0:
                row_index += 1

            square_coords[row_index].append(square_corners[c])

            # sort by x if all squares of a row are available
            if len(square_coords[row_index]) == 9:
                square_coords[row_index].sort(key=lambda p: p.x)

        return Result(
            f"detect squares: squares detected, returning extracted square coordinates",
            self.image_debug,
            detected_obj=square_coords,
        )

    def cut_squares(self, board_image, coords):
        image_h, image_w = board_image.shape[0:2]
        squares = [[] for _ in range(0, 8)]

        for row_index in range(0, 8):
            for col_index in range(0, 8):
                corner_pt1 = coords[row_index][col_index]
                corner_pt2 = coords[row_index + 1][col_index + 1]

                squares_img = board_image[
                    int(min(max(corner_pt1.y, 0), image_h)) : int(min(max(corner_pt2.y, 0), image_h)),
                    int(min(max(corner_pt1.x, 0), image_w)) : int(min(max(corner_pt2.x, 0), image_w)),
                ]
                if squares_img is None:
                    raise None
                square = Square(squares_img, corner_pt1, corner_pt2)
                squares[row_index].append(square)

        return Squares(squares)


class Board(chess.Board):
    # row, column indexes
    INDEX_BOTTOM_LEFT = (0, 0)
    INDEX_BOTTOM_RIGHT = (0, 7)
    INDEX_TOP_LEFT = (7, 0)
    INDEX_TOP_RIGHT = (7, 7)

    squares = [[]]
    a1_corner = None

    def _determine_orientation(self, squares):
        # determine board orientation
        if self.a1_corner is not None:
            return

        # 1. horizontal or vertical?
        squares = Squares(squares)
        empty_squares_cnt = 0
        for row in range(0, 8):
            for col in [2, 3, 4, 5]:
                square = squares[row][col]
                if square.cl == square.CL_EMPTY:
                    empty_squares_cnt += 1

        print(empty_squares_cnt)
        if empty_squares_cnt >= 25:
            board_orientation = "vertical"
        else:
            board_orientation = "horizontal"

        # count white pieces of the 4 top-left squares
        top_left_4_white_pieces_num = 0
        for s in [squares[0][0], squares[0][1], squares[1][0], squares[1][1]]:
            if s.cl == Square.CL_WHITE:
                top_left_4_white_pieces_num += 1

        top_left_piece_color = Square.CL_BLACK
        if top_left_4_white_pieces_num >= 3:
            top_left_piece_color = Square.CL_WHITE

        if board_orientation == "vertical":
            if top_left_piece_color == Square.CL_WHITE:
                self.a1_corner = Board.INDEX_TOP_LEFT
                return "a1: top-left"
            else:
                self.a1_corner = Board.INDEX_BOTTOM_RIGHT
                return "a1: bottom-right"
        else:
            if top_left_piece_color == Square.CL_WHITE:
                self.a1_corner = Board.INDEX_TOP_RIGHT
                return "a1: top-right"
            else:
                self.a1_corner = Board.INDEX_BOTTOM_LEFT
                return "a1: bottom-left"

    def update_squares(self, squares):
        result = None
        if not self.a1_corner:
            result = self._determine_orientation(squares)
        self.squares = squares
        return result

    @property
    def squares_cl_probability_mean(self):
        return self.squares.cl_probability_mean

    def diff(self, squares_to_diff):
        cl_probability_min_bw = 0.96
        cl_probability_min_empty = 0.98
        squares_to_diff = list(squares_to_diff.get_flat())
        legal_moves = self.generate_legal_moves()

        # king and rook moved?
        # => castling?
        king_square = self.king(self.turn)
        king_moved = squares_to_diff[king_square].cl == Square.CL_EMPTY
        rook_moved = False
        rook_squares = list(self.pieces(chess.ROOK, self.turn))
        for square in rook_squares:
            if squares_to_diff[square].cl == Square.CL_EMPTY:
                rook_moved = True
                break

        if king_moved and rook_moved:
            for move in legal_moves:
                square_from = squares_to_diff[move.from_square]
                square_to = squares_to_diff[move.to_square]
                if (
                    move.from_square == king_square
                    and square_to.cl != Square.CL_EMPTY
                    and abs(move.from_square - move.to_square) == 2
                    and square_from.cl_probability > cl_probability_min_bw
                    and square_to.cl_probability > cl_probability_min_bw
                ):
                    return move
            return None

        # std move
        for move in legal_moves:
            square_from = squares_to_diff[move.from_square]
            square_to = squares_to_diff[move.to_square]
            if (
                square_from.cl == Square.CL_EMPTY
                and square_to.cl == self.turn
                and square_from.cl_probability > cl_probability_min_empty
                and square_to.cl_probability > cl_probability_min_bw
            ):
                return move


class Square:
    CL_EMPTY = -1
    CL_WHITE = chess.WHITE
    CL_BLACK = chess.BLACK

    name = None
    cl = None
    cl_probability = 0

    def __init__(self, image: Mat, pt1: Point, pt2: Point):
        self.image = image
        self.pt1 = pt1
        self.pt2 = pt2

    @property
    def cl_readable(self):
        if self.cl == self.CL_EMPTY:
            return "empty"
        if self.cl == self.CL_WHITE:
            return "white"
        if self.cl == self.CL_BLACK:
            return "black"
        return "unclassified"


class Squares:
    def __init__(self, squares):
        self.squares = squares
        self._classify()

    def __str__(self):
        return str(
            chess.SquareSet([i for i, s in enumerate(self.get_flat()) if s.cl in (Square.CL_BLACK, Square.CL_WHITE)])
        )

    def __getitem__(self, row_index):
        return self.squares[row_index]

    def get_flat(self):
        for row in self.squares:
            for square in row:
                yield square

    def sort(self, a1_corner):
        if a1_corner == Board.INDEX_BOTTOM_LEFT:
            self.squares.reverse()
        elif a1_corner == Board.INDEX_BOTTOM_RIGHT:
            self.squares = [list(row) for row in zip(*self.squares)]
            for row in self.squares:
                row.reverse()
        elif a1_corner == Board.INDEX_TOP_RIGHT:
            for row in self.squares:
                row.reverse()
        elif a1_corner == Board.INDEX_TOP_LEFT:
            self.squares = [list(row) for row in zip(*self.squares)]
        else:
            raise NotImplementedError(f"a1_corner={a1_corner} unknown")

        # add names
        for index, square in enumerate(self.get_flat()):
            square.name = chess.SQUARE_NAMES[index]

    @torch.no_grad()
    def _classify(self):
        model = SquareModelSetup.model
        square_images = [cv2.cvtColor(square.image, cv2.COLOR_BGR2RGB) for square in self.get_flat()]
        square_images = [Image.fromarray(square) for square in square_images]
        square_images = [square.resize(SquareModelSetup.image_size) for square in square_images]
        square_images = [SquareModelSetup.transforms["val"](square) for square in square_images]
        dataloader = DataLoader(square_images, batch_size=64, shuffle=False, num_workers=0)
        predictions = get_trainer().predict(model, dataloader)
        prob = nn.functional.softmax(predictions[0], dim=1)
        pred_probabilities, pred_classes = prob.topk(1, dim=1)

        for index, square in enumerate(self.get_flat()):
            predicted_class_id = int(pred_classes[index])
            predicted_class = SquareModelSetup.classes[predicted_class_id]
            probability = float(pred_probabilities[index] * 100)

            if predicted_class == "black":
                square.cl = square.CL_BLACK
            elif predicted_class == "white":
                square.cl = square.CL_WHITE
            elif predicted_class == "empty":
                square.cl = square.CL_EMPTY
            else:
                raise NotImplementedError(f"Unknown predicted_class={predicted_class}")

            square.cl_probability = probability

    @property
    def cl_probability_mean(self):
        return mean([s.cl_probability for s in self.get_flat()])
