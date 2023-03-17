from typing import Optional

import numpy as np
from numpy.linalg import norm


class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.arr = np.array([x, y])

    def __str__(self):
        return f"({self.x},{self.y})"

    def __sub__(self, other):
        return other.arr - self.arr

    def __eq__(self, other):
        return self.set == other.set

    def set(self) -> tuple[int, int]:
        return int(self.x), int(self.y)


class Line:
    HORIZONTAL = 0
    VERTICAL = 1

    def __init__(self, p1, p2) -> None:
        self.p1 = p1
        self.p2 = p2
        self.arr = np.array([p1, p2])

    def __str__(self):
        return f"[{self.p1} -> {self.p2}]"

    def __hash__(self):
        return hash(f"{self.p1.x} {self.p1.y} {self.p2.x} {self.p2.y}")

    def __eq__(self, other):
        return (self.p1 == other.p1 and self.p2 == other.p2) or (self.p1 == other.p2 and self.p2 == other.p1)

    def set(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.p1.x, self.p1.y), (self.p2.x, self.p2.y)

    def set_flat(self) -> tuple[float, float, float, float]:
        return self.p1.x, self.p1.y, self.p2.x, self.p2.y

    @property
    def dx(self) -> float:
        return self.p2.x - self.p1.x

    @property
    def dy(self) -> float:
        return self.p2.y - self.p1.y

    @property
    def slope(self) -> float:
        return self.dy / self.dx

    @property
    def direction(self) -> int:
        if abs(self.dx) > abs(self.dy):
            return self.HORIZONTAL
        else:
            return self.VERTICAL

    def distance(self, other) -> float:
        d1 = np.abs(np.cross(self.p2 - self.p1, self.p1 - other.p1) / norm(self.p2 - self.p1))
        d2 = np.abs(np.cross(self.p2 - self.p1, self.p1 - other.p2) / norm(self.p2 - self.p1))
        return (d1 + d2) / 2

    def intersection(self, other) -> Optional[Point]:
        # line 1 dy, dx and determinant
        a11 = self.p1.y - self.p2.y
        a12 = self.p2.x - self.p1.x
        b1 = self.p1.x * self.p2.y - self.p2.x * self.p1.y

        # line 2 dy, dx and determinant
        a21 = other.p1.y - other.p2.y
        a22 = other.p2.x - other.p1.x
        b2 = other.p1.x * other.p2.y - other.p2.x * other.p1.y

        # linear system, coefficient matrix
        a = np.array([[a11, a12], [a21, a22]])

        # right hand side vector
        b = -np.array([b1, b2])
        # solve
        try:
            intersection_point = np.linalg.solve(a, b)
            return Point(*intersection_point)
        except np.linalg.LinAlgError:
            return None
