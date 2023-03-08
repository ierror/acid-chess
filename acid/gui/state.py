class GameState:
    NULL = 0
    RUNNING = 1
    PAUSED = 2
    FINISHED = 3


class BoardDetectorState:
    NULL = 0
    RUNNING_CORNER_DETECTION = 1
    RUNNING_SQUARE_DETECTION = 2
    DETECTED = 3
