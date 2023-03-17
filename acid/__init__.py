import io
from pathlib import Path

__version__ = f"v{io.open(Path(__file__).parent / '..' / 'VERSION').readline().strip()}"
