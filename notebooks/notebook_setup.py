import sys
from pathlib import Path

print(str((Path(".").parent / "..")))

sys.path.insert(0, str((Path(".").parent / "..")))
