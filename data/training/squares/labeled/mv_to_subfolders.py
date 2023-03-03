#!/usr/bin/env python3
import os
from glob import glob
from pathlib import Path

for image_path in glob("*/*.jp*g"):
    image_path = Path(image_path)
    target_dir = image_path.parent / image_path.name[0:2]
    target_dir.mkdir(parents=True, exist_ok=True)
    os.rename(image_path, target_dir / image_path.name)
