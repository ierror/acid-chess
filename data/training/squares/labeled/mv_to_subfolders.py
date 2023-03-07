#!/usr/bin/env python3
from pathlib import Path

for image_path in Path(__file__).parent.glob("*/*.jp*g"):
    target_dir = image_path.parent / image_path.name[0:2]
    target_dir.mkdir(parents=True, exist_ok=True)
    image_path.rename(target_dir / image_path.name)
