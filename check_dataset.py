
from pathlib import Path
from PIL import Image

# Dataset location
DATASET_DIR = Path("dataset/tomato")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total_images = 0
total_corrupted = 0

if not DATASET_DIR.exists():
    print("Dataset folder not found:", DATASET_DIR.resolve())
else:
    print("\nTOMATO DATASET REPORT")
    print("-" * 55)

    for class_folder in sorted(DATASET_DIR.iterdir()):
        if not class_folder.is_dir():
            continue

        images = [
            p for p in class_folder.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        corrupted = 0

        for image_path in images:
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception:
                corrupted += 1

        valid = len(images) - corrupted
        total_images += valid
        total_corrupted += corrupted

        print(f"{class_folder.name}")
        print(f"  Valid images: {valid}")
        print(f"  Corrupted images: {corrupted}")
        print()

    print("-" * 55)
    print("Total valid images:", total_images)
    print("Total corrupted images:", total_corrupted)
