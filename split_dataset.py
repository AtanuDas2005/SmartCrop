
from pathlib import Path
import random
import shutil

# Original dataset: do not modify it
SOURCE_DIR = Path("dataset/tomato")

# New folder for the split dataset
OUTPUT_DIR = Path("dataset_split")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

random.seed(42)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found: {SOURCE_DIR.resolve()}"
    )

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"{OUTPUT_DIR} already contains files. "
        "Rename or remove it before creating a new split."
    )

for class_dir in sorted(SOURCE_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    images = [
        p for p in class_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.20)

    splits = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in splits.items():
        destination = OUTPUT_DIR / split_name / class_dir.name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(image_path, destination / image_path.name)

    print(
        f"{class_dir.name}: "
        f"train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, "
        f"test={len(splits['test'])}"
    )

print("\nDataset split completed!")
print("Output:", OUTPUT_DIR.resolve())
