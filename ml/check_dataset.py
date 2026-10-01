from pathlib import Path
from PIL import Image

ROOT = Path("data/disease_dataset")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total = 0
bad = []

for path in ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
        total += 1

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            bad.append((str(path), str(e)))

print(f"Image files checked: {total}")
print(f"Corrupted/unreadable: {len(bad)}")

if bad:
    print("\nProblem files:")
    for path, error in bad:
        print(path)
        print(f"  {error}")
else:
    print("SUCCESS: All images passed the integrity check.")