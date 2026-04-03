"""
Conversion des annotations VisDrone2019 vers le format YOLO.

VisDrone format : x_min,y_min,width,height,score,category,truncation,occlusion
YOLO format     : class x_center y_center width height  (tout normalisé 0-1)

Catégories VisDrone :
  0=ignored  1=pedestrian  2=people  3=bicycle  4=car  5=van
  6=truck  7=tricycle  8=awning-tricycle  9=bus  10=motor
On ignore la classe 0 et on remappe 1-10 → 0-9 pour YOLO.
"""

import os
import cv2
from pathlib import Path

# Dossier racine du dataset
BASE = Path("S:/YOLO-DS/data/VisDrone/images")

SPLITS = [
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-test-dev",
]

def convert_split(split: str):
    img_dir = BASE / split / "images"
    ann_dir = BASE / split / "annotations"
    lbl_dir = BASE / split / "labels"
    lbl_dir.mkdir(exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    print(f"\n[{split}] {len(ann_files)} fichiers à convertir...")

    converted, skipped = 0, 0

    for ann_file in ann_files:
        # Image correspondante
        img_file = img_dir / ann_file.with_suffix(".jpg").name
        if not img_file.exists():
            skipped += 1
            continue

        # Lire dimensions image
        img = cv2.imread(str(img_file))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        yolo_lines = []
        with open(ann_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue

                x_min = int(parts[0])
                y_min = int(parts[1])
                bw    = int(parts[2])
                bh    = int(parts[3])
                cat   = int(parts[5])

                # Ignorer classe 0 (ignored region)
                if cat == 0 or cat > 10:
                    continue

                # Conversion vers YOLO (normalisé, centré)
                x_center = (x_min + bw / 2) / w
                y_center  = (y_min + bh / 2) / h
                bw_norm   = bw / w
                bh_norm   = bh / h
                cls       = cat - 1  # remappe 1-10 → 0-9

                # Clamp pour éviter valeurs hors [0,1]
                x_center  = max(0, min(1, x_center))
                y_center  = max(0, min(1, y_center))
                bw_norm   = max(0, min(1, bw_norm))
                bh_norm   = max(0, min(1, bh_norm))

                yolo_lines.append(
                    f"{cls} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}"
                )

        # Écrire le fichier label YOLO
        lbl_file = lbl_dir / ann_file.name
        with open(lbl_file, "w") as f:
            f.write("\n".join(yolo_lines))
        converted += 1

    print(f"  OK : {converted} convertis, {skipped} ignorés")


def main():
    print("=" * 55)
    print("  Conversion VisDrone → YOLO")
    print("=" * 55)

    for split in SPLITS:
        ann_dir = BASE / split / "annotations"
        if not ann_dir.exists():
            print(f"\n[SKIP] {split} — annotations introuvables")
            continue
        convert_split(split)

    print("\n" + "=" * 55)
    print("  Conversion terminée — tu peux lancer l'entraînement")
    print("=" * 55)


if __name__ == "__main__":
    main()