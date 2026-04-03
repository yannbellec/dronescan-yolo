"""
Téléchargement et préparation du dataset VisDrone2019-DET
À placer dans S:\YOLO-DS\data\ et exécuter depuis cet emplacement
"""

import os
import zipfile
import urllib.request
from pathlib import Path

# Dossier de destination
BASE_DIR = Path(__file__).parent / "VisDrone"
BASE_DIR.mkdir(exist_ok=True)

# Liens officiels VisDrone2019-DET (GitHub officiel)
URLS = {
    "train": "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-train.zip",
    "val":   "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-val.zip",
    "test":  "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-test-dev.zip",
}

def download_with_progress(url: str, dest: Path):
    """Télécharge un fichier avec affichage de la progression"""
    filename = url.split("/")[-1]
    dest_file = dest / filename

    if dest_file.exists():
        print(f"  [SKIP] {filename} déjà téléchargé")
        return dest_file

    print(f"  [DL] {filename}...")

    def progress(count, block_size, total_size):
        pct = min(int(count * block_size * 100 / total_size), 100)
        print(f"\r  Progression : {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest_file, reporthook=progress)
    print()
    return dest_file


def extract(zip_path: Path, dest: Path):
    """Extrait une archive zip"""
    print(f"  [UNZIP] {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"  [OK] Extrait dans {dest}")


def convert_visdrone_to_yolo(split: str):
    """
    Convertit les annotations VisDrone au format YOLO (xywh normalisé).
    VisDrone format : x_min, y_min, width, height, score, category, truncation, occlusion
    YOLO format     : class x_center y_center width height (normalisé 0-1)
    """
    # Catégories VisDrone (0=ignored, on garde 1-10)
    # On remappe 1-10 → 0-9 pour YOLO
    VALID_CLASSES = set(range(1, 11))

    img_dir = BASE_DIR / f"VisDrone2019-DET-{split}" / "images"
    ann_dir = BASE_DIR / f"VisDrone2019-DET-{split}" / "annotations"
    lbl_dir = BASE_DIR / f"VisDrone2019-DET-{split}" / "labels"
    lbl_dir.mkdir(exist_ok=True)

    if not ann_dir.exists():
        print(f"  [WARN] Annotations introuvables pour {split}, skip conversion")
        return

    ann_files = list(ann_dir.glob("*.txt"))
    print(f"  [CONVERT] {len(ann_files)} fichiers annotations → YOLO ({split})")

    for ann_file in ann_files:
        img_file = img_dir / ann_file.with_suffix(".jpg").name
        if not img_file.exists():
            continue

        # Lire dimensions image
        import cv2
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]

        yolo_lines = []
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                x_min, y_min, bw, bh = map(int, parts[:4])
                category = int(parts[5])

                if category not in VALID_CLASSES:
                    continue

                # Conversion YOLO
                x_center = (x_min + bw / 2) / w
                y_center = (y_min + bh / 2) / h
                bw_norm  = bw / w
                bh_norm  = bh / h
                cls      = category - 1  # remappe 1-10 → 0-9

                yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")

        lbl_out = lbl_dir / ann_file.name
        with open(lbl_out, "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"  [OK] Conversion {split} terminée")


def create_yaml():
    """Crée le fichier de config dataset pour Ultralytics"""
    yaml_content = f"""# VisDrone2019-DET — config Ultralytics
path: {BASE_DIR.resolve()}

train: VisDrone2019-DET-train/images
val:   VisDrone2019-DET-val/images
test:  VisDrone2019-DET-test-dev/images

nc: 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    yaml_path = Path(__file__).parent / "visdrone.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  [OK] visdrone.yaml créé → {yaml_path}")


def main():
    print("=" * 50)
    print("  DroneScan — Téléchargement VisDrone2019-DET")
    print("=" * 50)

    for split, url in URLS.items():
        print(f"\n[{split.upper()}]")
        zip_file = download_with_progress(url, BASE_DIR)
        extract(zip_file, BASE_DIR)
        convert_visdrone_to_yolo(split)

    print("\n[YAML]")
    create_yaml()

    print("\n" + "=" * 50)
    print("  Tout est prêt. Lance train_baseline.py")
    print("=" * 50)


if __name__ == "__main__":
    main()