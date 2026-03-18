"""
Entraînement baseline YOLOv8s sur VisDrone2019-DET
Ce modèle sert de référence pour comparer DroneScan-YOLO

Résultats sauvegardés dans : runs/detect/baseline/
"""

from ultralytics import YOLO
from pathlib import Path

# Chemins
YAML    = "data/visdrone.yaml"
PROJET  = "runs/detect"
NOM     = "baseline"
POIDS   = "yolov8s.pt"

def main():
    print("=" * 55)
    print("  DroneScan — Entraînement Baseline YOLOv8s")
    print("=" * 55)

    # Charger le modèle pré-entraîné
    model = YOLO(POIDS)

    # Lancer l'entraînement
    results = model.train(
        data        = YAML,
        epochs      = 50,        # suffisant pour converger sur VisDrone
        imgsz       = 640,
        batch       = 16,        # optimal pour 16GB VRAM
        workers     = 0,         # évite WinError 1455 sur Windows
        device      = 0,         # RTX 4090
        project     = PROJET,
        name        = NOM,
        exist_ok    = True,
        pretrained  = True,
        optimizer   = "AdamW",
        lr0         = 0.001,
        lrf         = 0.01,
        momentum    = 0.937,
        weight_decay= 0.0005,
        warmup_epochs    = 3,
        close_mosaic     = 10,
        amp         = True,      # mixed precision → plus rapide
        plots       = True,
        save        = True,
        verbose     = True,
    )

    print("\n" + "=" * 55)
    print("  Entraînement terminé !")
    print(f"  Meilleurs poids : {PROJET}/{NOM}/weights/best.pt")
    print("=" * 55)

    # Afficher les métriques finales
    print("\n  Métriques finales (val) :")
    print(f"  mAP@50    : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP@50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"  Precision : {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"  Recall    : {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    main()