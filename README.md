# DroneScan-YOLO

**Efficient Real-Time Aerial Object Detection via Adaptive Pruning and Wasserstein-Guided Loss**

## Quick Setup (Plug & Play)

### 1. Clone and setup environment

```bash
git clone https://github.com/TON_USERNAME/dronescan-yolo.git
cd dronescan-yolo
conda create -n dronescan python=3.10 -y
conda activate dronescan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics==8.3.0 numpy pandas matplotlib seaborn scipy scikit-learn opencv-python tqdm pyyaml tensorboard
```

### 2. Download and prepare VisDrone dataset

```bash
python data/convert_visdrone.py
```

VisDrone will be downloaded automatically on first training run.

### 3. Verify setup

```bash
python check.py
```

### 4. Train baseline

```bash
python train_baseline.py
```

### 5. Run ablation study

```bash
# Single run
python run_ablations.py dronescan_full

# All runs
python run_ablations.py
```

### 6. Resume interrupted run

```bash
python run_ablations.py --resume dronescan_full
```

### 7. Skip completed runs

```bash
python run_ablations.py --skip abl_sal_nwd,abl_msfd
```

## Project Structure

```
dronescan-yolo/
├── models/
│   ├── modules/
│   │   ├── rpa_block.py      # RPA-Block: redundancy-pruned attention
│   │   ├── msfd.py           # MSFD: multi-scale feature distillation
│   │   └── sal_nwd.py        # SAL-NWD: size-adaptive Wasserstein loss
│   └── dronescan.py          # Full architecture assembly
├── data/
│   ├── visdrone.yaml         # Dataset config
│   └── convert_visdrone.py   # Annotation converter
├── dronescan_loss.py         # Deep SAL-NWD loss integration
├── dronescan_model.py        # Deep model integration
├── run_ablations.py          # Master ablation script
├── train_baseline.py         # Baseline training
├── experiments/
│   └── sensitivity_rpa.py   # RPA sensitivity analysis
├── results/                  # CSV results (auto-generated)
└── check.py                  # Environment verification
```

## Architecture

DroneScan-YOLO extends YOLOv8s with three novel modules:

| Module | Description | Gain |
|--------|-------------|------|
| RPA-Block | Redundancy-pruned attention via cosine similarity | Efficiency |
| MSFD | P2 detection head for sub-32px objects | Recall |
| SAL-NWD | Size-adaptive Wasserstein loss | mAP@50-95 |

## Results

| Model | mAP@50 | mAP@50-95 | FPS | Params |
|-------|--------|-----------|-----|--------|
| YOLOv8s baseline | 0.387 | 0.233 | 71.0 | 9.83M |
| DroneScan-YOLO | TBD | TBD | TBD | +0.40M |

## Hardware

Tested on: NVIDIA RTX 4090 16GB, Windows 11, CUDA 13.1

## Citation

```bibtex
@article{dronescan2026,
  title={DroneScan-YOLO: Efficient Real-Time Aerial Object Detection
         via Adaptive Pruning and Wasserstein-Guided Loss},
  author={[Yann Bellec]},
  journal={SSNR},
  year={2026}
}
```
