"""
DroneScan-YOLO Native Training
================================
Uses dronescan_s.yaml (native P2 head, scale s forced) + SAL-NWD + RPA.
"""

import torch
import yaml
import sys
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils import RANK

sys.path.insert(0, str(Path(__file__).parent))
from models.modules.rpa_block import RPAModule
from dronescan_loss import DroneScanDetectionLoss


class DroneScanNativeTrainer(DetectionTrainer):

    def __init__(self, use_sal_nwd=True, use_rpa=True, **kwargs):
        self.use_sal_nwd = use_sal_nwd
        self.use_rpa     = use_rpa
        self.rpa_modules = []
        super().__init__(**kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Load YAML and force scale 's'
        with open("dronescan_s.yaml") as f:
            model_cfg = yaml.safe_load(f)
        model_cfg["scale"] = "s"

        model = DetectionModel(
            model_cfg,
            ch      = 3,
            nc      = self.data["nc"],
            verbose = verbose and RANK == -1,
        )

        # Load pretrained YOLOv8s weights (partial match)
        if weights and isinstance(weights, str) and Path(weights).exists():
            model.load(weights)
        else:
            # Load from yolov8s.pt pretrained
            try:
                from ultralytics.nn.tasks import attempt_load_one_weight
                ckpt = torch.load("yolov8s.pt", map_location="cpu")
                state = ckpt.get("model", ckpt)
                if hasattr(state, "state_dict"):
                    state = state.state_dict()
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(f"  [Weights] Loaded yolov8s.pt "
                      f"({len(state)-len(missing)}/{len(state)} layers matched)")
            except Exception as e:
                print(f"  [Weights] Starting from scratch: {e}")

        # SAL-NWD criterion
        if self.use_sal_nwd:
            orig_init = model.init_criterion
            def custom_criterion():
                try:
                    return DroneScanDetectionLoss(
                        de_parallel(model), lambda_nwd=0.5
                    )
                except Exception as e:
                    print(f"  [SAL-NWD] Fallback: {e}")
                    return orig_init()
            model.init_criterion = custom_criterion
            print("  [SAL-NWD] Criterion registered")

        # RPA hooks
        if self.use_rpa:
            self._inject_rpa(model)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  [DroneScan] {n_params:,} parameters | "
              f"P2+P3+P4+P5 heads | SAL-NWD={self.use_sal_nwd} | "
              f"RPA={self.use_rpa}")
        return model

    def _inject_rpa(self, model):
        device = next(model.parameters()).device
        # Scale 's': layer 2 = 32ch, layer 4 = 64ch (width=0.5 applied)
        for idx, ch in [(2, 32), (4, 64)]:
            rpa = RPAModule(
                in_channels=ch, out_channels=ch,
                warmup_epochs=10, update_interval=5, threshold=0.85,
            ).to(device)

            def make_hook(r, c):
                def hook(module, input, output):
                    if not isinstance(output, torch.Tensor): return output
                    if output.shape[1] != c: return output
                    try:
                        dt = output.dtype
                        return (r(output.detach().float()) +
                                output.float() -
                                output.float().detach()).to(dt)
                    except Exception:
                        return output
                return hook

            model.model[idx].register_forward_hook(make_hook(rpa, ch))
            self.rpa_modules.append(rpa)
            setattr(model, f"rpa_{idx}", rpa)

        print(f"  [RPA] Injected at layers 2 ({32}ch) and 4 ({64}ch)")


def rpa_callback(trainer):
    for rpa in trainer.rpa_modules:
        rpa.step_epoch()
    if trainer.rpa_modules and trainer.epoch % 10 == 0:
        sp = sum(
            m.get_stats()["rpa1_sparsity"] + m.get_stats()["rpa2_sparsity"]
            for m in trainer.rpa_modules
        ) / (2 * len(trainer.rpa_modules))
        print(f"\n  [RPA] Epoch {trainer.epoch} sparsity={sp:.2%}")


def train(run_name="dronescan_native", epochs=100, imgsz=1280, resume=False):
    print(f"\n{'='*60}")
    print(f"  DroneScan-YOLO Native — {run_name}")
    print(f"  {epochs} epochs | {imgsz}px | P2+P3+P4+P5 | SAL-NWD | RPA")
    print(f"{'='*60}")

    weights = "yolov8s.pt"
    if resume:
        last = Path(f"runs/detect/{run_name}/weights/last.pt")
        if last.exists():
            weights = str(last)

    args = dict(
        model           = weights,
        data            = "data/visdrone.yaml",
        epochs          = epochs,
        imgsz           = imgsz,
        batch           = 4,
        workers         = 0,
        device          = 0,
        project         = "runs/detect",
        name            = run_name,
        exist_ok        = True,
        pretrained      = True,
        optimizer       = "AdamW",
        lr0             = 0.001,
        lrf             = 0.01,
        weight_decay    = 0.0005,
        warmup_epochs   = 5,
        close_mosaic    = 15,
        amp             = True,
        plots           = True,
        copy_paste      = 0.3,
        copy_paste_mode = "flip",
        mosaic          = 1.0,
        mixup           = 0.0,
        scale           = 0.9,
        resume          = resume,
    )

    trainer = DroneScanNativeTrainer(
        use_sal_nwd = True,
        use_rpa     = True,
        overrides   = args,
    )
    trainer.add_callback("on_fit_epoch_end", rpa_callback)
    trainer.train()
    print(f"\n  Done -> runs/detect/{run_name}/")


if __name__ == "__main__":
    import sys
    resume = "--resume" in sys.argv
    train(run_name="dronescan_native", epochs=100, imgsz=1280, resume=resume)