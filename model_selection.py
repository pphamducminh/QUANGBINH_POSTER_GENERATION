"""Find completed LoRA exports without importing GPU dependencies."""

from dataclasses import dataclass
from pathlib import Path
import gc

WEIGHTS = "pytorch_lora_weights.safetensors"
REMOTE_MODEL = "kTon/quangbinh-flux-lora"


@dataclass(frozen=True)
class ModelSelection:
    source: str
    label: str
    revision: tuple = ()


def local_models(root):
    root = Path(root).resolve()
    # Only final exports: never silently select an intermediate checkpoint.
    files = [root / "dataset_lora_flux_nf4" / WEIGHTS]
    files += sorted(root.glob(f"*_lora_flux_nf4/{WEIGHTS}"))
    files += [root / WEIGHTS]
    models = {}
    for path in files:
        if path.is_file() and path.stat().st_size:
            models.setdefault(str(path.parent), path)
    return models


def resolve_model(root, choice="auto"):
    root = Path(root).resolve()
    if choice == "remote":
        return ModelSelection(REMOTE_MODEL, "LoRA từ Hugging Face · Quảng Bình")
    models = local_models(root)
    if choice == "auto":
        if not models:
            return ModelSelection(REMOTE_MODEL, "LoRA từ Hugging Face · Quảng Bình (chưa có model local)")
        path = next(iter(models.values()))
    else:
        path = Path(choice).resolve() / WEIGHTS
        if str(path.parent) not in models:
            raise FileNotFoundError("Model đã chọn không còn tồn tại hoặc chưa có trọng số hoàn chỉnh.")
    stat = path.stat()
    label = path.parent.name if path.parent != root else "Model ở thư mục gốc"
    return ModelSelection(str(path.parent), f"Model đã train · {label}",
                          (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino))


class PipelineCache:
    """Keep one pipeline; release the old GPU model before loading another."""

    def __init__(self, loader, release=None):
        self.loader = loader
        self.release = release or (lambda: None)
        self.pipeline = None
        self.selection = None

    def get(self, selection):
        if self.pipeline is not None and self.selection == selection:
            return self.pipeline
        if self.pipeline is not None:
            self.pipeline = None
            self.selection = None
            gc.collect()
            self.release()
        # On failure, keep the cache empty; never return the previous model.
        self.pipeline = self.loader(selection.source)
        self.selection = selection
        return self.pipeline
