import torch
import torch.nn as nn

class Checkpointer:
    def __init__(self, cfg):
        self.cfg = cfg

    def save_checkpoint(self, epoch: str, model: nn.Module, optimizer) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "metrics": metrics or {},
            # "config": config or {},
            # "timestamp": datetime.now().isoformat()
        }

        path = f"{self.cfg.save_dir}/{self.cfg.name}"
        torch.save(checkpoint, path)

    def load_checkpoint(self, weights_only: bool = True) -> nn.Module:
        path = f"{self.cfg.save_dir}/{self.cfg.name}"
        model = torch.load(path, weights_only=weights_only)
        return model