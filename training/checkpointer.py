import torch
import torch.nn as nn

class Checkpointer:
    def __init__(self, path):
        self.path = path

    def save_checkpoint(self, model: nn.Module, name: str) -> None:
        path = f"{self.path}/{name}"
        torch.save(model, path)

    def load_checkpoint(self, name: str, weights_only: bool = True) -> nn.Module:
        path = f"{self.path}/{name}"
        model = torch.load(path, weights_only=weights_only)
        return model