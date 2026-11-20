import torch
import os
from typing import BinaryIO


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | BinaryIO):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["iteration"] = iteration
    torch.save(checkpoint, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]
    return iteration
