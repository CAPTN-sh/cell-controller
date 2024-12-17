import math
import torch


def custom_decay_lr(
    type: str,
    optimizer: torch.optim.Optimizer,
    init_lr: float,
    epoch: int,
    num_epochs: int = 10,
) -> None:
    if type not in ["halved", "cosine"]:
        print("Type must be 'halved' or 'cosine', set to 'halved' by default")
        type = "halved"
    epoch = max(1, epoch)
    if type == "halved":
        lr = init_lr * (0.5 ** (epoch - 1))
    elif type == "cosine":
        lr = init_lr / 2 * (1 + math.cos(epoch / num_epochs * math.pi))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
