import math
import re
from typing import Iterable, Sequence

import torch


def get_learning_rate(FLAGS, init=0.0, end=0.0):
    warmup_steps = max(int(FLAGS.lr_warmup_steps), 0)
    total_steps = max(int(FLAGS.total_steps), 1)
    peak = float(FLAGS.lr)
    init = float(init)
    end = float(end)

    def schedule(step: int) -> float:
        step = max(int(step), 0)
        if warmup_steps > 0 and step < warmup_steps:
            return init + (peak - init) * (step / warmup_steps)
        progress = min(max(step - warmup_steps, 0) / max(total_steps - warmup_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return float(end + (peak - end) * cosine)

    return schedule


def _should_exclude(name: str, exclusions: Iterable[str]) -> bool:
    for rule in exclusions:
        if re.search(rule, name) is not None:
            return True
    return False


def get_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float, exclusions: Sequence[str]):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if exclusions and _should_exclude(name, exclusions):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    if not param_groups:
        raise ValueError("No trainable parameters found")

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
    return optimizer
