import torch.optim as optim


SCHEDULER_REGISTRY = {
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
    "step": optim.lr_scheduler.StepLR,
    "exp": optim.lr_scheduler.ExponentialLR,
}

def build_scheduler(config, optimizer):
    name = config["scheduler"]["name"]

    if name is None:
        return None

    if name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Scheduler '{name}' is not supported. "
            f"Available: {list(SCHEDULER_REGISTRY.keys())}"
        )

    params = config["scheduler"].get("params", {})

    scheduler_fn = SCHEDULER_REGISTRY[name]
    return scheduler_fn(optimizer, **params)