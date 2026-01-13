import torch.nn as nn


LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
}

def build_loss(config):
    name = config["loss"]["name"]
    params = config["loss"].get("params", {})

    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Loss '{name}' not supported. "
            f"Available: {list(LOSS_REGISTRY.keys())}"
        )

    return LOSS_REGISTRY[name](**params)