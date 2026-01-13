import torch


OPTIMIZER_REGISTRY = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD
}


def build_optimizer(config, net):
    optimizer_name = config['optimizer']['name']
    optimizer_params = {
        k: float(v) if k == "lr" else v
        for k, v in config["optimizer"]["params"].items()
    }

    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Optimizer '{optimizer_name}' is not supported. "
            f"Available models: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    
    optimizer_fn = OPTIMIZER_REGISTRY[optimizer_name]
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())

    optimizer = optimizer_fn(
        trainable_params,
        **optimizer_params
    )

    return optimizer