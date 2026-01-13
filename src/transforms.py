import torch
import torchvision.transforms.v2 as transforms


TRANSFORM_REGISTRY = {
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "Resize": transforms.Resize,
    "TrivialAugmentWide": transforms.TrivialAugmentWide,
    "PILToTensor": transforms.PILToTensor,
    "Normalize": transforms.Normalize,
    "RandomErasing": transforms.RandomErasing,
    "ToDtype": transforms.ToDtype,
}

def build_transform(config, split):
    transform_list = []

    for x in config['transforms'][split]:
        name = x["name"]
        params = dict(x.get("params", {}))

        if name not in TRANSFORM_REGISTRY:
            raise ValueError(
                f"{name} not available. "
                f"Available: {list(TRANSFORM_REGISTRY.keys())}"
            )

        if name == "ToDtype" and "dtype" in params:
            DTYPE_MAP = {
                "float16": torch.float16,
                "float32": torch.float32
            }
            params["dtype"] = DTYPE_MAP[params["dtype"]]

        transform_fn = TRANSFORM_REGISTRY[name]
        transform_list.append(transform_fn(**params))

    return transforms.Compose(transform_list)
        