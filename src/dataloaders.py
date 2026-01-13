from torch.utils.data import DataLoader
from src.utils import seed_worker


def build_dataloader(dataset, config, is_train, generator=None):
    common_kwargs = dict(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers = True
    )

    if is_train:
        return DataLoader(
            dataset,
            shuffle=config['data']['shuffle'],
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=generator,
            **common_kwargs
        )

    return DataLoader(dataset, **common_kwargs)