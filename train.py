import os
import glob
import wandb
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from src.engine import train_one_epoch, validate_one_epoch
from src.dataset import ImageDataset
from src.dataloaders import build_dataloader
from src.utils import fix_seeds, enable_determinism, seed_worker
from src.models import build_model
from src.optimizers import build_optimizer
from src.losses import build_loss
from src.schedulers import build_scheduler
from src.transforms import build_transform
from src.metrics import build_metrics
from src.logger import Logger
from src.early_stopping import build_early_stopper


def main(config):
    # --------- device ---------
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Device: ', device)

    # --------- fix random ---------
    if config['enable_determenism']:
        enable_determinism()
    fix_seeds(config['seed'])
    generator = torch.Generator()
    generator.manual_seed(config['seed'])

    # --------- read data ---------
    all_files = sorted(glob.glob(os.path.join(config['paths']['data_root'], '*/*')))
    classes = [x.split('\\')[-2] for x in all_files]
    df = pd.DataFrame(zip(all_files, classes), columns=['imname', 'class'])
    unique_classes = sorted(df["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}

    df['class'] = df['class'].map(class_to_idx)
    train_df, test_df = train_test_split(
        df, 
        test_size=config['data']['test_size'],
        stratify=df['class'],
        random_state=config['seed']
    )
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # --------- transforms ---------
    transform_train = build_transform(config, 'train')
    transform_test = build_transform(config, 'val')

    # --------- datasets ---------
    trainset = ImageDataset(dataframe=train_df, transform=transform_train)
    testset = ImageDataset(dataframe=test_df, transform=transform_test)

    # --------- dataloaders ---------
    trainloader = build_dataloader(trainset, config, True, generator)
    testloader = build_dataloader(testset, config, False)

    # --------- model ---------
    net = build_model(config)
    net = net.to(device)

    # --------- training utils ---------
    logger = Logger(config)
    optimizer = build_optimizer(config, net)
    criterion = build_loss(config)
    scheduler = build_scheduler(config, optimizer)
    metrics_cfg = build_metrics(config)
    early_stopper = build_early_stopper(config, logger)
    scaler = torch.amp.GradScaler(enabled=config['training']['use_amp'])

    # --------- training ---------
    for epoch in range(config["training"]["epochs"]):
        logger.info(f"Training epoch: {epoch + 1} / {config['training']['epochs']}")
        _, _, train_metrics = train_one_epoch(
            net=net,
            loader=trainloader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metrics_cfg=metrics_cfg,
            use_amp=config["training"]["use_amp"]
        )
        logger.info(f"Loss: {train_metrics['loss']} | Acc: {train_metrics['accuracy']} | Balanced Acc: {train_metrics['balanced_accuracy']}")

        # --------- validation ---------
        logger.info('Validation')
        _, _, val_metrics = validate_one_epoch(
            net=net,
            loader=testloader,
            criterion=criterion,
            device=device,
            metrics_cfg=metrics_cfg
        )
        logger.info(f"Loss: {val_metrics['loss']} | Acc: {val_metrics['accuracy']} | Balanced Acc: {val_metrics['balanced_accuracy']}")
    
        logger.log(train_metrics, step=epoch, prefix="train")
        logger.log(val_metrics, step=epoch, prefix="val")

        # --------- early stopping ---------
        if early_stopper is not None:
            should_stop = early_stopper.step(
                score=val_metrics["balanced_accuracy"],
                model=net
            )

            if should_stop:
                logger.info("Early stopping triggered")
                break
        logger.info('----------------------')
    logger.close()


if __name__ == '__main__':
    # --------- read config ---------
    with open('configs/classification.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    main(config)