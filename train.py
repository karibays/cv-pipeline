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
from src.logger import Logger
from src.early_stopping import EarlyStopping


def main():
    with open('configs/classification.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print('Device: ', device)

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

    if config['enable_determenism']:
        enable_determinism()
    fix_seeds(config['seed'])

    generator = torch.Generator()
    generator.manual_seed(config['seed'])
    transform_train = build_transform(config, 'train')
    transform_test = build_transform(config, 'val')
    trainset = ImageDataset(dataframe=train_df, transform=transform_train)
    testset = ImageDataset(dataframe=test_df, transform=transform_test)

    trainloader = build_dataloader(trainset, config, True, generator)
    testloader = build_dataloader(testset, config, False)
    net = build_model(config)
    net = net.to(device)
    optimizer = build_optimizer(config, net)
    criterion = build_loss(config)
    scheduler = build_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler(enabled=config['training']['use_amp'])
    logger = Logger(config)

    early_stopper = None
    if config['training']['early_stopping']:
        early_stopper = EarlyStopping(
            patience=config["training"]["patience"],
            mode="max",  # потому что balanced_accuracy должна расти
            checkpoint_path=os.path.join(config["paths"]["output_dir"], config["paths"]["checkpoint"]),
            logger=logger
        )

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
            use_amp=config["training"]["use_amp"]
        )
        logger.info(f"Loss: {train_metrics['loss']} | Acc: {train_metrics['accuracy']} | Balanced Acc: {train_metrics['balanced_accuracy']}")

        logger.info('Validation')
        _, _, val_metrics = validate_one_epoch(
            net=net,
            loader=testloader,
            criterion=criterion,
            device=device
        )
        logger.info(f"Loss: {val_metrics['loss']} | Acc: {val_metrics['accuracy']} | Balanced Acc: {val_metrics['balanced_accuracy']}")
    
        logger.log(train_metrics, step=epoch, prefix="train")
        logger.log(val_metrics, step=epoch, prefix="val")

        # EARLY STOPPING
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
    main()