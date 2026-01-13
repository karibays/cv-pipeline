import torch
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from src.metrics import compute_metrics


def train_one_epoch(
    net,
    loader,
    criterion,
    device,
    optimizer,
    scheduler,
    scaler,
    metrics_cfg,
    use_amp
):
    net.train()

    running_loss = 0.0
    all_preds, all_labels = [], []
    lr = None

    with torch.enable_grad():
        for images, labels in tqdm(loader, total=len(loader)):
            images, labels = images.to(device), labels.to(device)

            enabled = use_amp and scaler is not None
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=enabled
            ):
                outputs = net(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_metrics(all_labels, all_preds, metrics_cfg)
    metrics["loss"] = round(running_loss / len(loader), 4)
    metrics["lr"] = lr

    if scheduler:
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

    return all_preds, all_labels, metrics

def validate_one_epoch(
    net,
    loader,
    criterion,
    device,
    metrics_cfg
):
    net.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, total=len(loader)):
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_metrics(all_labels, all_preds, metrics_cfg)
    metrics["loss"] = round(running_loss / len(loader), 4)

    return all_preds, all_labels, metrics