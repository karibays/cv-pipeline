import torch
import os


class EarlyStopping:
    def __init__(self, patience, mode="max", checkpoint_path=None, logger=None):
        self.patience = patience
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.logger = logger

        self.best_score = None
        self.bad_epochs = 0

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self._save(model)
            self._log(f"Initial best score: {score:.4f}")
            return False

        if self._is_improvement(score):
            self.best_score = score
            self.bad_epochs = 0
            self._save(model)
            self._log(f"New best score: {score:.4f} — checkpoint saved")
        else:
            self.bad_epochs += 1
            self._log(
                f"No improvement ({self.bad_epochs}/{self.patience}). "
                f"Best: {self.best_score:.4f}, Current: {score:.4f}"
            )

        if self.bad_epochs >= self.patience:
            self._log("Early stopping triggered")
            return True

        return False

    def _is_improvement(self, score):
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def _save(self, model):
        if self.checkpoint_path:
            torch.save(model.state_dict(), self.checkpoint_path)

    def _log(self, message):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)


def build_early_stopper(config, logger):
    early_stopper = None
    if config['training']['early_stopping']:
        early_stopper = EarlyStopping(
            patience=config["training"]["patience"],
            mode="max",  # потому что balanced_accuracy должна расти
            checkpoint_path=os.path.join(config["paths"]["output_dir"], config["paths"]["checkpoint"]),
            logger=logger
        )
    return early_stopper