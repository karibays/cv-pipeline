import wandb


class Logger:
    def __init__(self, config: dict):
        self.enabled = config.get("wandb", {}).get("enabled", False)

        if self.enabled:
            wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"]["run_name"],
                config=config
            )

    def log(self, metrics: dict, step: int, prefix: str):
        if not self.enabled:
            return

        wandb.log(
            {f"{prefix}/{k}": v for k, v in metrics.items()},
            step=step
        )

    def info(self, message: str):
        """Вывод обычных сообщений (замена print)"""
        print(message)

    def close(self):
        if self.enabled:
            wandb.finish()