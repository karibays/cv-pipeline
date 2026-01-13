from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

METRIC_REGISTRY = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}


def build_metrics(config):
    metrics = {}

    for m in config["metrics"]:
        name = m["name"]
        params = m.get("params", {})

        if name not in METRIC_REGISTRY:
            raise ValueError(
                f"Metric '{name}' is not supported. "
                f"Available: {list(METRIC_REGISTRY.keys())}"
            )

        metrics[name] = (METRIC_REGISTRY[name], params)

    return metrics


def compute_metrics(y_true, y_pred, metrics_cfg):
    results = {}

    for name, (fn, params) in metrics_cfg.items():
        value = fn(y_true, y_pred, **params)

        # Все метрики приводим к %
        results[name] = round(float(value) * 100, 3)

    return results
