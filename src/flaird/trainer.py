import numpy as np
from transformers import Trainer

from flaird.scripts.evaluator import evaluate_all


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(eval_pred) -> dict[str, float]:

    logits = eval_pred.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    labels = eval_pred.label_ids
    if isinstance(labels, (tuple, list)):
        labels = labels[0]

    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probs = sigmoid(logits)
    results = evaluate_all(labels, probs)
    results = {k: v for k, v in results.items() if isinstance(v, float)}

    return results


class FlairdTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
