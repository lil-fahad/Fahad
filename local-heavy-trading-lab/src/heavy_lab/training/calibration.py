from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = np.asarray(logits, dtype=float) / float(temperature)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, *, temperature: float = 1.0) -> float:
    probs = softmax(logits, temperature=temperature)
    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(probs):
        raise ValueError("labels and logits length mismatch")
    chosen = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.geomspace(0.25, 4.0, num=65)
    losses = [negative_log_likelihood(logits, labels, temperature=float(t)) for t in candidates]
    return float(candidates[int(np.argmin(losses))])
