"""Centered binary logistic probes, validation selection and scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def _probe_tensors(probe: Mapping) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read and validate the saved weight, scalar bias and training mean."""
    try:
        weight = probe["weight"].detach().float().cpu()
        bias = probe["bias"].detach().float().cpu()
        mean = probe["training_mean"].detach().float().cpu()
    except (KeyError, AttributeError) as error:
        raise ValueError(
            "Probe requires weight, bias and training_mean tensors"
        ) from error
    if weight.ndim != 1 or weight.numel() == 0:
        raise ValueError("Probe weight must be a nonempty vector")
    if bias.ndim != 0 or mean.shape != weight.shape:
        raise ValueError("Probe requires a scalar bias and a mean matching its weight")
    if not all(torch.isfinite(tensor).all() for tensor in (weight, bias, mean)):
        raise ValueError("Probe tensors must be finite")
    return weight, bias, mean


def score_probe(probe: Mapping, features: torch.Tensor) -> torch.Tensor:
    """CPU float32 score (features - training_mean) @ weight + bias; positive is 1."""
    weight, bias, mean = _probe_tensors(probe)
    features = features.detach().float().cpu()
    if features.ndim not in (1, 2) or features.shape[-1] != weight.numel():
        raise ValueError("Feature width does not match the probe")
    scores = (features - mean) @ weight + bias
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite probe scores")
    return scores


def unit_probe_direction(probe: Mapping | torch.Tensor) -> torch.Tensor:
    """Positive-class unit direction, independent of centering and bias."""
    weight = probe if isinstance(probe, torch.Tensor) else probe["weight"]
    weight = weight.detach().float().cpu()
    if weight.ndim != 1 or weight.numel() == 0:
        raise ValueError("Probe weight must be a nonempty vector")
    norm = weight.norm()
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("Probe weight must be a finite nonzero vector")
    return weight / norm


def _binary_labels(labels: Sequence | np.ndarray, count: int) -> np.ndarray:
    """Require one 0/1 label for each example; sklearn also accepts other labels."""
    labels = np.asarray(labels)
    if labels.shape != (count,) or not np.isin(labels, [0, 1]).all() or count == 0:
        raise ValueError("Expected one binary 0/1 label per example")
    return labels.astype(np.int64)


def binary_probe_metrics(labels: Sequence | np.ndarray, scores: torch.Tensor) -> dict:
    """Strict zero-threshold accuracy, log loss, and AUROC (None for one class)."""
    scores = scores.detach().float().cpu()
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("Expected a finite vector of probe scores")
    labels = _binary_labels(labels, len(scores))
    return _metrics_from_scores(labels, scores)


def _metrics_from_scores(labels: np.ndarray, scores: torch.Tensor) -> dict:
    """Compute metrics after the caller has checked labels and scores."""
    score_values = scores.numpy()
    return {
        "accuracy": float(accuracy_score(labels, score_values > 0)),
        "log_loss": float(
            log_loss(labels, torch.sigmoid(scores).numpy(), labels=[0, 1])
        ),
        "auroc": (
            float(roc_auc_score(labels, score_values))
            if len(np.unique(labels)) == 2
            else None
        ),
    }


def fit_binary_probe(
    train_features: torch.Tensor,
    train_labels: Sequence | np.ndarray,
    validation_features: torch.Tensor,
    validation_labels: Sequence | np.ndarray,
    *,
    c_grid: Sequence[float],
    seed: int,
    max_iter: int = 2000,
) -> dict:
    """Fit an L2 logistic probe, choosing C by the lowest validation log loss.

    Break ties with the smaller C. Center both splits on the training mean, and
    keep test data out of fitting. Return weights, centering and split metrics
    in the checkpoint dictionary format.
    """
    train = train_features.detach().float().cpu().numpy()
    validation = validation_features.detach().float().cpu().numpy()
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("Expected train/validation matrices with equal width")
    y_train = _binary_labels(train_labels, len(train))
    y_validation = _binary_labels(validation_labels, len(validation))
    grid = tuple(float(c) for c in c_grid)
    if not grid or any(not np.isfinite(c) or c <= 0 for c in grid):
        raise ValueError("Regularization grid must contain positive finite C values")

    # Sklearn checks finite feature values and requires both training classes.
    # Check widths above because NumPy centering could otherwise broadcast them.
    training_mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered_train = train - training_mean
    centered_validation = validation - training_mean
    candidates = []
    for c in grid:
        # L2 is the default; omitting the deprecated penalty argument also works
        # with the older sklearn versions used by existing Colab notebooks.
        candidate = LogisticRegression(
            solver="lbfgs", C=c, max_iter=max_iter, random_state=seed
        )
        candidate.fit(centered_train, y_train)
        probability = candidate.predict_proba(centered_validation)[:, 1]
        validation_loss = log_loss(y_validation, probability, labels=[0, 1])
        candidates.append((validation_loss, c, candidate))
    _, selected_c, fitted = min(candidates, key=lambda item: (item[0], item[1]))
    probe = {
        "weight": torch.from_numpy(fitted.coef_[0].astype(np.float32, copy=True)),
        "bias": torch.tensor(float(fitted.intercept_[0]), dtype=torch.float32),
        "training_mean": torch.from_numpy(training_mean),
        "C": selected_c,
    }
    # Downstream J-Lens analysis needs a nonzero, normalizable probe direction.
    unit_probe_direction(probe["weight"])
    for name, features, labels in (
        ("train", centered_train, y_train),
        ("validation", centered_validation, y_validation),
    ):
        probe[name] = {
            "log_loss": float(
                log_loss(labels, fitted.predict_proba(features)[:, 1], labels=[0, 1])
            ),
            "accuracy": float(accuracy_score(labels, fitted.predict(features))),
        }
    return probe


@dataclass(frozen=True, slots=True)
class ProbeEvaluation:
    scores: torch.Tensor
    predictions: torch.Tensor
    gold_margins: torch.Tensor
    gold_probabilities: torch.Tensor
    correct: torch.Tensor
    metrics: dict


def evaluate_probe(
    probe: Mapping, features: torch.Tensor, labels: Sequence | np.ndarray
) -> ProbeEvaluation:
    """Apply a frozen binary probe with a fixed zero threshold and gold orientation."""
    scores = score_probe(probe, features)
    if scores.ndim != 1:
        raise ValueError("Probe evaluation requires a batch of feature vectors")
    labels = _binary_labels(labels, len(scores))
    gold_labels = torch.from_numpy(labels)
    predictions = scores > 0
    gold_margins = (2 * gold_labels - 1) * scores
    return ProbeEvaluation(
        scores=scores,
        predictions=predictions,
        gold_margins=gold_margins,
        gold_probabilities=torch.sigmoid(gold_margins),
        correct=predictions == gold_labels.bool(),
        metrics=_metrics_from_scores(labels, scores),
    )
