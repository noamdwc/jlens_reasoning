from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from jlens_reasoning.probing import (
    binary_probe_metrics,
    evaluate_probe,
    fit_binary_probe,
    load_probe_checkpoint,
    save_probe_checkpoint,
    score_probe,
    unit_probe_direction,
)


def test_fit_preserves_centering_and_validation_selection():
    # Asymmetric held-out values catch accidental centering on validation data.
    train = torch.tensor([[1.0, 3.0], [2.0, 1.0], [4.0, 4.0], [5.0, 2.0]])
    validation = torch.tensor([[8.0, 1.0], [-2.0, 3.0], [3.0, 1.0], [6.0, 4.0]])
    labels = np.array([0, 0, 1, 1])
    validation_labels = np.array([1, 0, 0, 1])
    grid = (0.01, 1.0, 100.0)
    fitted = fit_binary_probe(
        train, labels, validation, validation_labels, c_grid=grid, seed=1729
    )
    torch.testing.assert_close(fitted["training_mean"], torch.tensor([3.0, 2.5]))
    candidates = []
    for c in grid:
        reference = LogisticRegression(
            C=c, solver="lbfgs", max_iter=2000, random_state=1729
        ).fit(train.numpy() - np.array([3.0, 2.5], dtype=np.float32), labels)
        probability = reference.predict_proba(
            validation.numpy() - np.array([3.0, 2.5], dtype=np.float32)
        )[:, 1]
        candidates.append((log_loss(validation_labels, probability), c, reference))
    loss, expected_c, reference = min(candidates, key=lambda item: (item[0], item[1]))
    assert fitted["C"] == expected_c
    assert fitted["validation"]["log_loss"] == pytest.approx(loss)
    np.testing.assert_allclose(
        score_probe(fitted, validation).numpy(),
        reference.decision_function(validation.numpy() - [3.0, 2.5]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_scoring_uses_saved_training_mean_bias_and_strict_zero_threshold():
    probe = {
        "weight": torch.tensor([2.0, -1.0]),
        "bias": torch.tensor(0.5),
        "training_mean": torch.tensor([3.0, 4.0]),
    }
    scores = score_probe(probe, torch.tensor([[3.0, 4.5], [4.0, 3.0], [2.0, 5.0]]))
    torch.testing.assert_close(scores, torch.tensor([0.0, 3.5, -2.5]))
    evaluation = evaluate_probe(
        probe, torch.tensor([[3.0, 4.5], [4.0, 3.0], [2.0, 5.0]]), [0, 1, 0]
    )
    torch.testing.assert_close(evaluation.gold_margins, torch.tensor([0.0, 3.5, 2.5]))
    assert evaluation.predictions.tolist() == [False, True, False]
    metrics = binary_probe_metrics([0, 1, 0], scores)
    assert metrics["accuracy"] == 1
    assert metrics["auroc"] == 1
    torch.testing.assert_close(
        unit_probe_direction(probe), torch.tensor([2.0, -1.0]) / 5**0.5
    )


def test_direction_only_needs_the_weight():
    weight = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
    for probe in (weight, {"weight": weight}):
        direction = unit_probe_direction(probe)
        torch.testing.assert_close(direction, torch.tensor([0.6, 0.8]))
        assert not direction.requires_grad


def test_evaluation_handles_a_single_class():
    probe = {
        "weight": torch.tensor([1.0]),
        "bias": torch.tensor(0.0),
        "training_mean": torch.tensor([0.0]),
    }
    evaluation = evaluate_probe(probe, torch.tensor([[-2.0], [0.0]]), [0, 0])
    assert evaluation.metrics["auroc"] is None
    assert evaluation.metrics["accuracy"] == 1.0
    assert evaluation.correct.tolist() == [True, True]
    torch.testing.assert_close(evaluation.gold_margins, torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(
        evaluation.gold_probabilities, torch.tensor([0.8807971, 0.5])
    )


def test_validation_loss_ties_choose_the_smaller_c():
    fitted = fit_binary_probe(
        torch.tensor([[-1.0], [1.0]]),
        [0, 1],
        torch.zeros(2, 1),
        [0, 1],
        c_grid=(10.0, 0.01, 1.0),
        seed=1,
    )
    assert fitted["C"] == 0.01
    assert fitted["validation"]["log_loss"] == pytest.approx(np.log(2))


@pytest.mark.parametrize(
    "case",
    [
        "empty_grid",
        "nonpositive_c",
        "bad_labels",
        "single_training_class",
        "nonfinite",
        "nonfinite_validation",
        "width",
        "broadcastable_width",
    ],
)
def test_fitting_rejects_invalid_training_data(case):
    x = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    y = [0, 1]
    val = x.clone()
    grid = [1.0]
    if case == "empty_grid":
        grid = []
    if case == "nonpositive_c":
        grid = [0.0]
    if case == "bad_labels":
        y = [1, 2]
    if case == "single_training_class":
        y = [1, 1]
    if case == "nonfinite":
        x[0, 0] = float("nan")
    if case == "nonfinite_validation":
        val[0, 0] = float("nan")
    if case == "width":
        val = torch.ones(2, 3)
    if case == "broadcastable_width":
        val = torch.ones(2, 1)
    with pytest.raises(ValueError):
        fit_binary_probe(x, y, val, [0, 1], c_grid=grid, seed=1)


def test_checkpoint_roundtrip_and_validation(tmp_path):
    probe = {
        "weight": torch.tensor([2.0, -1.0]),
        "bias": torch.tensor(0.5),
        "training_mean": torch.tensor([3.0, 4.0]),
    }
    metadata = {
        "format_version": 2,
        "model_name": "test",
        "num_layers": 1,
        "hidden_dim": 2,
        "input_format": "chat_template_direct",
        "input_contract": {"tokenizer_sha256": "test"},
        "feature_contract": "test",
        "feature_position": "test",
    }
    path, sidecar = tmp_path / "probes.pt", tmp_path / "metadata.json"
    save_probe_checkpoint(path, {0: probe}, metadata=metadata, metadata_path=sidecar)
    loaded = load_probe_checkpoint(
        path, metadata_path=sidecar, expected_contract=metadata, model_name="test"
    )
    torch.testing.assert_close(
        score_probe(loaded["layers"][0], torch.tensor([4.0, 3.0])), torch.tensor(3.5)
    )
    with pytest.raises(ValueError, match="model"):
        load_probe_checkpoint(path, model_name="different")
    with pytest.raises(ValueError):
        load_probe_checkpoint(
            path, expected_contract={**metadata, "feature_position": "different"}
        )
    sidecar.write_text("{}")
    with pytest.raises(ValueError, match="metadata"):
        load_probe_checkpoint(path, metadata_path=sidecar)
    probe["weight"][0] = float("nan")
    with pytest.raises(ValueError):
        save_probe_checkpoint(path, {0: probe}, metadata=metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weight", torch.zeros(2)),
        ("bias", torch.tensor(float("nan"))),
        ("bias", torch.ones(2)),
        ("training_mean", torch.tensor([float("inf"), 0.0])),
        ("training_mean", torch.zeros(1)),
    ],
)
def test_checkpoint_rejects_invalid_parameters_on_save_and_load(tmp_path, field, value):
    probe = {
        "weight": torch.tensor([1.0, 2.0]),
        "bias": torch.tensor(0.0),
        "training_mean": torch.zeros(2),
        field: value,
    }
    metadata = {"format_version": 2, "num_layers": 1, "hidden_dim": 2}
    path = tmp_path / "probes.pt"
    with pytest.raises(ValueError):
        save_probe_checkpoint(path, {0: probe}, metadata=metadata)
    assert not path.exists()

    torch.save({"format_version": 2, "layers": {0: probe}, "metadata": metadata}, path)
    with pytest.raises(ValueError):
        load_probe_checkpoint(path)
