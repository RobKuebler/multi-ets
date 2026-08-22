"""Model selection via rolling-origin cross-validation."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from multi_ets.holt import Holt, _loss_key, _loss_value_np


def select_model(
    candidates: Iterable[Holt], y: Any, n_splits: int
) -> tuple[Holt, dict[Holt, float]]:
    """Pick the best of `candidates` by rolling-origin CV, each scored with its own loss/h.

    Split scheme: rolling-origin, expanding-window, non-overlapping, step =
    candidate's own h. Split i (earliest first) trains on
    y[:len(y) - (n_splits - i) * h] and scores against the next h points.

    Returns (best, scores): best is refit on the full series, scores is
    {candidate: mean_cv_loss} for every candidate.
    """
    y = np.asarray(y, dtype=float)
    scores: dict[Holt, float] = {}
    for candidate in candidates:
        loss_key = _loss_key(candidate.loss, candidate.delta, candidate.tau)
        h = candidate.h
        if len(y) < n_splits * h:
            raise ValueError(
                f"Series too short for n_splits={n_splits} with h={h}: "
                f"need at least {n_splits * h} points, got {len(y)}"
            )
        losses = []
        for split in range(n_splits):
            cutoff = len(y) - (n_splits - split) * h
            train, test = y[:cutoff], y[cutoff : cutoff + h]
            candidate.fit(train)
            pred = candidate.predict()
            loss = (
                loss_key(test, pred) if callable(loss_key) else _loss_value_np(loss_key, test, pred)
            )
            losses.append(float(loss))
        scores[candidate] = float(np.mean(losses))

    best = min(scores, key=lambda c: scores[c])
    best.fit(y)
    return best, scores
