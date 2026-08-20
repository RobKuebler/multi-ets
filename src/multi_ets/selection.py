"""Model selection via rolling-origin cross-validation."""

import numpy as np

from multi_ets.holt import _loss_fn_from_key, _loss_key


def select_model(candidates, y, n_splits):
    """Pick the best of `candidates` by rolling-origin CV, each scored with its own loss/h.

    Split scheme: rolling-origin, expanding-window, non-overlapping, step =
    candidate's own h. Split i (earliest first) trains on
    y[:len(y) - (n_splits - i) * h] and scores against the next h points.

    Returns (best, scores): best is refit on the full series, scores is
    {candidate: mean_cv_loss} for every candidate.
    """
    y = np.asarray(y, dtype=float)
    scores = {}
    for candidate in candidates:
        loss_fn = _loss_fn_from_key(_loss_key(candidate.loss, candidate.delta, candidate.tau))
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
            losses.append(float(loss_fn(test, candidate.predict())))
        scores[candidate] = float(np.mean(losses))

    best = min(scores, key=scores.get)
    best.fit(y)
    return best, scores
