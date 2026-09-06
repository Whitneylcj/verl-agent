import numpy as np
import pytest

from recipe.exact.alfworld_gradient_audit import audit_gradient_samples


def test_gradient_variance_uses_score_vectors_and_masks():
    scores = np.array([[[1.0, 2.0], [99.0, 99.0]], [[-1.0, -2.0], [99.0, 99.0]]])
    mask = np.array([[1, 0], [1, 0]])
    credits = {key: np.array([[2.0, 200.0], [2.0, 200.0]]) for key in ("graph", "temporal", "prefix_baseline")}
    result = audit_gradient_samples(scores, mask, credits, [2, 2])
    assert result["estimators"]["graph"]["variance_trace"] == 40
    assert result["estimators"]["graph"]["paired_mean_difference_norm"] == 0
    assert result["graph_prefix_sample_difference_max"] == 0
    # Credits are constant across samples; their variance would incorrectly be 0.
    with pytest.raises(ValueError, match="finite"):
        audit_gradient_samples(scores * np.nan, mask, credits, [2, 2])
