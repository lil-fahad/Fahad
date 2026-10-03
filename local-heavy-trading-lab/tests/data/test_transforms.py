from __future__ import annotations

import json

import pandas as pd
import pytest


def test_standard_transform_is_fit_on_train_only_and_persistable(tmp_path):
    from heavy_lab.data.transforms import FoldTransformer

    train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "feat_b": [10.0, 20.0, 30.0]})
    validation = pd.DataFrame({"feat_a": [1000.0], "feat_b": [-1000.0]})

    transformer = FoldTransformer(columns=["feat_a", "feat_b"], normalization="standard").fit(train)
    before = transformer.to_dict()
    transformed = transformer.transform(validation)

    assert transformer.to_dict() == before
    assert before["center"]["feat_a"] == pytest.approx(2.0)
    assert before["center"]["feat_b"] == pytest.approx(20.0)
    assert transformed.loc[0, "feat_a"] > 100.0

    path = tmp_path / "transform.json"
    transformer.save(path)
    assert json.loads(path.read_text(encoding="utf-8")) == before


def test_none_transform_preserves_timesfm_raw_scale():
    from heavy_lab.data.transforms import FoldTransformer

    frame = pd.DataFrame({"close": [500.25, 501.75, 499.5]})
    transformer = FoldTransformer(columns=["close"], normalization="none").fit(frame)

    result = transformer.transform(frame)

    pd.testing.assert_frame_equal(result, frame)
    assert transformer.to_dict()["normalization"] == "none"
    assert transformer.to_dict()["center"] == {}
    assert transformer.to_dict()["scale"] == {}
