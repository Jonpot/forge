import json

import pytest

from starforge.core.previews import build_preview


def test_value_preview_roundtrips_strict_json():
    preview = build_preview({"values": [1, 2.5, "x"], "ok": True, "none": None})
    assert preview["kind"] == "value"
    json.loads(json.dumps(preview, allow_nan=False))  # strict-JSON safe
    assert preview["value"]["values"] == [1, 2.5, "x"]


def test_nan_and_inf_are_stringified_not_poisonous():
    preview = build_preview({"bad": float("nan"), "worse": float("inf")})
    encoded = json.dumps(preview, allow_nan=False)  # would raise if NaN survived
    assert "nan" in encoded
    assert preview["value"]["bad"] == "nan"


def test_large_containers_are_capped():
    preview = build_preview({"big": list(range(500))})
    assert len(preview["value"]["big"]) <= 51  # 50 items + truncation marker
    assert any("more" in str(v) for v in preview["value"]["big"][-1:])


def test_unserializable_falls_back_to_repr():
    class Strange:
        def __repr__(self):
            return "<Strange thing>"

    preview = build_preview(Strange())
    assert preview == {"kind": "text", "text": "<Strange thing>"}


def test_dataframe_preview_is_cropped_table():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({f"c{i}": range(20) for i in range(15)})
    preview = build_preview(df)
    assert preview["kind"] == "table"
    assert preview["shape"] == [20, 15]
    assert len(preview["columns"]) == 10
    assert preview["columns_truncated"] is True
    assert len(preview["rows"]) == 8
    json.loads(json.dumps(preview, allow_nan=False))


def test_ndarray_preview_has_corner():
    np = pytest.importorskip("numpy")
    preview = build_preview(np.arange(100).reshape(10, 10))
    assert preview["kind"] == "array"
    assert preview["shape"] == [10, 10]
    assert len(preview["corner"]) == 8
    assert preview["corner"][0][:3] == [0, 1, 2]
