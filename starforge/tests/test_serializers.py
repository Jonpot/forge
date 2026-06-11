import pytest

from starforge.core import serializers
from starforge.core.serializers import EphemeralValueError, load_value, save_value


def test_json_roundtrip(tmp_path):
    value = {"values": [1, 2, 3], "label": "x", "nested": {"ok": True}}
    entry = save_value(value, tmp_path, "output")
    assert entry["serializer"] == "json"
    assert load_value(tmp_path, entry) == value


def test_unserializable_is_ephemeral_when_pickle_disabled(tmp_path):
    entry = save_value(object(), tmp_path, "output", pickle_enabled=False)
    assert entry["serializer"] == serializers.EPHEMERAL
    assert entry["file"] is None
    with pytest.raises(EphemeralValueError):
        load_value(tmp_path, entry)


def test_pickle_tier_is_opt_in(tmp_path):
    class Custom:
        def __init__(self, x):
            self.x = x

    # Module-level pickle needs an importable class; use a stdlib type that
    # json rejects instead.
    value = {1, 2, 3}
    entry = save_value(value, tmp_path, "output", pickle_enabled=True)
    assert entry["serializer"] == "pickle"
    assert load_value(tmp_path, entry) == value


def test_dataframe_roundtrip(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    entry = save_value(df, tmp_path, "output")
    assert entry["serializer"] == "parquet"
    assert entry["meta"]["shape"] == [2, 2]
    pd.testing.assert_frame_equal(load_value(tmp_path, entry), df)


def test_series_roundtrip(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    series = pd.Series([1.5, 2.5], name="scores")
    entry = save_value(series, tmp_path, "output")
    assert entry["serializer"] == "parquet"
    restored = load_value(tmp_path, entry)
    pd.testing.assert_series_equal(restored, series)


def test_ndarray_roundtrip(tmp_path):
    np = pytest.importorskip("numpy")
    arr = np.arange(12).reshape(3, 4)
    entry = save_value(arr, tmp_path, "output")
    assert entry["serializer"] == "npy"
    assert entry["meta"]["shape"] == [3, 4]
    assert (load_value(tmp_path, entry) == arr).all()
