from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymort.lifetables as lt

# ----------------------------
# Core conversions
# ----------------------------


def test_m_to_q_bounds_and_inverse_roundtrip():
    m = np.array([[1e-8, 0.02], [0.05, 0.1]], dtype=float)
    q = lt.m_to_q(m)
    assert q.shape == m.shape
    assert np.all((q > 0) & (q < 1))
    m_back = lt.q_to_m(q)
    # roundtrip should be close for typical magnitudes
    assert np.allclose(m_back, m, rtol=1e-3, atol=1e-8)


def test_q_to_m_clips_extremes():
    q = np.array([1e-300, 1 - 1e-300], dtype=float)
    m = lt.q_to_m(q)
    assert np.isfinite(m).all()
    assert (m > 0).all()


# ----------------------------
# Survival and validators
# ----------------------------


def test_survival_from_q_monotonic_and_shapes():
    q = np.array([[0.1, 0.2, 0.05]], dtype=float)
    S = lt.survival_from_q(q)
    assert S.shape == q.shape
    lt.validate_survival_monotonic(S)
    assert np.all(np.diff(S, axis=-1) <= 1e-12)
    assert np.all((S >= 0) & (S <= 1))


def test_survival_from_q_rejects_bad_q():
    with pytest.raises(ValueError):
        lt.survival_from_q(np.array([0.0, 0.5], dtype=float))
    with pytest.raises(ValueError):
        lt.survival_from_q(np.array([0.5, 1.0], dtype=float))
    with pytest.raises(ValueError):
        lt.survival_from_q(np.array([0.5, np.inf], dtype=float))


def test_validate_survival_monotonic_raises_if_increasing():
    S = np.array([0.9, 0.91, 0.89], dtype=float)
    with pytest.raises(AssertionError):
        lt.validate_survival_monotonic(S)


# ----------------------------
# survival_paths_from_q_paths
# ----------------------------


def test_survival_paths_from_q_paths_happy_path_and_shape_error():
    q_paths = np.array(
        [
            [[0.1, 0.2], [0.05, 0.05]],
            [[0.15, 0.1], [0.06, 0.04]],
        ],
        dtype=float,
    )  # (N=2, A=2, H=2)
    S_paths = lt.survival_paths_from_q_paths(q_paths)
    assert S_paths.shape == q_paths.shape
    assert np.isfinite(S_paths).all()
    # monotonic over time for each (n,a)
    assert np.all(np.diff(S_paths, axis=-1) <= 1e-12)

    with pytest.raises(ValueError):
        lt.survival_paths_from_q_paths(np.array([0.1, 0.2], dtype=float))


# ----------------------------
# Excel loader internals
# ----------------------------


def test__norm_normalizes_tokens():
    assert lt._norm(" Year ") == "year"
    assert lt._norm("Age\n") == "age"
    assert lt._norm("To_tal") == "total"
    assert lt._norm("Male-Female") == "malefemale"


def test__find_header_and_map_finds_header_row():
    # sheet has junk rows then header
    df = pd.DataFrame(
        [
            ["some", "junk", None],
            ["Age", "Year", "Total"],
            [60, 2000, 0.01],
        ]
    )
    hrow, cmap = lt._find_header_and_map(df)
    assert hrow == 1
    assert cmap["Age"] == 0
    assert cmap["Year"] == 1
    assert cmap["Total"] == 2


def test__find_header_and_map_returns_none_if_not_found():
    df = pd.DataFrame([["nope", "still", "nope"], [1, 2, 3]])
    hrow, cmap = lt._find_header_and_map(df)
    assert hrow is None
    assert cmap == {}


def test__read_table_with_header_parses_and_drops_open_age():
    sheet = pd.DataFrame(
        [
            ["Age", "Year", "Total"],
            ["110+", 2000, 0.5],
            [60, 2000, "0.01"],
            [61, 2000, "0.02"],
        ]
    )
    hrow, cmap = 0, {"Age": 0, "Year": 1, "Total": 2}
    table = lt._read_table_with_header(sheet, hrow, cmap)
    assert {"Age", "Year", "Total"} <= set(table.columns)
    assert (table["Age"] != "110+").all()
    assert table["Age"].dtype.kind in "iu"
    assert table["Year"].dtype.kind in "iu"
    assert table["Total"].dtype.kind == "f"


# ----------------------------
# load_m_from_excel (with monkeypatch)
# ----------------------------


def _fake_excel_dict_ok_with_gaps():
    """
    Build a dict of sheets as read_excel(sheet_name=None, header=None) would return.
    Includes:
      - a valid header row
      - a '110+' row to be dropped
      - a missing cell to trigger imputation
    """
    rows = [
        ["junk", None, None],
        ["Age", "Year", "Total"],
        ["110+", 2000, 0.5],
        [60, 2000, 0.01],
        [60, 2001, np.nan],  # gap to impute
        [61, 2000, 0.02],
        [61, 2001, 0.021],
    ]
    return {"Sheet1": pd.DataFrame(rows)}


def test_load_m_from_excel_happy_path_imputation_drop_years(monkeypatch, tmp_path):
    def fake_read_excel(*args, **kwargs):
        return _fake_excel_dict_ok_with_gaps()

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    out = lt.load_m_from_excel(
        path=str(tmp_path / "dummy.xlsx"),
        sex="Total",
        age_min=60,
        age_max=61,
        year_min=2000,
        year_max=2001,
        drop_years=[2000],
    )
    ages, years, m = out["m"]
    assert np.array_equal(ages, np.array([60, 61]))
    assert np.array_equal(years, np.array([2001]))
    assert m.shape == (2, 1)
    assert np.isfinite(m).all()
    assert (m > 0).all()


def test_load_m_from_excel_raises_if_no_valid_sheet(monkeypatch, tmp_path):
    def fake_read_excel(*args, **kwargs):
        return {"S": pd.DataFrame([["no", "header"], [1, 2]])}

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    with pytest.raises(ValueError):
        lt.load_m_from_excel(path=str(tmp_path / "x.xlsx"))


def test_load_m_from_excel_raises_if_filtered_empty(monkeypatch, tmp_path):
    def fake_read_excel(*args, **kwargs):
        return _fake_excel_dict_ok_with_gaps()

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    # Filter outside ranges so df becomes empty
    with pytest.raises(ValueError):
        lt.load_m_from_excel(
            path=str(tmp_path / "x.xlsx"),
            age_min=90,
            age_max=95,
        )


def test_load_m_from_excel_fallback_rate_col_when_requested_missing(monkeypatch, tmp_path):
    # Provide sheet with only Male column but request Female -> should fallback
    rows = [
        ["Age", "Year", "Male"],
        [60, 2000, 0.01],
        [61, 2000, 0.02],
    ]

    def fake_read_excel(*args, **kwargs):
        return {"S": pd.DataFrame(rows)}

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    out = lt.load_m_from_excel(path=str(tmp_path / "x.xlsx"), sex="Female", age_min=60, age_max=61)
    _ages, _years, m = out["m"]
    assert m.shape == (2, 1)
    assert np.isfinite(m).all()
    assert (m > 0).all()


def test_validate_q_raises_on_bounds_and_nonfinite():
    import numpy as np
    import pytest

    import pymort.lifetables as lt

    with pytest.raises(ValueError):
        lt.validate_q(np.array([0.0, 0.2], dtype=float))
    with pytest.raises(ValueError):
        lt.validate_q(np.array([0.2, 1.0], dtype=float))
    with pytest.raises(ValueError):
        lt.validate_q(np.array([0.2, np.nan], dtype=float))
    with pytest.raises(ValueError):
        lt.validate_q(np.array([0.2, np.inf], dtype=float))


def test_cohort_survival_from_q_paths_shape_and_nan_tail_and_clip():
    import numpy as np
    import pytest

    import pymort.lifetables as lt

    # wrong shape
    with pytest.raises(ValueError):
        lt.cohort_survival_from_q_paths(np.zeros((2, 3), dtype=float))

    # small valid case where H > A-a for some a => NaN tail expected
    # N=1, A=2, H=3: for a=1, only K=min(3,1)=1 reachable, rest NaN
    q = np.array([[[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]], dtype=float)
    S = lt.cohort_survival_from_q_paths(q)
    assert S.shape == q.shape

    # a=0: K=min(3,2)=2 -> first 2 finite
    assert np.isfinite(S[0, 0, 0:2]).all()
    # a=1: K=min(3,1)=1 -> only first finite, tail NaN
    assert np.isfinite(S[0, 1, 0])
    assert np.isnan(S[0, 1, 1:]).all()

    # clipped lower bound
    assert np.nanmin(S) >= 1e-12
    assert np.nanmax(S) <= 1.0


def test_survival_paths_from_q_paths_clips_and_shape_validation():
    import numpy as np
    import pytest

    import pymort.lifetables as lt

    with pytest.raises(ValueError):
        lt.survival_paths_from_q_paths(np.array([0.1, 0.2], dtype=float))

    # includes out-of-range values -> should be clipped to [0,1] internally
    q_paths = np.array(
        [[[-0.5, 0.2, 1.5], [0.1, 2.0, -3.0]]],
        dtype=float,
    )  # (N=1,A=2,H=3)
    S = lt.survival_paths_from_q_paths(q_paths)
    assert S.shape == q_paths.shape
    assert np.isfinite(S).all()
    assert np.all((S >= 0) & (S <= 1))


def test_find_header_and_map_keeps_first_occurrence():
    import pandas as pd

    import pymort.lifetables as lt

    # duplicate "Year" and "Age" columns; should keep the FIRST occurrence
    df = pd.DataFrame(
        [
            ["Age", "Year", "Year", "Total", "Age"],
            [60, 2000, 9999, 0.01, 777],
        ]
    )
    hrow, cmap = lt._find_header_and_map(df)
    assert hrow == 0
    assert cmap["Age"] == 0
    assert cmap["Year"] == 1
    assert cmap["Total"] == 3


def test_load_m_from_excel_any_accepts_bytes_and_filelike_and_uploadedfile(monkeypatch, tmp_path):
    import io

    import numpy as np
    import pandas as pd

    import pymort.lifetables as lt

    # fake sheets for pd.read_excel(excel_obj, sheet_name=None, header=None)
    rows = [
        ["junk", None, None],
        ["Age", "Year", "Total"],
        [60, 2000, 0.01],
        [61, 2000, 0.02],
        [60, 2001, 0.011],
        [61, 2001, 0.021],
    ]
    fake_sheets = {"S": pd.DataFrame(rows)}

    def fake_read_excel(*args, **kwargs):
        # ensure we are using sheet_name=None and header=None in both loaders
        assert kwargs.get("sheet_name") is None
        assert kwargs.get("header") is None
        return fake_sheets

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    # 1) bytes
    out1 = lt.load_m_from_excel_any(b"dummy-bytes", age_min=60, age_max=61)
    _ages, _years, m = out1["m"]
    assert m.shape == (2, 2)
    assert np.isfinite(m).all()

    # 2) file-like
    out2 = lt.load_m_from_excel_any(io.BytesIO(b"dummy"), age_min=60, age_max=61)
    _ages2, _years2, m2 = out2["m"]
    assert m2.shape == (2, 2)

    # 3) fake UploadedFile with getbuffer()
    class FakeUpload:
        def getbuffer(self):
            return b"dummy"

    out3 = lt.load_m_from_excel_any(FakeUpload(), age_min=60, age_max=61)
    _ages3, _years3, m3 = out3["m"]
    assert m3.shape == (2, 2)


def test_load_m_from_excel_any_raises_if_no_valid_sheet(monkeypatch, tmp_path):
    import pandas as pd
    import pytest

    import pymort.lifetables as lt

    def fake_read_excel(*args, **kwargs):
        return {"S": pd.DataFrame([["no", "header"], [1, 2]])}

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    with pytest.raises(ValueError):
        lt.load_m_from_excel_any(b"dummy")
