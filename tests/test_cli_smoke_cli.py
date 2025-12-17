from __future__ import annotations

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from pymort import cli


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() != ""


def test_data_validate_and_to_q(tmp_path):
    # simple 2x2 mortality surface
    ages = [60, 61]
    years = [2000, 2001]
    m = np.array([[0.01, 0.011], [0.015, 0.016]])
    df = pd.DataFrame(m, columns=years)
    df.insert(0, "Age", ages)
    m_path = tmp_path / "m.csv"
    df.to_csv(m_path, index=False)

    runner = CliRunner()
    res_val = runner.invoke(
        cli.app,
        ["--outdir", str(tmp_path), "data", "validate-m", "--m-path", str(m_path)],
    )
    assert res_val.exit_code == 0

    res_q = runner.invoke(
        cli.app,
        [
            "--outdir",
            str(tmp_path),
            "data",
            "to-q",
            "--m-path",
            str(m_path),
            "--output",
            str(tmp_path / "q.npz"),
        ],
    )
    assert res_q.exit_code == 0


def test_scen_build_p_pipeline(tmp_path):
    # Small surface (A=2, T=3) wide CSV
    ages = [60, 61]
    years = [2000, 2001, 2002]
    m = np.array([[0.01, 0.011, 0.012], [0.015, 0.016, 0.017]])
    df = pd.DataFrame(m, columns=years)
    df.insert(0, "Age", ages)
    m_path = tmp_path / "m.csv"
    df.to_csv(m_path, index=False)

    # Skip if cpsplines is not available in the environment
    import importlib.util

    if importlib.util.find_spec("cpsplines") is None:
        return

    runner = CliRunner()
    out_npz = tmp_path / "scenarios_P.npz"
    res = runner.invoke(
        cli.app,
        [
            "--outdir",
            str(tmp_path),
            "scen",
            "build-P",
            "--m-path",
            str(m_path),
            "--train-end",
            "2001",
            "--horizon",
            "1",
            "--n-scenarios",
            "2",
            "--output",
            str(out_npz),
        ],
    )
    assert res.exit_code == 0
    assert out_npz.exists()
    data = np.load(out_npz)
    assert "q_paths" in data
    q_paths = data["q_paths"]
    assert q_paths.ndim == 3
    assert q_paths.shape[0] > 0
