from benchmarks.reporting import write_csv


def test_write_csv_handles_rows_with_different_keys(tmp_path):
    rows = [
        {"method": "a", "mean": 0.9},
        {"method": "b", "mean": 0.8, "delta_mean": -0.1},
    ]
    path = tmp_path / "out.csv"
    write_csv(path, rows)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "method,mean,delta_mean"
    assert lines[1] == "a,0.9,"
    assert lines[2] == "b,0.8,-0.1"
