import csv

from benchmarks.reporting import load_config, write_csv, write_json


def test_write_csv_accepts_rows_with_different_keys(tmp_path):
    path = tmp_path / "out.csv"
    write_csv(path, [{"a": 1}, {"a": 2, "b": 3}])
    rows = list(csv.DictReader(path.open()))
    assert rows[0]["b"] == ""
    assert rows[1]["b"] == "3"


def test_write_json_and_missing_config(tmp_path):
    write_json(tmp_path / "x" / "r.json", {"k": 1})
    assert (tmp_path / "x" / "r.json").exists()
    assert load_config(tmp_path / "missing.yaml") == {}
