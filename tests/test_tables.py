import pandas as pd

from shiftguard.utils.tables import to_markdown, write_table


def test_to_markdown_formats_floats_and_nan():
    df = pd.DataFrame({"a": ["x", "y"], "b": [0.12345, float("nan")], "c": [1, 2]})
    assert to_markdown(df, digits=2).splitlines() == [
        "| a | b | c |",
        "|---|---|---|",
        "| x | 0.12 | 1 |",
        "| y | - | 2 |",
    ]


def test_write_table(tmp_path):
    df = pd.DataFrame({"a": [1], "b": [2.0]})
    csv_path, md_path = write_table(df, tmp_path / "sub" / "t", markdown_columns=["b"], title="T")
    assert pd.read_csv(csv_path).equals(df)
    assert md_path.read_text(encoding="utf-8").startswith("# T\n\n| b |")
    # LF on every OS, so generated tables are identical on Windows and Linux
    assert b"\r\n" not in csv_path.read_bytes()
    assert b"\r\n" not in md_path.read_bytes()
