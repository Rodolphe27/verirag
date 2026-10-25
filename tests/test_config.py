from pathlib import Path

from verirag.config import load_env_file


def test_values_are_read_and_quotes_removed(tmp_path: Path) -> None:
    file = tmp_path / ".env"
    file.write_text("A=1\nB=\"two words\"\nC='x'\n", encoding="utf-8")
    env: dict[str, str] = {}
    load_env_file(file, env)
    assert env == {"A": "1", "B": "two words", "C": "x"}


def test_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    file = tmp_path / ".env"
    file.write_text("# a comment\n\nnot a pair\nA=1\n", encoding="utf-8")
    env: dict[str, str] = {}
    load_env_file(file, env)
    assert env == {"A": "1"}


def test_the_real_environment_wins(tmp_path: Path) -> None:
    file = tmp_path / ".env"
    file.write_text("A=from-file\n", encoding="utf-8")
    env = {"A": "from-terminal"}
    load_env_file(file, env)
    assert env["A"] == "from-terminal"


def test_a_missing_file_is_fine(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    load_env_file(tmp_path / "none", env)
    assert env == {}
