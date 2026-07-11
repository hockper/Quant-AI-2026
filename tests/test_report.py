import pytest

from bubble_bi.report import CheckFailed, report, run_tests


def test_all_passing_prints_the_checks_and_what_we_have(capsys):
    report(
        "Setup",
        [("Settings understood", True, "30 companies"),
         ("Hardware", True, "CPU")],
        have="A checked configuration.",
    )
    out = capsys.readouterr().out
    assert "Setup" in out
    assert "30 companies" in out
    assert "What we have now" in out
    assert "A checked configuration." in out
    assert "❌" not in out


def test_a_failed_check_stops_the_notebook(capsys):
    # The whole point: you must NOT be able to keep going on a broken step.
    with pytest.raises(CheckFailed, match="PyTorch available"):
        report(
            "Setup",
            [("Settings understood", True, "30 companies"),
             ("PyTorch available", False, "required to train")],
            have="should never be printed",
        )
    out = capsys.readouterr().out
    assert "SOMETHING IS WRONG" in out
    assert "should never be printed" not in out    # no false reassurance


def test_run_tests_reports_success(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    passed, summary = run_tests(str(tmp_path))
    assert passed is True
    assert "1 passed" in summary


def test_run_tests_reports_failure(tmp_path):
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    passed, summary = run_tests(str(tmp_path))
    assert passed is False
    assert "failed" in summary
