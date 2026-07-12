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


def test_a_known_problem_shows_the_failure_but_lets_the_notebook_continue(capsys):
    """Some failures are open research questions, not broken code. The ❌ must stay --
    we do not dress up a bad result -- but the notebook has to be runnable."""
    report(
        "The predictor",
        [("Beats persistence", False, "no — worse than doing nothing"),
         ("Dictionary alive", False, "perplexity 12 of 512")],
        have="A model that reads sentences.",
        known_problem="Diagnosed, not fixed: docs/OPEN-QUESTION-codebook-collapse.md",
    )
    out = capsys.readouterr().out
    assert "❌" in out
    assert "A KNOWN PROBLEM" in out
    assert "do not pretend this passed" in out


def test_a_known_problem_does_not_excuse_an_unexplained_failure():
    # Without a written explanation, a failure still stops everything.
    with pytest.raises(CheckFailed):
        report("Setup", [("Torch", False, "missing")], have="nothing")


def test_a_failure_NAMES_the_tests_that_broke(tmp_path):
    """'2 failed' is useless — you cannot act on a count.

    This happened for real: the Colab run reported '2 failed, 176 passed' and the reader
    had no way to know WHICH two, or why.
    """
    (tmp_path / "test_mixed.py").write_text(
        "def test_fine():\n    assert True\n"
        "def test_broken_thing():\n    assert False\n"
        "def test_other_broken_thing():\n    assert False\n"
    )
    passed, summary = run_tests(str(tmp_path))
    assert passed is False
    assert "test_broken_thing" in summary
    assert "test_other_broken_thing" in summary


def test_a_clean_run_stays_short(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    passed, summary = run_tests(str(tmp_path))
    assert passed is True
    assert summary == "1 passed"
