import pytest

import bubble_bi as bb
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


def test_the_hardware_report_spots_a_cpu_only_torch_build(monkeypatch):
    """The trap this exists for.

    Colab ships a CUDA build of PyTorch, but a careless pip install can replace it with a
    CPU-only wheel. Then `cuda.is_available()` is False on a machine with a perfectly
    good GPU sitting idle, and "Hardware: CPU" tells you nothing about WHY.

    ⚠️ `torch.version.cuda is None` is NOT the tell, and believing it was cost us an
    afternoon. A Colab CPU runtime ALSO ships a CPU-only wheel — identical symptom, and
    nothing is wrong with it. The tell is a CPU-only wheel on a machine that HAS a GPU,
    which is why this test now pins `gpu_present`. Without that pin it was asserting we
    blame a pip install having never checked whether a GPU exists at all.
    """
    import torch

    from bubble_bi import settings as settings_module
    from bubble_bi.settings import hardware

    monkeypatch.setattr(settings_module, "gpu_present", lambda: True)   # a GPU IS here
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", None)

    kit = hardware()
    assert kit["where"] == "cpu"
    assert kit["built for cuda"] is None
    assert kit["gpu present"] is True
    assert "CPU-only wheel" in kit["why"]
    assert "not let anything install `torch`" in kit["why"]


def test_the_hardware_report_distinguishes_no_gpu_from_no_cuda_torch(monkeypatch):
    # A CUDA-capable torch that simply cannot find a GPU is a DIFFERENT problem, with a
    # different fix — turn the runtime on, rather than reinstall everything.
    import torch

    from bubble_bi import settings as settings_module
    from bubble_bi.settings import hardware

    monkeypatch.setattr(settings_module, "gpu_present", lambda: False)  # no GPU here
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", "12.1")

    kit = hardware()
    assert "Change runtime type" in kit["why"]
    # Must not mention the wheel at all. This torch is perfectly CUDA-capable — saying
    # "CPU-only" here would be false, and it would send the reader off to reinstall
    # PyTorch when all they need is to tick a different runtime.
    assert "CPU-only" not in kit["why"]
    assert "install `torch`" not in kit["why"]


def test_the_setup_check_does_not_run_the_test_suite(monkeypatch):
    """It used to run all 255 tests — 112 seconds, ~100 of them TRAINING MODELS — every
    single time you touched the setup cell, before a single price had been downloaded. On
    Colab that is two minutes of a paid GPU session, per run.

    The setup check answers "does this environment work". The test suite answers "does the
    science hold". Those are different questions, and only one of them belongs in a cell
    you re-run constantly. The suite is still one command away: `bb.run_tests()`.
    """
    import importlib

    from bubble_bi import verify

    # NOT `from bubble_bi import report` — the package re-exports the FUNCTION `report`
    # over its own submodule of the same name, so that would hand us the wrong object.
    report_module = importlib.import_module("bubble_bi.report")

    def never(*a, **k):
        raise AssertionError(
            "setup() ran the whole test suite again — that is 2 minutes per run, and ~90% "
            "of it is training models that have nothing to do with whether your install works."
        )

    # Patched at the SOURCE, so it fires however setup() might reach it.
    monkeypatch.setattr(report_module, "run_tests", never)
    monkeypatch.setattr(verify, "hardware", lambda: {
        "where": "gpu", "torch": "2.11.0+cu128", "built for cuda": "12.8",
        "cuda available": True, "gpu": "Tesla T4", "gpu present": True, "why": None,
    })

    verify.setup(bb.check({"tickers": ["AAPL"], "data_dir": "artifacts"}))   # must not raise


def test_the_setup_check_still_says_HOW_to_check_the_code(capsys):
    """Dropping the suite from setup buys speed at a cost: someone cloning this repo can
    now finish the setup cell without ever learning whether the code itself is healthy.
    So the check has to TELL them, or we have simply hidden the question."""
    from bubble_bi import verify

    verify.setup(bb.check({"tickers": ["AAPL"], "data_dir": "artifacts"}))
    said = capsys.readouterr().out
    assert "run_tests" in said, "never told the reader how to check the code"
