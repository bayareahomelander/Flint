"""scripts/sanitize.py: reading compute-sanitizer's summaries, without the tools."""

import importlib.util

import pytest

import flint_lib

spec = importlib.util.spec_from_file_location("sanitize", flint_lib.ROOT / "scripts" / "sanitize.py")
sanitize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sanitize)

# A clean run's line, as compute-sanitizer 2024.1 prints it on the laptop.
RACECHECK_CLEAN = "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"


@pytest.mark.parametrize(
    "output, errors, warnings",
    [
        ("12 passed in 3.1s\n========= ERROR SUMMARY: 0 errors", 0, None),
        ("========= ERROR SUMMARY: 1 error", 1, None),
        # Past the print limit, a second line counts the errors not printed.
        ("========= ERROR SUMMARY: 1500 errors\n========= ERROR SUMMARY: 1400 errors were not printed",
         1500, None),
        (RACECHECK_CLEAN, 0, 0),
        ("========= RACECHECK SUMMARY: 3 hazards displayed (1 error, 2 warnings)", 1, 2),
        ("========= RACECHECK SUMMARY: 2 hazards displayed (0 errors, 2 warnings)", 0, 2),
        ("========= RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)", 0, 1),
        # A run that crashed before its summary.
        ("========= Internal Sanitizer Error: ...", None, None),
    ],
)
def test_summary_counts(output, errors, warnings):
    assert sanitize.error_count(output) == errors
    assert sanitize.warning_count(output) == warnings


def fake_tools(monkeypatch, capsys, suite_output):
    """Runs main() for racecheck with the tool replaced: the suite prints
    suite_output and exits 0, and the memcheck canary it needs is caught."""

    def run(command):
        if "--canary" in command:
            return 1, sanitize.CANARIES["out-of-bounds"][1], 0.1
        return 0, suite_output, 1.0

    monkeypatch.setattr(sanitize, "run", run)
    monkeypatch.setattr(sanitize.shutil, "which", lambda name: "compute-sanitizer")
    monkeypatch.setattr(sanitize.sys, "argv", ["sanitize.py", "--tools", "racecheck"])
    code = sanitize.main()
    return code, capsys.readouterr().out


def test_racecheck_warnings_fail_the_suite(monkeypatch, capsys):
    output = "========= RACECHECK SUMMARY: 2 hazards displayed (0 errors, 2 warnings)"
    code, printed = fake_tools(monkeypatch, capsys, output)
    assert code == 1
    assert "racecheck (suite)        FAIL  errors=0 warnings=2" in printed


def test_clean_racecheck_passes(monkeypatch, capsys):
    code, printed = fake_tools(monkeypatch, capsys, RACECHECK_CLEAN)
    assert code == 0
    assert "racecheck (suite)        PASS  errors=0 warnings=0" in printed
