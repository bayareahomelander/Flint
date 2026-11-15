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


def fake_tools(monkeypatch, capsys, suite_output, canary_output=None, canary_code=1, tools="racecheck", named=None):
    """Runs main() with the tool replaced: the suite prints suite_output and exits 0, and a
    canary exits canary_code printing canary_output (by default the message it must have).
    `named` is what FLINT_COMPUTE_SANITIZER holds, if anything. Returns the exit code, what it
    printed, and the commands run."""
    commands = []

    def run(command):
        commands.append(command)
        if "--canary" in command:
            output = canary_output if canary_output is not None else sanitize.CANARIES["out-of-bounds"][1]
            return canary_code, output, 0.1
        return 0, suite_output, 1.0

    monkeypatch.delenv(sanitize.SANITIZER_VARIABLE, raising=False)
    if named is not None:
        monkeypatch.setenv(sanitize.SANITIZER_VARIABLE, str(named))
    monkeypatch.setattr(sanitize, "run", run)
    monkeypatch.setattr(sanitize, "tool_version", lambda sanitizer: "Version 2026.1.1")
    monkeypatch.setattr(sanitize.shutil, "which", lambda name: "compute-sanitizer")
    monkeypatch.setattr(sanitize.sys, "argv", ["sanitize.py", "--tools", tools])
    code = sanitize.main()
    return code, capsys.readouterr().out, commands


def test_racecheck_warnings_fail_the_suite(monkeypatch, capsys):
    output = "========= RACECHECK SUMMARY: 2 hazards displayed (0 errors, 2 warnings)"
    code, printed, _ = fake_tools(monkeypatch, capsys, output)
    assert code == 1
    assert "racecheck (suite)        FAIL  errors=0 warnings=2" in printed


def test_clean_racecheck_passes(monkeypatch, capsys):
    code, printed, _ = fake_tools(monkeypatch, capsys, RACECHECK_CLEAN)
    assert code == 0
    assert "racecheck (suite)        PASS  errors=0 warnings=0" in printed


def test_a_canary_with_its_message_passes(monkeypatch, capsys):
    code, printed, _ = fake_tools(monkeypatch, capsys, RACECHECK_CLEAN)
    assert code == 0
    assert "memcheck canary (out-of-bounds)" in printed and "FAIL" not in printed


def test_a_crashed_canary_fails_and_is_explained(monkeypatch, capsys):
    # The tool died while reporting: it proves nothing, whatever the exit code says.
    code, printed, _ = fake_tools(
        monkeypatch, capsys, RACECHECK_CLEAN, canary_output=f"========= {sanitize.REPORT_CRASH}"
    )
    assert code == 1
    assert "memcheck canary (out-of-bounds)" in printed and "FAIL" in printed
    assert "unreported" not in printed
    assert sanitize.CRASH_NOTE in printed and "2026.1.1" in printed and "sancheck" in printed


def test_a_canary_without_its_message_fails_unexplained(monkeypatch, capsys):
    code, printed, _ = fake_tools(monkeypatch, capsys, RACECHECK_CLEAN, canary_output="nothing wrong")
    assert code == 1 and "FAIL" in printed and sanitize.CRASH_NOTE not in printed


def test_a_canary_that_exits_zero_fails_even_with_its_message(monkeypatch, capsys):
    code, printed, _ = fake_tools(monkeypatch, capsys, RACECHECK_CLEAN, canary_code=0)
    assert code == 1 and "FAIL" in printed


def test_the_uninitialized_canary_must_report_too(monkeypatch, capsys):
    code, printed, _ = fake_tools(monkeypatch, capsys, "========= ERROR SUMMARY: 0 errors",
                                  canary_output=sanitize.REPORT_CRASH, tools="initcheck")  # fmt: skip
    assert code == 1 and "initcheck canary (uninitialized)" in printed and sanitize.CRASH_NOTE in printed


def test_the_named_tool_is_the_one_used(monkeypatch, capsys, tmp_path):
    tool = tmp_path / "compute-sanitizer"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    code, printed, commands = fake_tools(monkeypatch, capsys, RACECHECK_CLEAN, named=tool)
    assert code == 0
    assert commands and all(command[0] == str(tool) for command in commands)
    assert f"compute-sanitizer: {tool} (Version 2026.1.1)" in printed
    # Without it, the one on PATH.
    _, _, commands = fake_tools(monkeypatch, capsys, RACECHECK_CLEAN)
    assert commands and all(command[0] == "compute-sanitizer" for command in commands)


def test_a_missing_named_tool_fails_naming_the_variable(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sanitize, "run", lambda command: pytest.fail("nothing should run"))
    monkeypatch.setattr(sanitize.shutil, "which", lambda name: "compute-sanitizer")
    monkeypatch.setenv(sanitize.SANITIZER_VARIABLE, str(tmp_path / "missing"))
    monkeypatch.setattr(sanitize.sys, "argv", ["sanitize.py", "--tools", "racecheck"])
    assert sanitize.main() == 1
    assert sanitize.SANITIZER_VARIABLE in capsys.readouterr().out


def test_a_named_directory_is_not_a_tool(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv(sanitize.SANITIZER_VARIABLE, str(tmp_path))
    assert sanitize.find_sanitizer() is None
    assert sanitize.SANITIZER_VARIABLE in capsys.readouterr().out


def test_without_the_variable_the_tool_is_found_on_path(monkeypatch, capsys):
    monkeypatch.delenv(sanitize.SANITIZER_VARIABLE, raising=False)
    monkeypatch.setattr(sanitize.shutil, "which", lambda name: "/cuda/bin/compute-sanitizer")
    assert sanitize.find_sanitizer() == "/cuda/bin/compute-sanitizer"
    monkeypatch.setattr(sanitize.shutil, "which", lambda name: None)
    assert sanitize.find_sanitizer() is None
    printed = capsys.readouterr().out
    assert "PATH" in printed and sanitize.SANITIZER_VARIABLE in printed


def test_a_tool_that_does_not_answer_version_does_not_fail_the_run(tmp_path):
    assert sanitize.tool_version(str(tmp_path / "missing")) == "version unknown"
