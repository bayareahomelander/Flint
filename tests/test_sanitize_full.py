"""scripts/sanitize_full.py: which cases it runs and what passes one, without the tools."""

import importlib.util
import math
import sys

import pytest
import torch

import flint_lib

SCRIPTS = flint_lib.ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("sanitize_full", SCRIPTS / "sanitize_full.py")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)

CLEAN = "========= ERROR SUMMARY: 0 errors"
RACECHECK_CLEAN = "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
LOSS = f"{full.LOSS_LINE}10.97864"


def run_main(monkeypatch, capsys, arguments, outputs, use=(0, 0), exit_code=0):
    """main() with the tool replaced: each case's run returns exit_code and the next of outputs
    (the last one repeats). Returns its exit code, what it printed and the commands."""
    commands, pending = [], list(outputs)

    def run(command):
        commands.append(command)
        return exit_code, pending.pop(0) if len(pending) > 1 else pending[0], 1.0

    monkeypatch.setattr(full.sanitize, "run", run)
    monkeypatch.setattr(full.shutil, "which", lambda name: "compute-sanitizer")
    monkeypatch.setattr(full, "gpu_use", lambda: use)
    monkeypatch.setattr(full.sys, "argv", ["sanitize_full.py", *arguments])
    code = full.main()
    return code, capsys.readouterr().out, commands


def child_arguments(command):
    """A child's --layers, --sequences, --length and --separate, as integers."""
    return tuple(int(command[command.index(flag) + 1]) for flag in ("--layers", "--sequences", "--length", "--separate"))


def test_the_default_cases_are_gpt2s_micro_batch(monkeypatch, capsys):
    code, printed, commands = run_main(monkeypatch, capsys, [], [f"{LOSS}\n{CLEAN}\n{RACECHECK_CLEAN}"])
    assert code == 0, printed
    tools = [command[command.index("--tool") + 1] for command in commands]
    assert list(zip(tools, map(child_arguments, commands))) == [
        ("memcheck", (12, 16, 1024, 1)),
        ("racecheck", (2, 16, 1024, 1)),
        ("synccheck", (2, 16, 1024, 1)),
        ("initcheck", (12, 16, 1024, 1)),
        ("initcheck", (12, 16, 1024, 0)),
    ]
    assert printed.count("PASS") == 5 and "FAIL" not in printed
    # memcheck, racecheck and synccheck see only Flint's kernels; initcheck sees torch's writes.
    for tool, command in zip(tools, commands):
        assert ("kns=5flint" in command) == (tool != "initcheck")


def test_the_overrides_reach_every_case(monkeypatch, capsys):
    arguments = ["--tools", "memcheck,initcheck", "--layers", "1", "--sequences", "2", "--length", "64"]
    code, _, commands = run_main(monkeypatch, capsys, arguments, [f"{LOSS}\n{CLEAN}"])
    assert code == 0
    assert [child_arguments(c) for c in commands] == [(1, 2, 64, 1), (1, 2, 64, 1), (1, 2, 64, 0)]


def test_backward_reaches_every_case_and_only_when_asked(monkeypatch, capsys):
    code, printed, commands = run_main(monkeypatch, capsys, ["--backward"], [f"{LOSS}\n{CLEAN}\n{RACECHECK_CLEAN}"])
    assert code == 0, printed
    assert len(commands) == 5 and all("--backward" in command for command in commands)
    assert printed.count("backward") == 5
    _, printed, commands = run_main(monkeypatch, capsys, [], [f"{LOSS}\n{CLEAN}\n{RACECHECK_CLEAN}"])
    assert not any("--backward" in command for command in commands)
    assert "backward" not in printed


@pytest.mark.parametrize(
    "output",
    [
        f"{LOSS}\n========= ERROR SUMMARY: 1 error",
        # A run that crashed before its summary, or never printed a loss.
        LOSS,
        CLEAN,
        f"{full.LOSS_LINE}nan\n{CLEAN}",
        f"{full.LOSS_LINE}inf\n{CLEAN}",
        # A loss far from an untrained model's.
        f"{full.LOSS_LINE}3.2\n{CLEAN}",
    ],
    ids=["an-error", "no-summary", "no-loss", "nan-loss", "inf-loss", "wrong-loss"],
)
def test_a_case_fails_on_an_error_a_missing_summary_or_a_bad_loss(monkeypatch, capsys, output):
    code, printed, _ = run_main(monkeypatch, capsys, ["--tools", "memcheck"], [output])
    assert code == 1
    assert "FAIL" in printed and "PASS" not in printed


def test_racecheck_warnings_fail(monkeypatch, capsys):
    output = f"{LOSS}\n========= RACECHECK SUMMARY: 2 hazards displayed (0 errors, 2 warnings)"
    code, printed, _ = run_main(monkeypatch, capsys, ["--tools", "racecheck"], [output])
    assert code == 1 and "FAIL" in printed


def test_one_failed_case_fails_the_run(monkeypatch, capsys):
    outputs = [f"{LOSS}\n{CLEAN}", f"{LOSS}\n========= ERROR SUMMARY: 3 errors"]
    code, printed, _ = run_main(monkeypatch, capsys, ["--tools", "initcheck"], outputs)
    assert code == 1
    assert printed.count("PASS") == 1 and printed.count("FAIL") == 1


def test_a_nonzero_exit_fails_a_case_whatever_it_printed(monkeypatch, capsys):
    code, printed, _ = run_main(monkeypatch, capsys, ["--tools", "memcheck"], [f"{LOSS}\n{CLEAN}"], exit_code=1)
    assert code == 1 and "FAIL" in printed and "PASS" not in printed


@pytest.mark.parametrize("use", [(1, 0), (0, full.BUSY_MIB + 1), None], ids=["a-process", "memory-in-use", "unknown"])
def test_a_busy_or_unknown_gpu_is_refused_unless_allowed(monkeypatch, capsys, use):
    code, printed, commands = run_main(monkeypatch, capsys, ["--tools", "memcheck"], [f"{LOSS}\n{CLEAN}"], use)
    assert code == 2 and commands == [] and "--allow-busy" in printed
    code, _, commands = run_main(monkeypatch, capsys, ["--tools", "memcheck", "--allow-busy"], [f"{LOSS}\n{CLEAN}"], use)
    assert code == 0 and len(commands) == 1


def test_an_idle_gpu_with_a_little_memory_in_use_runs(monkeypatch, capsys):
    code, _, commands = run_main(monkeypatch, capsys, ["--tools", "memcheck"], [f"{LOSS}\n{CLEAN}"], (0, full.BUSY_MIB))
    assert code == 0 and len(commands) == 1


# -- --backward's check of the gradients (the child's exit code carries it) ------------------------

VOCAB, PADDED, WIDTH = 5, 8, 8


def wte_gradient(**changes):
    gradient = torch.zeros(PADDED, WIDTH)
    gradient[:VOCAB] = 1.0
    for what, value in changes.items():
        if what == "padded":
            gradient[VOCAB, 3] = value
        elif what == "real":
            gradient[:VOCAB] = value
    return gradient


def test_a_gradient_that_a_backward_pass_gives_is_accepted():
    assert full.gradient_problem("wte.weight", wte_gradient(), VOCAB) is None
    assert full.gradient_problem("blocks.0.ln_1.bias", torch.tensor([[0.0, 0.0, -2.5, 0.0]]), VOCAB) is None
    # Without padded rows, wte is an ordinary parameter.
    assert full.gradient_problem("wte.weight", torch.ones(VOCAB, WIDTH), VOCAB) is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_a_gradient_that_is_not_finite_is_refused(bad):
    gradient = torch.ones(3, 4)
    gradient[1, 2] = bad
    assert "not finite" in full.gradient_problem("blocks.1.mlp.c_fc.weight", gradient, VOCAB)
    assert "not finite" in full.gradient_problem("wte.weight", wte_gradient(real=bad), VOCAB)


def test_a_gradient_that_is_all_zero_is_refused():
    assert "all zero" in full.gradient_problem("ln_f.weight", torch.zeros(1, 8), VOCAB)
    assert "all zero" in full.gradient_problem("wpe.weight", -torch.zeros(4, 8), VOCAB)
    # wte's real rows must have a value too: nonzero padded rows don't count for it, and are
    # refused for being nonzero anyway.
    assert "all zero" in full.gradient_problem("wte.weight", wte_gradient(real=0.0), VOCAB)


@pytest.mark.parametrize("value", [1e-30, 1.0, -0.0], ids=["tiny", "one", "negative-zero"])
def test_wtes_padded_rows_must_be_exactly_plus_zero(value):
    assert "padded rows" in full.gradient_problem("wte.weight", wte_gradient(padded=value), VOCAB)
    # The same value in a row below vocab is fine, and in another parameter nothing is padded.
    assert full.gradient_problem("blocks.0.attn.c_attn.weight", wte_gradient(padded=value), VOCAB) is None


def test_a_child_that_reports_a_bad_gradient_fails_the_case(monkeypatch, capsys):
    # The child's exit code is what carries the check: the parent already requires 0.
    output = f"{LOSS}\n{full.PROBLEM_LINE}the gradient of ln_f.bias is all zero\n{CLEAN}"
    code, printed, _ = run_main(monkeypatch, capsys, ["--tools", "memcheck", "--backward"], [output], exit_code=1)
    assert code == 1 and "FAIL" in printed and "PASS" not in printed
    assert "the gradient of ln_f.bias is all zero" in printed
