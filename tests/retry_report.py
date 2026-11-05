"""A pytest plugin that keeps a gate that passed only on its retry from passing unnoticed.

test_gpt2.py's micro-batch gate retries once after a failure (gate_with_retry) and records the
first attempt's failure, and what the retry measured, as the test's user property PROPERTY.
This plugin reports such a test as "retried" in place of "passed", so that even pytest -q's last
line counts it ("3194 passed, 1 retried"), and prints a section with the recorded text. The
run's exit status doesn't change: the test passed. conftest.py registers the plugin."""

PROPERTY = "gate_retry"
CATEGORY = "retried"
TITLE = "GATE PASSED ONLY ON RETRY"


def retries(report):
    """The texts a test recorded under PROPERTY."""
    return [value for name, value in getattr(report, "user_properties", ()) if name == PROPERTY]


def pytest_report_teststatus(report, config):
    if report.when == "call" and report.passed and retries(report):
        return CATEGORY, "R", "RETRIED"
    return None


def pytest_terminal_summary(terminalreporter):
    reports = terminalreporter.stats.get(CATEGORY, [])
    if not reports:
        return
    terminalreporter.section(TITLE, sep="!", red=True, bold=True)
    for report in reports:
        terminalreporter.line(report.nodeid, red=True, bold=True)
        for text in retries(report):
            terminalreporter.line(text)
