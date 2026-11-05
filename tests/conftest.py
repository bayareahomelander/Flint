import pytest

from flint_lib import Flint

# pytester runs pytest inside a test, which test_gpt2.py uses to check retry_report's output.
pytest_plugins = ["pytester", "retry_report"]


@pytest.fixture(scope="session")
def flint() -> Flint:
    return Flint()
