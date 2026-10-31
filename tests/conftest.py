import pytest

from flint_lib import Flint


@pytest.fixture(scope="session")
def flint() -> Flint:
    return Flint()
