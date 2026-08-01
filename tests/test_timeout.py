from runtime.timeout import run_with_timeout


def fast_function():

    return 42


def test_run_with_timeout_success():

    result = run_with_timeout(
        fast_function,
        timeout=1,
    )

    assert result == 42