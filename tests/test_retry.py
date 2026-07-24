import pytest
from unittest.mock import patch

from runtime.retry import execute_with_retry


def test_execute_success_first_attempt():

    def succeed():
        return "success"

    with patch("runtime.retry.time.sleep") as mock_sleep:

        result = execute_with_retry(
            succeed,
            tool_name="dummy",
            max_retries=2,
        )

        assert result == "success"
        mock_sleep.assert_not_called()


def test_execute_success_after_retry():

    attempts = 0

    def flaky():

        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise ValueError("Temporary failure")

        return "success"

    with patch("runtime.retry.time.sleep") as mock_sleep:

        result = execute_with_retry(
            flaky,
            tool_name="dummy",
            max_retries=2,
        )

        assert result == "success"
        assert attempts == 2
        mock_sleep.assert_called_once_with(1)


def test_execute_failure_after_max_retries():

    def always_fail():
        raise ValueError("Permanent failure")

    with patch("runtime.retry.time.sleep"):

        with pytest.raises(ValueError, match="Permanent failure"):

            execute_with_retry(
                always_fail,
                tool_name="dummy",
                max_retries=2,
            )


def test_exponential_backoff():

    def always_fail():
        raise RuntimeError("Boom")

    with patch("runtime.retry.time.sleep") as mock_sleep:

        with pytest.raises(RuntimeError):

            execute_with_retry(
                always_fail,
                tool_name="dummy",
                max_retries=3,
            )

        assert mock_sleep.call_args_list == [
            ((1,),),
            ((2,),),
            ((4,),),
        ]