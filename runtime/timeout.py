from concurrent.futures import ThreadPoolExecutor, TimeoutError


def run_with_timeout(
    func,
    *args,
    timeout: float,
    **kwargs,
):
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        future = executor.submit(
            func,
            *args,
            **kwargs,
        )

        return future.result(timeout=timeout)

    except TimeoutError:
        future.cancel()

        raise TimeoutError(
            f"Step timed out after {timeout} seconds."
        )

    finally:
        executor.shutdown(wait=False)