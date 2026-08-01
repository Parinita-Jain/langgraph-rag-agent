class RetryError(Exception):

    retries: int
    original_exception: Exception

    def __init__(
        self,
        retries: int,
        original_exception: Exception,
    ):
        super().__init__(str(original_exception))
        self.retries = retries
        self.original_exception = original_exception