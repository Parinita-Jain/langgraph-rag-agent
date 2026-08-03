from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:

    default_timeout: float = 30.0

    default_retries: int = 2

    max_parallel_workers: int = 4

    enable_metrics: bool = True

    enable_audit: bool = True

    enable_logging: bool = True