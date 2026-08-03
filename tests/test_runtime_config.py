from runtime.runtime_config import RuntimeConfig


def test_runtime_config_defaults():

    config = RuntimeConfig()

    assert config.default_timeout == 30.0
    assert config.default_retries == 2
    assert config.max_parallel_workers == 4
    assert config.enable_metrics is True
    assert config.enable_audit is True
    assert config.enable_logging is True


def test_runtime_config_custom_values():

    config = RuntimeConfig(
        default_timeout=60,
        default_retries=5,
        max_parallel_workers=10,
        enable_metrics=False,
    )

    assert config.default_timeout == 60
    assert config.default_retries == 5
    assert config.max_parallel_workers == 10
    assert config.enable_metrics is False