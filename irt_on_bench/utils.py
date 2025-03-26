from loguru import logger
import sys


def enable_logging() -> list[int]:
    """Set up the chembench logging with sane defaults."""
    logger.enable("chembench")

    config = dict(
        handlers=[
            dict(
                sink=sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC}</>"
                " <red>|</> <lvl>{level}</> <red>|</> <cyan>{name}:{function}:{line}</>"
                " <red>|</> <lvl>{message}</>",
                level="INFO",
            ),
            dict(
                sink=sys.stderr,
                format="<red>{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC} | {level} | {name}:{function}:{line} | {message}</>",  # noqa
                level="WARNING",
            ),
        ]
    )
    return logger.configure(**config)
