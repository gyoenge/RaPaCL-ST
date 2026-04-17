from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logger(
        log_dir: str, 
        name: str, 
        rank: int, # for distributed setting 
    ) -> tuple[str, logging.Logger]:
    """setup logger, including distributed setting"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"run_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear() 

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0: 
        # main process logger setting 
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler() 
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    else: 
        # other processes logger setting 
        logger.setLevel(logging.ERROR)
        logger.addHandler(logging.NullHandler())

    logger.info("Logger initialized")
    logger.info("Log file: %s", log_path)

    return timestamp, logger


def setup_warnings(mode: str = "ignore"):
    """setup warnings"""
    import warnings

    if mode == "ignore":
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    elif mode == "default":
        warnings.resetwarnings()

    elif mode == "strict":
        warnings.filterwarnings("error")

