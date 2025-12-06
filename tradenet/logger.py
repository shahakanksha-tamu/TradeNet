import logging
import os
from datetime import datetime

    
# Creates a logger for a training or evaluation run.
# Logs both to console and to a file in LOG_DIR.

def create_logger(run_name, log_dir):

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")


    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )
    ch.setFormatter(ch_formatter)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger