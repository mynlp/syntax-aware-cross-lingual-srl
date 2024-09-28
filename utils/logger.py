import os
import logging
from utils import file


def add_handler(log_dir, log_filename, is_gold):
    logger = logging.getLogger(__name__)

    formatter = logging.Formatter('%(message)s')

    specific_log_dir = os.path.join(
        log_dir,
        'logs',
        'gold' if is_gold else 'pred'
    )

    log_filename = os.path.join(
        specific_log_dir,
        f'{log_filename}.log'
    )

    handler = logging.FileHandler(log_filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(log_dir, idx_run, is_gold):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    specific_log_dir = os.path.join(
        log_dir,
        'logs',
        'gold' if is_gold else 'pred'
    )

    log_filename = os.path.join(
        specific_log_dir,
        f'general_run_{idx_run}.log'
    )

    handler = logging.FileHandler(log_filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_eval_logger(log_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    log_filename = os.path.join(
        log_dir,
        'evals',
        'summary.log'
    )

    handler = logging.FileHandler(log_filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_dep_eval_logger(log_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    log_filename = os.path.join(
        log_dir,
        'evals',
        'dep_summary.log'
    )

    handler = logging.FileHandler(log_filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_dep_stat_logger(log_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    log_filename = os.path.join(
        log_dir,
        'evals',
        'dep_stat.log'
    )

    handler = logging.FileHandler(log_filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_arg_stat_logger(log_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    log_filename = os.path.join(
        log_dir,
        'evals',
        'arg_stat.log'
    )

    handler = logging.FileHandler(log_filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def remove_last_handler():
    logger = logging.getLogger(__name__)
    logger.removeHandler(logger.handlers[-1])


def clean_logger():
    logger = logging.getLogger(__name__)
    logger_handlers_len = len(logger.handlers)

    for i in range(logger_handlers_len):
        logger.removeHandler(logger.handlers[0])
