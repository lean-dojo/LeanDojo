"""
General utilities for LeanAgent.
"""

import os
import re
import subprocess
import sys
import time
import typing
import urllib
from contextlib import contextmanager
from functools import cache
from typing import Generator, List, Optional, Tuple, Union

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from loguru import logger
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy


def zip_strict(*args):
    """
    Zip iterables with strict length checking.

    Args:
        *args: Iterables to zip together

    Returns:
        Iterator of tuples

    Raises:
        AssertionError: If the iterables have different lengths
    """
    assert len(args) > 1 and all(len(args[0]) == len(a) for a in args[1:])
    return zip(*args)


def _is_deepspeed_checkpoint(path: str):
    """Check if a checkpoint is a DeepSpeed checkpoint."""
    if not os.path.exists(path):
        raise FileExistsError(f"Checkpoint {path} does not exist.")
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "zero_to_fp32.py"))


def load_checkpoint(model_cls, ckpt_path: str, device, freeze: bool, config: dict):
    """Handle DeepSpeed checkpoints in model loading."""
    if not _is_deepspeed_checkpoint(ckpt_path):
        model = model_cls.load_from_checkpoint(ckpt_path, strict=False, **config).to(
            device
        )
    else:
        import tempfile

        path = os.path.join(tempfile.mkdtemp(), "lightning.cpkt")
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, path)
        model = model_cls.load_from_checkpoint(path, strict=False)
        model = model.to(device)
    if freeze:
        model.freeze()
    return model


def cpu_checkpointing_enabled(pl_module) -> bool:
    """Check if CPU checkpointing is enabled for the given PyTorch Lightning module."""
    try:
        trainer = pl_module.trainer
        return (
            trainer.strategy is not None
            and isinstance(trainer.strategy, DeepSpeedStrategy)
            and trainer.strategy.config["activation_checkpointing"]["cpu_checkpointing"]
        )
    except RuntimeError:
        return False


@contextmanager
def report_critical_failure(msg: str) -> Generator[None, None, None]:
    """Context manager logging ``msg`` in case of any exception.

    Args:
        msg (str): The message to log in case of exceptions.

    Raises:
        ex: Any exception that may be raised within the context manager.
    """
    try:
        yield
    except Exception as ex:
        logger.error(msg)
        raise ex


def execute(
    cmd: Union[str, List[str]], capture_output: bool = False
) -> Optional[Tuple[str, str]]:
    """Execute the shell command ``cmd`` and optionally return its output.

    Args:
        cmd (Union[str, List[str]]): The shell command to execute.
        capture_output (bool, optional): Whether to capture and return the output. Defaults to False.

    Returns:
        Optional[Tuple[str, str]]: The command's output, including stdout and stderr (None if ``capture_output == False``).
    """
    try:
        res = subprocess.run(cmd, shell=True, capture_output=capture_output, check=True)
    except subprocess.CalledProcessError as ex:
        if capture_output:
            logger.info(ex.stdout.decode())
            logger.error(ex.stderr.decode())
        raise ex
    if not capture_output:
        return None
    output = res.stdout.decode()
    error = res.stderr.decode()
    return output, error


_CAMEL_CASE_REGEX = re.compile(r"(_|-)+")


def camel_case(s: str) -> str:
    """Convert the string ``s`` to camel case."""
    return _CAMEL_CASE_REGEX.sub(" ", s).title().replace(" ", "")


def is_optional_type(tp: type) -> bool:
    """Test if ``tp`` is Optional[X]."""
    if typing.get_origin(tp) != Union:
        return False
    args = typing.get_args(tp)
    return len(args) == 2 and args[1] == type(None)


def remove_optional_type(tp: type) -> type:
    """Given Optional[X], return X."""
    if typing.get_origin(tp) != Union:
        return False
    args = typing.get_args(tp)
    if len(args) == 2 and args[1] == type(None):
        return args[0]
    else:
        raise ValueError(f"{tp} is not Optional")


@cache
def read_url(url: str, num_retries: int = 2) -> str:
    """Read the contents of the URL ``url``. Retry if failed"""
    backoff = 1
    while True:
        try:
            with urllib.request.urlopen(url) as f:
                return f.read().decode()
        except Exception as ex:
            if num_retries <= 0:
                raise ex
            num_retries -= 1
            logger.debug(f"Request to {url} failed. Retrying...")
            time.sleep(backoff)
            backoff *= 2


@cache
def url_exists(url: str) -> bool:
    """Return True if the URL ``url`` exists."""
    try:
        with urllib.request.urlopen(url) as _:
            return True
    except urllib.error.HTTPError:
        return False


def parse_int_list(s: str) -> List[int]:
    assert s.startswith("[") and s.endswith("]")
    return [int(_) for _ in s[1:-1].split(",") if _ != ""]


def parse_str_list(s: str) -> List[str]:
    assert s.startswith("[") and s.endswith("]")
    return [_.strip()[1:-1] for _ in s[1:-1].split(",") if _ != ""]
