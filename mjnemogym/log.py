# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Centralized logging for all mjnemogym scoring domains.
#
# Usage:
#   from mjnemogym.log import get_logger
#   _logger = get_logger("math")   # creates logger "mjnemogym.math"
#
# Control via environment variable:
#   MJNEMOGYM_DEBUG=1  → DEBUG level (entry/exit, timing, values)
#   default (unset)    → WARNING level (errors, timeouts only)

import logging
import os

_DEBUG = os.environ.get("MJNEMOGYM_DEBUG", "0") == "1"
_CONFIGURED = set()


def get_logger(domain: str) -> logging.Logger:
    """
    Get a domain-specific logger with consistent formatting.

    Args:
        domain: Short domain name, e.g. "math", "code", "mcqa".
                Creates logger named "mjnemogym.{domain}".

    Returns:
        Configured logging.Logger instance.
    """
    name = f"mjnemogym.{domain}"
    logger = logging.getLogger(name)

    if name not in _CONFIGURED:
        _CONFIGURED.add(name)
        level = logging.DEBUG if _DEBUG else logging.WARNING
        logger.setLevel(level)

        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s][pid=%(process)d] %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
