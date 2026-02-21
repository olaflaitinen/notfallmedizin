# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Setup script for notfallmedizin (backward compatibility with pip install .).

Installation is driven by pyproject.toml. This file exists for tools that
still expect setup.py. Prefer: pip install -e .
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
