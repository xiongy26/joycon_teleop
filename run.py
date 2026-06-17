#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launcher for standalone Cartesian control GUI."""

import sys
from pathlib import Path

# Ensure we're in the right directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cartesian_control_gui import main

if __name__ == "__main__":
    main()
