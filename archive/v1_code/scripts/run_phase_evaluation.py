#!/usr/bin/env python
"""Run the registered, resumable phase evaluation observer."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from scglm_phase.runner import main

if __name__ == '__main__':
    main()
