"""Launcher used by autostart and `pythonw run.pyw`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vibetts.app import main  # noqa: E402

raise SystemExit(main())
