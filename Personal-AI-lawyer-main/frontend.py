"""
Deprecated entry point.

Run the unified app instead:

    streamlit run app.py

This file remains only so old shortcuts keep working.
"""

import runpy
from pathlib import Path

print("frontend.py is deprecated — launching app.py …")
runpy.run_path(str(Path(__file__).resolve().parent / "app.py"), run_name="__main__")
