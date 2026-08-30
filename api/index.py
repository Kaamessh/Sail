"""
Vercel Serverless Entrypoint for Maritime Freight Intelligence API.
"""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.main import app

# Export both handler and app for compatibility with various Vercel configurations
handler = app

__all__ = ["app", "handler"]
