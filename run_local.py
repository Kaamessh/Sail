"""
Local Launcher for Maritime Freight Intelligence System.
Starts the FastAPI server with live static dashboard hosting on http://localhost:8000
"""

import sys
import uvicorn
from pathlib import Path

# Ensure workspace root is in path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

if __name__ == "__main__":
    print("=" * 70)
    print("🚢 AURA MARITIME FREIGHT INTELLIGENCE & CHARTERING SYSTEM")
    print("=" * 70)
    print("🌐 Dashboard UI & API live at: http://localhost:8000")
    print("📑 Swagger Documentation at:   http://localhost:8000/docs")
    print("❤️  System Health at:          http://localhost:8000/api/health")
    print("=" * 70)
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
