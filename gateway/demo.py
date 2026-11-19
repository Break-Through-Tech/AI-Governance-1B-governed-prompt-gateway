"""Serve the local dashboard and gateway with a single Uvicorn worker."""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="local HTTP port (default: 8000)")
    args = parser.parse_args()

    # Set this before PyTorch or FAISS is imported.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import uvicorn

    api_dir = Path(__file__).resolve().parents[1] / "toxicity-detection-api"
    sys.path.insert(0, str(api_dir))
    uvicorn.run("main:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
