import argparse
import os

import uvicorn


def run_app(reload: bool = False):
    # Get the number of CPU cores
    workers = 1

    # Get port from environment variable, default to 9191
    port = int(os.getenv("COPILOT_SERVER_PORT", "9191"))

    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=workers, reload=reload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI application")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    args = parser.parse_args()

    run_app(reload=args.reload)
