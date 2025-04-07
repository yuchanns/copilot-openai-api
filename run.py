import argparse
import multiprocessing
import os
import secrets

import uvicorn


def run_app(reload: bool = False):
    workers = (
        None
        if reload
        else min(
            int(os.getenv("COPILOT_SERVER_WORKERS", multiprocessing.cpu_count())), 4
        )
    )

    # Get port from environment variable, default to 9191
    port = int(os.getenv("COPILOT_SERVER_PORT", "9191"))
    os.environ["COPILOT_TOKEN"] = os.environ.get(
        "COPILOT_TOKEN"
    ) or secrets.token_urlsafe(32)

    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=workers, reload=reload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI application")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    args = parser.parse_args()

    run_app(reload=args.reload)
