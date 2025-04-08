import asyncio
import fcntl
import json
import logging
import os
import platform
import time

from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class TokenFileHandler(FileSystemEventHandler):
    def __init__(self, auth):
        self.auth = auth

    def on_modified(self, event):
        if event.src_path == str(self.auth.token_file):
            asyncio.create_task(self.auth.load_token_from_file())


class CopilotAuth:
    def __init__(self):
        self.oauth_token: Optional[str] = None
        self.github_token: Optional[Dict[str, Any]] = None
        self.refresh_task: Optional[asyncio.Task] = None
        self.observer: Optional[BaseObserver] = None
        self.file_handler: Optional[TokenFileHandler] = None

        if platform.system() == "Windows":
            config_dir = os.path.expanduser("~/AppData/Local")
        else:
            config_dir = os.path.expanduser("~/.config")

        self.config_dir = Path(config_dir)
        self.token_file = self.config_dir / "github-copilot" / "token.json"
        self.token_file.parent.mkdir(parents=True, exist_ok=True)

        # API endpoints
        self.auth_url = "https://api.github.com/copilot_internal/v2/token"

    async def get_oauth_token(self) -> str:
        """Get OAuth token from GitHub configuration file"""
        for path in ["apps.json", "hosts.json"]:
            file_path = self.config_dir / "github-copilot" / path
            if file_path.exists():
                async with aiofiles.open(file_path) as f:
                    hosts_data = json.loads(await f.read())
                    for host, data in hosts_data.items():
                        if "github.com" in host:
                            return data["oauth_token"]
        raise Exception("GitHub OAuth token not found")

    async def acquire_lock(self):
        """Acquire file lock for token refresh"""
        try:
            # Open the lock file in append mode (creates if not exists)
            lock_file = open(str(self.token_file) + ".lock", "a")
            # Try to acquire an exclusive lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        except IOError:
            # Another process has the lock
            return None

    async def save_token_to_file(self):
        """Save token to file"""
        async with aiofiles.open(self.token_file, "w") as f:
            await f.write(json.dumps(self.github_token))

    async def load_token_from_file(self):
        """Load token from file"""
        try:
            async with aiofiles.open(self.token_file, "r") as f:
                content = await f.read()
                if content:
                    self.github_token = json.loads(content)
                    logging.info("Token loaded from file")
        except FileNotFoundError:
            pass

    async def refresh_token(self, force: bool = False) -> bool:
        """Refresh Copilot token"""

        # Check if we already have a valid token in the memory
        if (
            not force
            and self.github_token
            and self.github_token["expires_at"] > time.time() + 120
        ):
            return True

        # Load newest token from file since other process might have updated it
        await self.load_token_from_file()

        # check again
        if (
            not force
            and self.github_token
            and self.github_token["expires_at"] > time.time() + 120
        ):
            return True

        # If we need to refresh, try to acquire lock
        lock_file = await self.acquire_lock()
        if not lock_file:
            # Another process is refreshing, wait and load from file again
            await asyncio.sleep(5)
            await self.load_token_from_file()
            # Return True if we got a valid token from file after waiting
            return bool(
                self.github_token
                and self.github_token["expires_at"] > time.time() + 120
            )

        try:
            # Send authentication request
            headers = {
                "Authorization": f"token {self.oauth_token}",
                "Accept": "application/json",
                "Editor-Plugin-Version": "copilot.lua",  # Simulate copilot.lua plugin
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(self.auth_url, headers=headers)
                if response.status_code == 200:
                    self.github_token = response.json()
                    await self.save_token_to_file()
                    return True

                raise Exception(f"Token refresh failed: {response.text}")
        finally:
            # Release the lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    async def setup(self):
        """Initialize"""
        # Get OAuth token
        self.oauth_token = await self.get_oauth_token()

        # Try to load existing token
        await self.load_token_from_file()

        # If no token exists or it's expired, force refresh
        if not self.github_token or self.github_token["expires_at"] <= time.time():
            await self.refresh_token(force=True)

        # Setup file watcher
        self.file_handler = TokenFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(
            self.file_handler, str(self.token_file.parent), recursive=False
        )
        self.observer.start()

        # Start refresh timer
        self.refresh_task = asyncio.create_task(self.setup_refresh_timer())

    async def cleanup(self):
        """Cleanup resources"""
        if self.refresh_task:
            self.refresh_task.cancel()
        if self.observer:
            self.observer.stop()
            self.observer.join()

    async def setup_refresh_timer(self):
        """Setup token refresh timer"""
        while True:
            try:
                await self.refresh_token()
                # Wait until next refresh time
                if self.github_token:
                    next_refresh = (
                        self.github_token["expires_at"] - 120
                    )  # Refresh 2 minutes before expiration
                    sleep_time = max(0, next_refresh - time.time())
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(60)  # If no token, retry after 1 minute
            except Exception as e:
                logging.error(f"Error in refresh timer: {e}")
                await asyncio.sleep(60)  # On error, retry after 1 minute


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    auth = CopilotAuth()
    await auth.setup()
    app.state.auth = auth

    # Generate or get token from environment
    app.state.access_token = os.environ.get("COPILOT_TOKEN")

    if not app.state.access_token:
        raise Exception("COPILOT_TOKEN environment variable not set")

    logging.info(f"Access token: {app.state.access_token}")

    yield
    await auth.cleanup()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allow all origins, should be restricted to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


async def proxy_stream(request: Request, url: str):
    # Get original request method, headers and body
    method = request.method
    headers = dict(request.headers)
    body = await request.body()

    # Remove some headers that don't need to be forwarded
    headers.pop("host", None)
    headers.pop("connection", None)
    headers.pop("Authorization", None)
    headers.pop("authorization", None)
    # Add authorization header
    auth = app.state.auth
    if auth.github_token:
        headers["Authorization"] = f"Bearer {auth.github_token['token']}"

    headers["Copilot-Integration-Id"] = "vscode-chat"
    headers["Editor-Version"] = "Neovim/0.9.0"

    try:

        async def stream_response():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    method=method, url=url, headers=headers, content=body, timeout=30.0
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_response())

    except Exception as e:
        return {"error": str(e)}


def require_auth(func):
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Missing or invalid authorization header"
            )

        token = auth_header.split(" ")[1]
        if token != request.app.state.access_token:
            raise HTTPException(status_code=403, detail="Invalid access token")

        return await func(request, *args, **kwargs)

    return wrapper


@app.api_route("/chat/completions", methods=["POST"])
@require_auth
async def proxy(request: Request):
    target_url = "https://api.githubcopilot.com/chat/completions"
    return await proxy_stream(request, target_url)
