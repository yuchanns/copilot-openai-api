import asyncio
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
import fasteners
import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class CopilotAuth:
    def __init__(self):
        self.oauth_token: Optional[str] = None
        self.github_token: Optional[Dict[str, Any]] = None
        self.refresh_task: Optional[asyncio.Task] = None

        if platform.system() == "Windows":
            config_dir = os.path.expanduser("~/AppData/Local")
        else:
            config_dir = os.path.expanduser("~/.config")

        self.config_dir = Path(config_dir)

        # Ensure github-copilot directory exists
        copilot_dir = self.config_dir / "github-copilot"
        copilot_dir.mkdir(parents=True, exist_ok=True)

        # Create lock file path
        self.lock_file = copilot_dir / ".copilot.lock"
        self.lock = fasteners.InterProcessLock(str(self.lock_file))

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

    async def refresh_token(self, force: bool = False) -> bool:
        """Refresh Copilot token"""
        # Check if refresh is needed
        if not force and self.github_token:
            if (
                self.github_token["expires_at"] > time.time() + 120
            ):  # Refresh 2 minutes before expiration
                return False

        # Try to acquire the lock
        if not self.lock.acquire(blocking=False):
            # If we can't get the lock, wait for a short time and check if token was refreshed by another process
            await asyncio.sleep(1)
            try:
                async with aiofiles.open(
                    self.config_dir / "github-copilot" / "token.json", "r"
                ) as f:
                    self.github_token = json.loads(await f.read())
                    if (
                        self.github_token
                        and self.github_token["expires_at"] > time.time() + 120
                    ):
                        return True
            except (FileNotFoundError, json.JSONDecodeError):
                pass

            # If still no valid token, wait for lock
            self.lock.acquire()

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
                    # Save token to file for other processes
                    async with aiofiles.open(
                        self.config_dir / "github-copilot" / "token.json", "w"
                    ) as f:
                        await f.write(json.dumps(self.github_token))
                    return True

                raise Exception(f"Token refresh failed: {response.text}")
        finally:
            self.lock.release()

    async def setup(self):
        """Initialize"""
        # Get OAuth token
        self.oauth_token = await self.get_oauth_token()

        # Force refresh token once
        await self.refresh_token(force=True)

        # Start refresh timer
        self.refresh_task = asyncio.create_task(self.setup_refresh_timer())

    async def cleanup(self):
        """Cleanup resources"""
        if self.refresh_task:
            self.refresh_task.cancel()
        if self.lock.acquired:
            self.lock.release()

    async def setup_refresh_timer(self):
        """Setup token refresh timer"""
        while True:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    auth = CopilotAuth()
    await auth.setup()
    app.state.auth = auth

    # Generate or get token from environment
    app.state.access_token = os.environ.get("COPILOT_TOKEN")
    if not app.state.access_token:
        raise Exception(
            "COPILOT_TOKEN environment variable is not set. Please set it to a valid token."
        )
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
