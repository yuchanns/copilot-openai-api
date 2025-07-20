import asyncio
import json
import logging
import os
import platform
import time

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

import aiofiles
import httpx

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from watchfiles import awatch


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
        self.tasks: list[asyncio.Task] = []

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
        """Acquire file lock for token refresh using file existence"""
        lock_path = str(self.token_file) + ".lock"
        try:
            Path(lock_path).touch(exist_ok=False)
            return True
        except FileExistsError:
            return False
        except Exception as e:
            logging.error(f"Error acquiring lock: {e}")
            return False

    async def release_lock(self):
        """Release the file lock"""
        lock_path = str(self.token_file) + ".lock"
        try:
            Path(lock_path).unlink()
            logging.debug("Lock file released successfully")
        except FileNotFoundError:
            logging.debug("Lock file already removed")
        except Exception as e:
            logging.error(f"Error releasing lock file: {e}")
            # Even if we fail to release the lock, don't raise the exception
            # as it might prevent cleanup in finally blocks

    async def save_token_to_file(self):
        """Save token to file"""
        temp_file = self.token_file.with_suffix(".tmp")
        self.is_self_writing = (
            True  # Set writing flag to indicate file changes triggered by self
        )
        try:
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(self.github_token))
            # Use atomic replace to ensure file consistency
            temp_file.replace(self.token_file)
            logging.info("Token successfully saved to file")
        except Exception as e:
            logging.error(f"Failed to save token to file: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logging.error(f"Failed to clean up temporary file: {e}")
            self.is_self_writing = False  # Reset flag when exception occurs
            raise

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

    def is_token_valid(self):
        return bool(
            self.github_token and self.github_token["expires_at"] > time.time() + 120
        )

    async def wait_for_token_refresh(self) -> bool:
        """Wait for another process to refresh the token"""
        await asyncio.sleep(5)
        await self.load_token_from_file()
        return self.is_token_valid()

    async def refresh_token(self, force: bool = False) -> bool:
        """Refresh Copilot token"""
        try:
            # Skip refresh if token is still valid and not forced
            if not force:
                if self.is_token_valid():
                    logging.debug("Token still valid, skipping refresh")
                    return True

                # Load newest token from file since other process might have updated it
                await self.load_token_from_file()
                if self.is_token_valid():
                    logging.debug("Valid token loaded from file, skipping refresh")
                    return True

            # Try to acquire lock for refresh once
            if not await self.acquire_lock():
                logging.info("Another process is refreshing, waiting for token update")
                return await self.wait_for_token_refresh()

            try:
                # Send authentication request
                headers = {
                    "Authorization": f"token {self.oauth_token}",
                    "Accept": "application/json",
                    "Editor-Plugin-Version": "copilot.lua",
                }

                async with httpx.AsyncClient() as client:
                    try:
                        response = await client.get(self.auth_url, headers=headers)
                        if response.status_code == 200:
                            self.github_token = response.json()
                            await self.save_token_to_file()
                            logging.info("Token successfully refreshed")
                            return True

                        error_msg = f"Token refresh failed with status {response.status_code}: {response.text}"
                        logging.error(error_msg)
                        raise Exception(error_msg)
                    except httpx.HTTPError as e:
                        logging.error(f"HTTP error during token refresh: {e}")
                        raise
            finally:
                # Release the lock
                await self.release_lock()
        except Exception as e:
            logging.error(f"Error in refresh_token: {e}")
            return False

    async def watch_token_file(self):
        """Watch token.json file for changes"""
        token_path = self.config_dir / "github-copilot" / "token.json"
        async for changes in awatch(str(token_path.parent)):
            for _, path in changes:
                if Path(path).name == "token.json":
                    if getattr(self, "is_self_writing", False):
                        self.is_self_writing = (
                            False  # If it's self-written, reset flag and skip loading
                        )
                        continue
                    await self.load_token_from_file()

    async def setup(self):
        """Initialize"""
        # Get OAuth token
        self.oauth_token = await self.get_oauth_token()

        # Load token from file
        await self.load_token_from_file()
        if not self.is_token_valid():
            # Force refresh token once
            await self.refresh_token(force=True)

        # Start refresh timer and stale lock checker
        self.tasks.append(asyncio.create_task(self.setup_refresh_timer()))
        self.tasks.append(asyncio.create_task(self.check_stale_locks()))

        # Start token file watcher
        self.tasks.append(asyncio.create_task(self.watch_token_file()))

    async def cleanup(self):
        """Cleanup resources"""
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for all tasks to complete their cancellation
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.tasks.clear()

        await self.release_lock()

    async def check_stale_locks(self):
        """Periodically check and clean up stale lock files"""
        while True:
            lock_path = str(self.token_file) + ".lock"
            try:
                if Path(lock_path).exists():
                    # Check if lock is older than 5 minutes
                    lock_age = time.time() - Path(lock_path).stat().st_mtime
                    if lock_age > 300:  # 5 minutes
                        try:
                            Path(lock_path).unlink()
                            logging.info("Removed stale lock file")
                        except FileNotFoundError:
                            pass
            except Exception as e:
                logging.error(f"Error checking stale locks: {e}")

            await asyncio.sleep(60)  # Check every minute

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


def verify_auth(
    authorization: Annotated[HTTPAuthorizationCredentials, Security(HTTPBearer())],
):
    token = authorization.credentials
    if token != app.state.access_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid access token",
        )


@app.post("/chat/completions", dependencies=[Depends(verify_auth)])
async def proxy(request: Request):
    target_url = "https://api.githubcopilot.com/chat/completions"
    return await proxy_stream(request, target_url)


# Mount self to the /v1 path
app.mount("/v1", app)
