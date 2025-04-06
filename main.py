from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from typing import Optional, Dict, Any
import asyncio
import platform
from pathlib import Path
import os
import aiofiles
import json
import time
from contextlib import asynccontextmanager


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
            if self.github_token["expires_at"] > time.time() + 120:  # Refresh 2 minutes before expiration
                return False
                
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
                return True
                
            raise Exception(f"Token refresh failed: {response.text}")

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

    async def setup_refresh_timer(self):
        """Setup token refresh timer"""
        while True:
            await self.refresh_token()
            # Wait until next refresh time
            if self.github_token:
                next_refresh = self.github_token["expires_at"] - 120  # Refresh 2 minutes before expiration
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
    yield
    await auth.cleanup()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, should be restricted to specific domains in production
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
    headers.pop('host', None)
    headers.pop('connection', None)
    headers.pop('Authorization', None)
    headers.pop('authorization', None)
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
                    method=method,
                    url=url,
                    headers=headers,
                    content=body,
                    timeout=30.0
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
            
        return StreamingResponse(stream_response())
        
    except Exception as e:
        return {"error": str(e)}

@app.api_route("/chat/completions", methods=["POST"])
async def proxy(request: Request):
    target_url = f"https://api.githubcopilot.com/chat/completions"
    
    return await proxy_stream(request, target_url)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
