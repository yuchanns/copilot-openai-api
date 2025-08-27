import asyncio
import json
import logging
import os
import platform
import time
import uuid

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

import aiofiles
import httpx

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    Security,
    status,
)
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
                await asyncio.sleep(5)
                # Failed to acquire lock, wait for token to be updated by file watcher
                return False

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


async def proxy(request: Request, url: str):
    # Get original request method, headers and body
    method = request.method
    headers = dict(request.headers)
    body = await request.body()

    # Remove some headers that don't need to be forwarded
    headers.pop("host", None)
    headers.pop("connection", None)
    headers.pop("Authorization", None)
    headers.pop("authorization", None)
    headers.pop("content-length", None)
    # Add authorization header
    auth = app.state.auth
    if auth.github_token:
        headers["Authorization"] = f"Bearer {auth.github_token['token']}"

    headers["Copilot-Integration-Id"] = "vscode-chat"
    headers["Editor-Version"] = "Neovim/0.9.0"

    try:
        client = httpx.AsyncClient()
        req = client.build_request(
            method=method,
            url=url,
            headers=headers,
            content=body,
            timeout=30.0,
        )
        res = await client.send(request=req, stream=True)
        if "text/event-stream" not in res.headers.get("Content-Type"):
            content = (await res.aread()).strip()
            await res.aclose()
            await client.aclose()
            return Response(content)

        async def stream_response():
            async for chunk in res.aiter_bytes():
                yield chunk
            await res.aclose()
            await client.aclose()

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
async def proxy_completions(request: Request):
    target_url = "https://api.githubcopilot.com/chat/completions"
    return await proxy(request, target_url)


@app.post("/embeddings", dependencies=[Depends(verify_auth)])
async def proxy_embeddings(request: Request):
    target_url = "https://api.githubcopilot.com/embeddings"
    return await proxy(request, target_url)


def convert_request_anthropic_to_openai(body: Dict[str, Any]) -> Dict[str, Any]:
    messages = []
    if "system" in body:
        system = body["system"]
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = [{"type": "text", "text": item["text"]} for item in system]
            if text_parts:
                messages.append({"role": "system", "content": text_parts})
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role not in ["user", "assistant"]:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            if role == "user":
                tool_parts = [
                    tool
                    for tool in content
                    if tool.get("type") == "tool_result" and tool.get("tool_use_id")
                ]
                for tool in tool_parts:
                    tool_content = tool.get("content")
                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_content
                            if isinstance(tool_content, str)
                            else json.dumps(tool_content, ensure_ascii=False),
                            "tool_call_id": tool["tool_use_id"],
                        }
                    )
                text_media_parts = [
                    part
                    for part in content
                    if (part.get("type") == "text" and part.get("text"))
                    or (part.get("type") == "image" and part.get("source"))
                ]
                if not text_media_parts:
                    continue
                openai_content = []
                for part in text_media_parts:
                    if part["type"] == "image":
                        source = part["source"]
                        openai_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": source["data"]
                                    if source.get("type") == "base64"
                                    else source["url"]
                                },
                                "media_type": source.get("media_type"),
                            }
                        )
                    else:
                        openai_content.append(part)
                messages.append({"role": "user", "content": openai_content})
            elif role == "assistant":
                assistant_message = {"role": "assistant", "content": None}
                text_parts = [
                    part
                    for part in content
                    if part.get("type") == "text" and part.get("text")
                ]
                if text_parts:
                    assistant_message["content"] = "\n".join(
                        [part["text"] for part in text_parts]
                    )

                tool_call_parts = [
                    part
                    for part in content
                    if part.get("type") == "tool_use" and part.get("id")
                ]
                if tool_call_parts:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tool["id"],
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "arguments": json.dumps(
                                    tool.get("input") or {}, ensure_ascii=False
                                ),
                            },
                        }
                        for tool in tool_call_parts
                    ]
                messages.append(assistant_message)
    tools = None
    if body.get("tools"):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in body["tools"]
        ]
    return {
        "messages": messages,
        "model": body.get("model"),
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "stream": body.get("stream"),
        "tools": tools,
        "tool_choice": body.get("tool_choice"),
    }


anthropicStopResponseMap = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
}


async def iterator_convert_stream_response_openai_to_anthropic(body_iterator, charset):
    previousChunk: Optional[str] = None
    messageStart = False
    stopReason: Optional[Dict[str, Any]] = None
    currentContentBlockIndex = -1
    thinkingStart = False
    contentIndex = 0
    contentChunks = 0
    textContentStart = False
    toolCallChunks = 0
    toolCalls: Dict[str, Dict[str, Any]] = {}
    toolCallIndexToContentBlockIndex: Dict[int, int] = {}

    async def convert_stream_response_openai_to_anthropic(chunk: str):
        nonlocal previousChunk
        nonlocal stopReason
        nonlocal messageStart
        nonlocal currentContentBlockIndex
        nonlocal thinkingStart
        nonlocal contentIndex
        nonlocal contentChunks
        nonlocal textContentStart
        nonlocal toolCallChunks
        if chunk.startswith("data: "):
            chunk = chunk[6:]
        elif previousChunk:
            # might be a continuation of previous chunk
            chunk = previousChunk + chunk
            previousChunk = None
            logging.warning(f"Continuing previous chunk: {chunk}")
        if chunk == "[DONE]":
            logging.info(f"Stream DONE, previous_chunk: {previousChunk}")
            if not stopReason:
                stopReason = {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                    },
                }
            data = json.dumps(stopReason, ensure_ascii=False)
            stopReason = None
            yield b"event: message_delta\ndata: " + data.encode("utf-8") + b"\n\n"
            data = json.dumps({"type": "message_stop"}, ensure_ascii=False)
            yield b"event: message_stop\ndata: " + data.encode("utf-8") + b"\n\n"
            return

        if chunk == "":
            return
        try:
            body = json.loads(chunk)
            logging.info(f"Parsed chunk: {body}")
            if "error" in body:
                data = json.dumps(
                    {
                        "type": "error",
                        "message": {
                            "type": "api_error",
                            "message": json.dumps(body["error"]),
                        },
                    },
                    ensure_ascii=False,
                )
                yield b"event: error\ndata: " + data.encode("utf-8") + b"\n\n"
                return

            message_id = f"{int(time.time() * 1000)}"
            model = body.get("model") or "unknown"
            if not messageStart:
                messageStart = True
                data = json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "message_id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": 0,
                                "output_tokens": 0,
                            },
                        },
                    },
                    ensure_ascii=False,
                )
                yield b"event: message_start\ndata: " + data.encode("utf-8") + b"\n\n"
            if "usage" in body:
                if not stopReason:
                    stopReason = {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                        },
                        "usage": {
                            "input_tokens": body["usage"].get("prompt_tokens", 0),
                            "output_tokens": body["usage"].get("completion_tokens", 0),
                            "cache_read_input_tokens": body["usage"].get(
                                "cache_read_input_tokens", 0
                            ),
                        },
                    }
                else:
                    stopReason["usage"] = {
                        "input_tokens": stopReason["usage"].get("input_tokens", 0)
                        + body["usage"].get("prompt_tokens", 0),
                        "output_tokens": stopReason["usage"].get("output_tokens", 0)
                        + body["usage"].get("completion_tokens", 0),
                        "cache_read_input_tokens": stopReason["usage"].get(
                            "cache_read_input_tokens", 0
                        )
                        + body["usage"].get("cache_read_input_tokens", 0),
                    }
            if "choices" not in body or len(body["choices"]) == 0:
                return
            choice = body["choices"][0]
            if "delta" in choice and "thinking" in choice["delta"]:
                # close previous unclosed content block
                if currentContentBlockIndex >= 0:
                    data = json.dumps(
                        {
                            "type": "content_block_stop",
                            "index": currentContentBlockIndex,
                        },
                        ensure_ascii=False,
                    )
                    currentContentBlockIndex = -1
                    yield (
                        b"event: content_block_stop\ndata: "
                        + data.encode("utf-8")
                        + b"\n\n"
                    )
                if not thinkingStart:
                    thinkingStart = True
                    currentContentBlockIndex = contentIndex
                    data = json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": contentIndex,
                            "content_block": {"type": "thinking", "thinking": ""},
                        },
                        ensure_ascii=False,
                    )
                    yield (
                        b"event: content_block_start\ndata: "
                        + data.encode("utf-8")
                        + b"\n\n"
                    )
                if "signature" in choice["delta"]["thinking"]:
                    data = json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": contentIndex,
                            "delta": {
                                "type": "signature_delta",
                                "signature": choice["delta"]["thinking"]["signature"],
                            },
                        },
                        ensure_ascii=False,
                    )
                    yield (
                        b"event: content_block_delta\ndata: "
                        + data.encode("utf-8")
                        + b"\n\n"
                    )
                    data = json.dumps(
                        {"type": "content_block_stop", "index": contentIndex},
                        ensure_ascii=False,
                    )
                    yield (
                        b"event: content_block_delta\ndata: "
                        + data.encode("utf-8")
                        + b"\n\n"
                    )
                    currentContentBlockIndex = -1
                    contentIndex += 1
                elif "content" in choice["delta"]["thinking"]:
                    thinking_text = choice["delta"]["thinking"]["content"]
                    data = json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": contentIndex,
                            "delta": {
                                "type": "thinking_delta",
                                "thinking": thinking_text,
                            },
                        },
                        ensure_ascii=False,
                    )
                    yield (
                        b"event: content_block_delta\ndata: "
                        + data.encode("utf-8")
                        + b"\n\n"
                    )
            if "delta" in choice and "content" in choice["delta"]:
                contentChunks += 1

                # close previous unclosed non text content block
                if currentContentBlockIndex >= 0:
                    isCurrentTextBlock = textContentStart
                    if not isCurrentTextBlock:
                        data = json.dumps(
                            {
                                "type": "content_block_stop",
                                "index": currentContentBlockIndex,
                            },
                            ensure_ascii=False,
                        )
                        currentContentBlockIndex = -1
                        yield (
                            b"event: content_block_stop\ndata: "
                            + data.encode("utf-8")
                            + b"\n\n"
                        )
                if not textContentStart:
                    textContentStart = True
                    data = json.dumps(
                        {
                            "type": "content_block_start",
                            "index": contentIndex,
                            "content_block": {"type": "text", "text": ""},
                        },
                        ensure_ascii=False,
                    )
                    yield (
                        b"event: content_block_start\ndata: "
                        + data.encode("utf-8")
                        + b"\n\n"
                    )
                    currentContentBlockIndex = contentIndex
                data = json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": currentContentBlockIndex,
                        "delta": {
                            "type": "text_delta",
                            "text": choice["delta"]["content"] or "",
                        },
                    },
                    ensure_ascii=False,
                )
                yield (
                    b"event: content_block_delta\ndata: "
                    + data.encode("utf-8")
                    + b"\n\n"
                )
            if (
                "delta" in choice
                and "annotations" in choice["delta"]
                and len(choice["delta"]["annotations"]) > 0
            ):
                # close previous unclosed text content block
                if currentContentBlockIndex >= 0 and textContentStart:
                    data = json.dumps(
                        {
                            "type": "content_block_stop",
                            "index": currentContentBlockIndex,
                        },
                        ensure_ascii=False,
                    )
                    currentContentBlockIndex = -1
                    yield (
                        b"event: content_block_stop\ndata: "
                        + data.encode("utf-8")
                        + b"\n\n"
                    )
                    textContentStart = False
                    currentContentBlockIndex = contentIndex
                for annotation in choice["delta"]["annotations"]:
                    contentIndex += 1
                    data = json.dumps(
                        {
                            "type": "content_block_start",
                            "index": contentIndex,
                            "content_block": {
                                "type": "web_search_tool_result",
                                "tool_use_id": f"srvtoolu_{uuid.uuid4()}",
                                "content": {
                                    "type": "web_search_result",
                                    "title": annotation.get("url_citation", {}).get(
                                        "title"
                                    ),
                                    "url": annotation.get("url_citation", {}).get(
                                        "url"
                                    ),
                                },
                            },
                        },
                        ensure_ascii=False,
                    )
                    yield (
                        b"event: content_block_start\ndata: "
                        + data.encode("utf-8")
                        + b"\n\n"
                    )
                    data = json.dumps(
                        {
                            "type": "content_block_stop",
                            "index": contentIndex,
                        },
                        ensure_ascii=False,
                    )
                    yield (
                        b"event: content_block_stop\ndata: "
                        + data.encode("utf-8")
                        + b"\n\n"
                    )
                    currentContentBlockIndex = -1
            if "delta" in choice and "tool_calls" in choice["delta"]:
                toolCallChunks += 1
                processed = set()
                for toolCall in choice["delta"]["tool_calls"]:
                    toolCallIndex = toolCall.get("index", 0)
                    if toolCallIndex in processed:
                        continue
                    processed.add(toolCallIndex)
                    unknownIndex = toolCallIndex not in toolCallIndexToContentBlockIndex
                    if unknownIndex:
                        # close previous unclosed content block
                        if currentContentBlockIndex >= 0:
                            data = json.dumps(
                                {
                                    "type": "content_block_stop",
                                    "index": currentContentBlockIndex,
                                },
                                ensure_ascii=False,
                            )
                            currentContentBlockIndex = -1
                            yield (
                                b"event: content_block_stop\ndata: "
                                + data.encode("utf-8")
                                + b"\n\n"
                            )
                        newContentBlockIndex = contentIndex
                        toolCallIndexToContentBlockIndex[toolCallIndex] = (
                            newContentBlockIndex
                        )
                        contentIndex += 1
                        toolCallId = (
                            toolCall.get("id") or f"call_{time.time()}_{toolCallIndex}"
                        )
                        toolCallName = toolCall.get("name") or f"tool_{toolCallIndex}"
                        data = json.dumps(
                            {
                                "type": "content_block_start",
                                "index": newContentBlockIndex,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": toolCallId,
                                    "name": toolCallName,
                                    "input": {},
                                },
                            },
                            ensure_ascii=False,
                        )
                        yield (
                            b"event: content_block_start\ndata: "
                            + data.encode("utf-8")
                            + b"\n\n"
                        )
                        currentContentBlockIndex = newContentBlockIndex
                        toolCalls[toolCallId] = {
                            "id": toolCallId,
                            "name": toolCallName,
                            "arguments": "",
                            "content_block_index": newContentBlockIndex,
                        }
                    elif (
                        "id" in toolCall
                        and "function" in toolCall
                        and "name" in toolCall["function"]
                    ):
                        existToolCall = toolCalls.get(toolCall["id"])
                        if existToolCall:
                            wasTemp = existToolCall["id"].startswith(
                                "call_"
                            ) and existToolCall["name"].startswith("tool_")
                            if wasTemp:
                                existToolCall["id"] = toolCall["id"]
                                existToolCall["name"] = toolCall["function"]["name"]
                    if "function" in toolCall and "arguments" in toolCall["function"]:
                        blockIndex = toolCallIndexToContentBlockIndex[toolCallIndex]
                        if not blockIndex:
                            continue
                        currentToolCall = toolCalls.get(toolCallIndex)
                        if currentToolCall:
                            currentToolCall["arguments"] += toolCall["function"][
                                "arguments"
                            ]
                        try:
                            data = json.dumps(
                                {
                                    "type": "content_block_delta",
                                    "index": blockIndex,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": toolCall["function"][
                                            "arguments"
                                        ],
                                    },
                                },
                                ensure_ascii=False,
                            )
                        except Exception:
                            import re

                            fixed_argument = toolCall["function"]["arguments"]
                            fixed_argument = re.sub(
                                r"[\x00-\x1F\x7F-\x9F]", "", fixed_argument
                            )
                            fixed_argument = fixed_argument.replace("\\", "\\\\")
                            fixed_argument = fixed_argument.replace('"', '\\"')
                            try:
                                data = json.dumps(
                                    {
                                        "type": "content_block_delta",
                                        "index": blockIndex,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": fixed_argument,
                                        },
                                    },
                                    ensure_ascii=False,
                                )
                            except Exception as e:
                                logging.error(
                                    f"Failed to parse tool call arguments: {fixed_argument}, error: {e}"
                                )
                                continue

                        yield (
                            b"event: content_block_delta\ndata: "
                            + data.encode("utf-8")
                            + b"\n\n"
                        )
            if "finish_reason" in choice and choice["finish_reason"]:
                if contentChunks == 0 and toolCallChunks == 0:
                    logging.warning("No content in the stream response")
                # close previous unclosed content block
                if currentContentBlockIndex >= 0:
                    data = json.dumps(
                        {
                            "type": "content_block_stop",
                            "index": currentContentBlockIndex,
                        },
                        ensure_ascii=False,
                    )
                    currentContentBlockIndex = -1
                    yield (
                        b"event: content_block_stop\ndata: "
                        + data.encode("utf-8")
                        + b"\n\n"
                    )
                reason = anthropicStopResponseMap.get(
                    choice["finish_reason"], "end_turn"
                )
                stopReason = {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": reason,
                        "stop_sequence": None,
                    },
                    "usage": {
                        "input_tokens": body["usage"].get("prompt_tokens", 0),
                        "output_tokens": body["usage"].get("completion_tokens", 0),
                        "cache_read_input_tokens": body["usage"].get(
                            "cache_read_input_tokens", 0
                        ),
                    },
                }
                return

        except Exception as e:
            logging.warning(f"Failed to parse chunk: {chunk}, error: {e}")
            previousChunk = previousChunk + chunk if previousChunk else chunk

    async for chunk in body_iterator:
        if not isinstance(chunk, (bytes, memoryview)):
            chunk = chunk.encode(charset)
        elif isinstance(chunk, memoryview):
            chunk = chunk.tobytes()
        chunk = chunk.decode("utf-8").strip()
        # each chunk may contains multiple lines
        for line in chunk.split("\n\n"):
            async for ch in convert_stream_response_openai_to_anthropic(line):
                yield ch


def convert_response_openai_to_anthropic(body: Dict[str, Any]) -> Dict[str, Any]:
    choices = body.get("choices", [])
    try:
        if len(choices) == 0:
            raise ValueError("No choices in the response")
        choice = choices[0]
        content = []
        if "message" in choice:
            if "annotations" in choice["message"]:
                id = f"srvtoolu_{uuid.uuid4()}"
                content.append(
                    {
                        "type": "server_tool_use",
                        "id": id,
                        "name": "web_search",
                        "input": {"query": ""},
                    }
                )
                content.append(
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": id,
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": annotation.get("url_citation", {}).get("url"),
                                "title": annotation.get("url_citation", {}).get(
                                    "title"
                                ),
                            }
                            for annotation in choice["message"]["annotations"]
                        ],
                    }
                )
            if "content" in choice["message"]:
                content.append({"type": "text", "text": choice["message"]["content"]})
            if (
                "tool_calls" in choice["message"]
                and len(choice["message"]["tool_calls"]) > 0
            ):
                for tool_call in choice["message"]["tool_calls"]:
                    input_args = tool_call.get("function", {}).get("arguments", "{}")
                    input_json = {}
                    if isinstance(input_args, dict):
                        input_json = input_args
                    elif isinstance(input_args, str):
                        try:
                            input_json = json.loads(input_args)
                        except Exception:
                            input_json = {"text": input_args}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.get("id"),
                            "name": tool_call.get("function", {}).get("name"),
                            "input": input_json,
                        }
                    )
        usage = body.get("usage", {})
        return {
            "id": body.get("id", ""),
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "unknown"),
            "content": content,
            "stop_reason": anthropicStopResponseMap.get(
                choice.get("finish_reason", "end_turn"), "end_turn"
            ),
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0)
                + body["usage"].get("cache_read_input_tokens", 0),
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"{e}",
                    "type": "api_error",
                    "code": "internal_error",
                }
            },
        ) from e


@app.post("/messages", dependencies=[Depends(verify_auth)])
async def proxy_messages(request: Request):
    body = convert_request_anthropic_to_openai(await request.json())
    request._body = json.dumps(body, ensure_ascii=False).encode("utf-8")
    target_url = "https://api.githubcopilot.com/chat/completions"
    res = await proxy(request, target_url)
    if isinstance(res, StreamingResponse):
        res.body_iterator = iterator_convert_stream_response_openai_to_anthropic(
            res.body_iterator, res.charset
        )
    elif isinstance(res, Response):
        body = res.body
        if isinstance(body, memoryview):
            body = body.tobytes()
        headers = dict(res.headers)
        headers.pop("content-length", None)
        res = Response(
            content=json.dumps(
                convert_response_openai_to_anthropic(json.loads(body)),
                ensure_ascii=False,
            ),
            status_code=res.status_code,
            headers=headers,
            media_type=res.media_type,
        )
    return res


# Mount self to the /v1 path
app.mount("/v1", app)
