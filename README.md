# 🤖 Copilot OpenAI API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-modern-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Image Tags](https://ghcr-badge.yuchanns.xyz/yuchanns/copilot-openai-api/tags?ignore=latest)](https://ghcr.io/yuchanns/copilot-openai-api)
![Image Size](https://ghcr-badge.yuchanns.xyz/yuchanns/copilot-openai-api/size)

A FastAPI proxy server that seamlessly turns GitHub Copilot's chat completion/embeddings capabilities into OpenAI compatible API service.

## ✨ Key Features

🚀 **Advanced Integration**
- Seamless GitHub Copilot chat completion API proxy
- Real-time streaming response support
- High-performance request handling

🔐 **Security & Reliability**
- Secure authentication middleware
- Automatic token management and refresh
- Built-in CORS support for web applications

💻 **Universal Compatibility**
- Cross-platform support (Windows and Unix-based systems)
- Docker containerization ready
- Flexible deployment options

## 🚀 Prerequisites

- Python 3.10+
- pip (Python package manager)
- GitHub Copilot subscription
- GitHub authentication token

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yuchanns/copilot-openai-api.git
cd copilot_provider
```

2. Install dependencies:
```bash
# Install PDM first if you haven't
pip install -U pdm

# Install project dependencies using PDM
pdm install --prod
```

## ⚙️ Configuration

1. Set up environment variables:
```bash
# Windows
set COPILOT_TOKEN=your_access_token_here
set COPILOT_SERVER_PORT=9191          # Optional: Server port (default: 9191)
set COPILOT_SERVER_WORKERS=4          # Optional: Number of workers (default: min(CPU_COUNT, 4))

# Unix/Linux/macOS
export COPILOT_TOKEN=your_access_token_here
export COPILOT_SERVER_PORT=9191       # Optional: Server port (default: 9191)
export COPILOT_SERVER_WORKERS=4       # Optional: Number of workers (default: min(CPU_COUNT, 4))
```

📝 **Note**: 
- `COPILOT_TOKEN`: Required for authentication. If not set, a random token will be generated.
- `COPILOT_SERVER_PORT`: Optional. Controls which port the server listens on.
- `COPILOT_SERVER_WORKERS`: Optional. Controls the number of worker processes.

2. Configure GitHub Copilot:
   - Windows users: Check `%LOCALAPPDATA%/github-copilot/`
   - Unix/Linux/macOS users: Check `~/.config/github-copilot/`

Required configuration files:
- `apps.json` or `hosts.json` (containing GitHub OAuth token)
- `token.json` (will be created automatically)

> **💡 How to get a valid Github Copilot configuration?**
>
> Choose any of these official GitHub Copilot plugins:
>
> - [GitHub Copilot for VS Code](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)
> - [GitHub Copilot for Visual Studio](https://marketplace.visualstudio.com/items?itemName=GitHub.copilotvs)
> - [GitHub Copilot for Vim](https://github.com/github/copilot.vim)
> - [GitHub Copilot for JetBrains IDEs](https://plugins.jetbrains.com/plugin/17718-github-copilot)
>
> After installing and signing in, configuration files will be automatically created in your system's config directory.

## 🚀 Usage

Choose between local or Docker deployment:

### 🖥️ Local Run

Start the server with:
```bash
pdm dev
```

### 🐳 Docker Run

Launch the containerized version:
```bash
# Unix/Linux/macOS
docker run --rm -p 9191:9191 \
    -v ~/.config/github-copilot:/home/appuser/.config/github-copilot \
    ghcr.io/yuchanns/copilot-openai-api

# Windows
docker run --rm -p 9191:9191 ^
    -v %LOCALAPPDATA%/github-copilot:/home/appuser/.config/github-copilot ^
    ghcr.io/yuchanns/copilot-openai-api
```

The Docker setup:
- Maps port 9191 to your host
- Mounts your Copilot configuration
- Provides identical functionality to local deployment

### 🔄 Making API Requests

Access the chat completion endpoint:
```bash
curl -X POST http://localhost:9191/v1/chat/completions \
  -H "Authorization: Bearer your_access_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, Copilot!"}]
  }'
```

Access the embeddings endpoint:
```bash
curl -X POST http://localhost:9191/v1/embeddings \
  -H "Authorization: Bearer your_access_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "copilot-text-embedding-ada-002",
    "input": ["The quick brown fox", "Jumped over the lazy dog"]
  }'
```

## 🔌 API Reference

### POST /v1/chat/completions
Proxies requests to GitHub Copilot's Completions API.

**Required Headers:**
- `Authorization: Bearer <your_access_token>`
- `Content-Type: application/json`

**Request Body:**
- Follow GitHub Copilot chat completion API format

**Response:**
- Streams responses directly from GitHub Copilot's API
- Supports both streaming and non-streaming modes

---

### POST /v1/embeddings
Proxies requests to Github Copilot's Embeddings API.

**Required Headers:**
- `Authorization: Bearer <your_access_token>`
- `Content-Type: application/json`

**Request Body:**
- Follow GitHub Copilot embeddings API format

**Response:**
- JSON response from GitHub Copilot's embeddings API

## 🔒 Authentication

Secure your endpoints:

1. Set `COPILOT_TOKEN` in your environment
2. Include in request headers:
   ```
   Authorization: Bearer your_access_token_here
   ```

## ⚠️ Error Handling

The server provides clear error responses:
- 401: Missing/invalid authorization header
- 403: Invalid access token
- Other errors are propagated from GitHub Copilot API

## 🛡️ Security Best Practices

- Configure CORS for your specific domains (default: `*`)
- Safeguard your `COPILOT_TOKEN` and GitHub OAuth token
- Built-in token management with concurrent access protection

## 📄 License

Licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
