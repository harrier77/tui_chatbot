# Terminal Chatbot with Llama.cpp

A terminal-based AI chatbot powered by Llama.cpp with tool-calling capabilities.

## Features

- **Terminal UI** — Interactive chat with model selection via inquirer
- **Multiple Model Support** — LFM, LFMFunctionCall, Gemma, Qwen, and other OpenAI-compatible models via Llama.cpp
- **Tool Calling** — `read` (file operations with pagination), `bash` (Windows command execution via Git Bash)
- **Streaming Responses** — Real-time token-by-token output with interrupt support (ESC to abort)
- **Model Switching** — Switch models mid-session with `/model` command (uses inquirer interactive menu)
- **Protocol Handling** — Automatic handling of LFM vs standard OpenAI tool protocols
- **Config Persistence** — Remembers selected model in `.config/status.yaml`

## Requirements

- Python 3.11+
- [llama-server.exe](https://github.com/ggerganov/llama.cpp) (place in `C:\down\llama-latest`)
- Git Bash (for `bash` tool on Windows)

### Python Dependencies

```
pip install httpx inquirer pyyaml
```

## Usage

### Terminal UI (main)
```bash
python frontend.py
```

### Commands

| Command | Description |
|---------|-------------|
| `/quit`, `/exit`, or `/q` | Exit the chatbot |
| `/model` | Switch to a different model (uses interactive inquirer menu) |
| `ESC` | Abort current inference |

### Available Tools

| Tool | Description |
|------|-------------|
| `read` | Read file contents with optional offset/limit pagination |
| `bash` | Execute shell commands via Git Bash (`C:\Program Files\git\bin\bash.exe`) |

## Architecture

```
.
├── frontend.py        # TUI - Terminal chat loop, ESC watchdog (msvcrt), model selection
├── backend.py         # Llama.cpp server management (start, check, load models)
├── tools.py           # Tool implementations (read, bash), protocol handling
├── test_protocol.py   # Protocol testing utility
└── .config/
    ├── status.yaml    # Model persistence
    └── system_prompts.yaml  # System prompts per model type (lfm, lfmfunctioncall, default)
```

## Configuration

### Llama.cpp Server
Edit `backend.py` to change:
- `LLAMA_BASE_URL` — Server address (default: `http://127.0.0.1:8080`)
- Model paths and server launch parameters (launches with `--models-preset config.ini --models-max 1 --no-warmup --parallel 1 --jinja`)

### System Prompts
Edit `.config/system_prompts.yaml` to customize prompts for different model types:
- `lfm` — For LFM models
- `lfmfunctioncall` — For LFMFunctionCall models
- `default` — For all other models (Gemma, Qwen, etc.)

### Model-Specific Notes
- **LFM models**: Require JSON tool results format `{"content": "..."}`
- **LFMFunctionCall models**: Use LFM function call protocol
- **Gemma models**: Thinking is disabled via `chat_template_kwargs`
- **Windows**: Uses `msvcrt` for ESC detection, Git Bash for `bash` tool

## Debug Mode

Enable debug output by setting `_DEBUG = True` in `frontend.py` and calling `set_debug_mode(True)` in `tools.py`.
