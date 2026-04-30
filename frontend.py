#!/usr/bin/env python
"""Terminal chatbot con Llama.cpp — Interfaccia utente e chat loop."""

from __future__ import annotations

import asyncio
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import httpx
import inquirer

import msvcrt

import json
import yaml
import re

from backend import (
    check_server,
    fetch_models,
    start_server,
)

from tools import (
    set_debug_mode,
    _read_impl,
    _bash_impl,
    _format_tool_result,
    _tool_debug,
)

# ── Debug flag ──────────────────────────────────────────────────────────────
_DEBUG = False

def set_frontend_debug(mode: bool):
    global _DEBUG
    _DEBUG = mode

def _debug_print(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs)

def _debug_print_stderr(*args, **kwargs):
    if _DEBUG:
        print(*args, file=sys.stderr, **kwargs)

# ── Config ──────────────────────────────────────────────────────────────────
_CONFIG_DIR = Path(__file__).parent / ".config"
_CONFIG_FILE = _CONFIG_DIR / "status.yaml"
_LLAMA_BASE_URL = "http://127.0.0.1:8080"


def _load_config() -> dict:
    """Load config from YAML file."""
    if _CONFIG_FILE.exists():
        try:
            return yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}


def _save_config(config: dict):
    """Save config to YAML file."""
    import datetime
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config["last_updated"] = datetime.datetime.now().isoformat()
    _CONFIG_FILE.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")


async def _llama_chat_with_data(messages: list, model_id: str) -> dict:
    """Chiama llama.cpp e restituisce dati completi (incluso tool_calls)."""
    from tools import _SYSTEM_PROMPTS

    headers = {"Content-Type": "application/json"}

    # Scegli system prompt in base al modello
    if "FunctionCall" in model_id or "LFM" in model_id:
        system_prompt = _SYSTEM_PROMPTS.get("lfmfunctioncall", "")
    else:
        system_prompt = _SYSTEM_PROMPTS.get("default", "")

    # Debug: mostra system prompt per gemma
    if _DEBUG and model_id == "gemma-4-E2B-it-Q4_K_M":
        _debug_print(f"\n[DEBUG GEMMA] System prompt being used:")
        _debug_print(f"'{system_prompt[:500]}...'")
        _debug_print(f"[DEBUG GEMMA] Number of messages sent: {len(messages)}")
        import json
        _debug_print(f"[DEBUG GEMMA] Last message: {json.dumps(messages[-1], indent=2, ensure_ascii=False)}")
        _debug_print("[END DEBUG GEMMA]\n")

    # Definizione del tool read
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read the content of a file. Supports offset/limit for pagination.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        },
                        "offset": {
                            "type": "number",
                            "description": "Line number to start reading from (1-indexed)"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of lines to read"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command in the current working directory. Returns stdout and stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Bash command to execute"
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (optional)"
                        }
                    },
                    "required": ["command"]
                }
            }
        }
    ]

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)
    payload = {
        "model": model_id,
        "messages": full_messages,
        "tools": tools,
    }

    # Disabilita thinking per gemma (stessa logica di tools.py)
    model_lower = model_id.lower().replace("_", "-")
    if "gemma-" in model_lower or model_lower.startswith("gemma") and any(c.isdigit() for c in model_lower):
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{_LLAMA_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()


async def _llama_chat_stream(messages: list, model_id: str, tool_calls_collector: list = None):
    """Stream della risposta da llama.cpp, yielda i token di contenuto."""
    from tools import _SYSTEM_PROMPTS

    headers = {"Content-Type": "application/json"}

    if "FunctionCall" in model_id or "LFM" in model_id:
        system_prompt = _SYSTEM_PROMPTS.get("lfmfunctioncall", "")
    else:
        system_prompt = _SYSTEM_PROMPTS.get("default", "")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read the content of a file. Supports offset/limit for pagination.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"},
                        "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                        "limit": {"type": "number", "description": "Maximum number of lines to read"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command in the current working directory. Returns stdout and stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Bash command to execute"},
                        "timeout": {"type": "number", "description": "Timeout in seconds (optional)"}
                    },
                    "required": ["command"]
                }
            }
        }
    ]

    # Prepara messaggi con system prompt
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    payload = {
        "model": model_id,
        "messages": full_messages,
        "tools": tools,
        "stream": True,
    }

    # Disabilita thinking per gemma (stessa logica di tools.py)
    model_lower = model_id.lower().replace("_", "-")
    if "gemma-" in model_lower or model_lower.startswith("gemma") and any(c.isdigit() for c in model_lower):
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{_LLAMA_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                            # Collect tool calls if collector is provided
                            if tool_calls_collector is not None:
                                tool_calls_delta = delta.get("tool_calls", [])
                                for tc_delta in tool_calls_delta:
                                    tc_index = tc_delta.get("index", 0)
                                    tc_id = tc_delta.get("id")
                                    func_delta = tc_delta.get("function") or {}

                                    # Ensure collector is large enough
                                    while len(tool_calls_collector) <= tc_index:
                                        tool_calls_collector.append(None)

                                    existing_tc = tool_calls_collector[tc_index]

                                    if existing_tc is None:
                                        # New tool call - initialize
                                        args = func_delta.get("arguments", "")
                                        existing_tc = {
                                            "id": tc_id,
                                            "type": tc_delta.get("type", "function"),
                                            "function": {
                                                "name": func_delta.get("name", ""),
                                                "arguments": args
                                            }
                                        }
                                        tool_calls_collector[tc_index] = existing_tc
                                        if _DEBUG:
                                            _debug_print_stderr(f"[DEBUG STREAM] New TC[{tc_index}]: id={tc_id}, name='{func_delta.get('name', '')}', args='{args}'")
                                    else:
                                        # Update existing tool call with new chunks
                                        if tc_id:
                                            existing_tc["id"] = tc_id
                                        if func_delta.get("name"):
                                            existing_tc["function"]["name"] = func_delta["name"]
                                        new_args = func_delta.get("arguments", "")
                                        if new_args:
                                            old_args = existing_tc["function"]["arguments"]
                                            existing_tc["function"]["arguments"] = old_args + new_args
                                            if _DEBUG:
                                                _debug_print_stderr(f"[DEBUG STREAM] TC[{tc_index}] args: '{old_args}' + '{new_args}' => '{existing_tc['function']['arguments']}'")
                    except json.JSONDecodeError:
                        pass


# ── Configurazione visuale ───────────────────────────────────────────────────
_PADDING_H = 4   # spazi a sinistra
_PADDING_RIGHT = 10  # spazio riservato a destra (causa wrap anticipato)


def _get_terminal_width() -> int:
    """Ritorna larghezza terminale, default 80 se non disponibile."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def _pad_line(text: str, right: bool = False, width: int = None) -> str:
    """Aggiunge padding a sinistra e opzionalmente a destra."""
    result = " " * _PADDING_H + text
    if right and width:
        result = result.ljust(width)
    return result


def _wrap_and_pad(text: str) -> str:
    """Wrap del testo alla larghezza del terminale e applica padding a ogni riga."""
    if not text:
        return ""
    term_width = _get_terminal_width()
    # Larghezza disponibile (escluso padding sx e riservato dx)
    available = term_width - _PADDING_H - _PADDING_RIGHT
    if available <= 0:
        return _pad_line(text)

    lines = text.split('\n')
    result = []
    for line in lines:
        if len(line) <= available:
            result.append(line)
        else:
            # Wrap manuale
            words = line.split(' ')
            current = ""
            for word in words:
                if not current:
                    current = word
                elif len(current) + 1 + len(word) <= available:
                    current += " " + word
                else:
                    result.append(current)
                    current = word
            if current:
                result.append(current)
    # Applica padding a ogni riga (dx e sx)
    return '\n'.join(_pad_line(line, right=True, width=term_width) for line in result)


def pprint(msg: str = '', *, end: str = '\n', pad: bool = True, pad_right: bool = False):
    """Print con padding a sinistra."""
    if pad:
        print(_pad_line(msg, pad_right), end=end)
    else:
        print(msg, end=end)


# ── ESC watchdog (thread separato, polling msvcrt) ────────────────────────────

_esc_pressed = threading.Event()


def _esc_watchdog():
    """Polla msvcrt in thread separato. Quando rileva ESC, setta _esc_pressed."""
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch == b'\x1b':
                _esc_pressed.set()
                break
        time.sleep(0.02)


def select_model(models: list[dict]) -> str:
    """Mostra un menu interattivo e restituisce il model_id scelto."""
    label_to_id: dict[str, str] = {}
    choices = []
    default_label = None

    for m in models:
        status = m.get("status", {})
        model_id = m["id"]
        if status.get("value") == "loaded":
            label = f"⭐ {model_id}  [loaded]"
            default_label = label
        else:
            label = f"  {model_id}"
        choices.append(label)
        label_to_id[label] = model_id

    questions = [
        inquirer.List(
            "model",
            message="Seleziona il modello (↑↓ per navigare, Enter per confermare)",
            choices=choices,
            carousel=True,
            default=default_label,
        )
    ]
    answer = inquirer.prompt(questions)
    if answer is None:
        sys.exit(0)
    while msvcrt.kbhit():
        msvcrt.getch()
    return label_to_id[answer["model"]]


async def run_chatbot_async(model_id: str):
    """Chatbot async: streaming interruptible — ESC abortisce l'inferenza."""
    try:
        models = fetch_models()
        model_id = select_model(models)
    except Exception as e:
        pprint(f"❌ Errore nel caricamento modelli: {e}")
        return

    print(f"\n🔄 Modello: {model_id}")
    pprint("💬 Chat avviata — ESC abortisce l'inferenza, /quit per uscire, /model per cambiare modello")
    pprint("─" * 50, pad=False)
    while msvcrt.kbhit():
        msvcrt.getch()

    messages: list = []

    while True:
        user_input = input(_pad_line("👤 Tu: ")).strip()
        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            pprint("👋 Arrivederci!")
            break
        if user_input.lower() == "/model":
            try:
                models = fetch_models()
                new_model_id = select_model(models)
                model_id = new_model_id
                # Persist model selection
                config = _load_config()
                config["model"] = model_id
                _save_config(config)
                messages = []
                print(f"\n✅ Modello cambiato a: {model_id}")
                pprint("   (verrà caricato automaticamente alla prima richiesta)")
            except Exception as e:
                print(f"\n❌ Errore nel cambio modello: {e}")
            continue

        _esc_pressed.clear()
        watchdog = threading.Thread(target=_esc_watchdog, daemon=True)
        watchdog.start()

        current_model_id = model_id

        try:
            # Aggiungi messaggio utente alla cronologia
            messages.append({"role": "user", "content": user_input})
            _debug_print_stderr(f"[DEBUG] User message added. Total messages: {len(messages)}")
            _debug_print_stderr(f"[DEBUG] User input: '{user_input}'")

            # Ciclo per gestire tool calls
            loop_count = 0
            while True:
                loop_count += 1
                _debug_print_stderr(f"\n[DEBUG] Loop {loop_count}, model: {current_model_id}")
                if _DEBUG:
                    import json
                    _debug_print_stderr("[DEBUG GEMMA] Messages being sent (last 3):")
                    for i, msg in enumerate(messages[-3:]):
                        _debug_print_stderr(f"  [{len(messages)-3+i}] {json.dumps(msg, ensure_ascii=False)[:8000]}")
                tool_calls_collected = []
                print("🤖 Assistant: ", end="", flush=True)
                full_content = ""
                try:
                    async for token in _llama_chat_stream(messages, current_model_id, tool_calls_collector=tool_calls_collected):
                        if _esc_pressed.is_set():
                            print("\n[Interrotto]")
                            break
                        print(token, end="", flush=True)
                        full_content += token
                    print()
                except Exception as e:
                    print(f"\n❌ Errore durante lo streaming: {e}")
                    break
                if _esc_pressed.is_set():
                    break

                # Usa i dati raccolti dallo streaming
                content = full_content.strip()
                tool_calls = tool_calls_collected
               
                if tool_calls:
                    # Esegui tool call
                    for tool_call in tool_calls:
                        func = tool_call.get("function", {})
                        if func.get("name") == "read":
                            import json
                            args = func.get("arguments", {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            file_path = args.get("path", "")
                            _tool_debug("read", {"path": file_path})
                            tool_result = _read_impl(file_path, args.get("offset"), args.get("limit"))

                            if _DEBUG and "gemma" in current_model_id.lower():
                                _debug_print_stderr("\n[DEBUG GEMMA] Tool 'read' executed:")
                                _debug_print_stderr(f"  Args: {args}")
                                _debug_print_stderr(f"  Raw result: {tool_result}")

                            try:
                                parsed = json.loads(tool_result)
                                result_content = parsed.get("content", tool_result)
                            except:
                                result_content = tool_result
                            result_content = re.sub(r'\s+', ' ', result_content).strip()
                            if _DEBUG and "gemma" in current_model_id.lower():
                                _debug_print_stderr(f"  Parsed content: {result_content[:200]}")
                                _debug_print_stderr("[END DEBUG GEMMA]\n")

                            rebuilt_tool_calls = [{
                                "id": tool_call.get("id"),
                                "type": "function",
                                "function": {
                                    "name": func.get("name"),
                                    "arguments": json.dumps(args)
                                }
                            }]
                            messages.append({
                                "role": "assistant",
                                "content": content or None,
                                "tool_calls": rebuilt_tool_calls
                            })
                            messages.append({
                                "role": "tool",
                                "name": func.get("name"),
                                "tool_call_id": tool_call.get("id"),
                                "content": result_content or None
                            })

                            if _DEBUG and "gemma" in current_model_id.lower():
                                import json
                                _debug_print_stderr("[DEBUG GEMMA] Messages sent back to model after 'read':")
                                _debug_print_stderr(json.dumps(messages[-4:], indent=2, ensure_ascii=False))
                                _debug_print_stderr(f"[DEBUG GEMMA] result_content was: '{result_content[:200]}'")

                            break

                        elif func.get("name") == "bash":
                            import json
                            args = func.get("arguments", {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            command = args.get("command", "")
                            _tool_debug("bash", {"command": command})
                            tool_result = _bash_impl(command, args.get("timeout"))

                            if _DEBUG and "gemma" in current_model_id.lower():
                                _debug_print_stderr("\n[DEBUG GEMMA] Tool 'bash' executed:")
                                _debug_print_stderr(f"  Args: {args}")
                                _debug_print_stderr(f"  Command: {command}")
                                _debug_print_stderr(f"  Raw result: {tool_result}")

                            try:
                                parsed = json.loads(tool_result)
                                result_content = parsed.get("content", tool_result)
                            except:
                                result_content = tool_result
                            result_content = re.sub(r'\s+', ' ', result_content).strip()
                            if _DEBUG and "gemma" in current_model_id.lower():
                                _debug_print_stderr(f"  Parsed content: {result_content[:200]}")
                                _debug_print_stderr("[END DEBUG GEMMA]\n")

                            rebuilt_tool_calls = [{
                                "id": tool_call.get("id"),
                                "type": "function",
                                "function": {
                                    "name": func.get("name"),
                                    "arguments": json.dumps(args)
                                }
                            }]
                            messages.append({
                                "role": "assistant",
                                "content": content if content else "Then I will answer and tell you any content...",
                                "tool_calls": rebuilt_tool_calls
                            })
                            messages.append({
                                "role": "tool",
                                "name": func.get("name"),
                                "tool_call_id": tool_call.get("id"),
                                "content": result_content or None
                            })

                            if _DEBUG and "gemma" in current_model_id.lower():
                                import json
                                _debug_print_stderr("[DEBUG GEMMA] Messages sent back to model after 'bash':")
                                _debug_print_stderr(json.dumps(messages[-4:], indent=2, ensure_ascii=False))
                                _debug_print_stderr(f"[DEBUG GEMMA] result_content was: '{result_content[:200]}'")

                            break
                    continue

                else:
                    # Nessun tool call, già mostrato via streaming
                    messages.append({"role": "assistant", "content": full_content})
                    break

            pprint()
            sys.stdout.flush()
            while msvcrt.kbhit():
                msvcrt.getch()
        except Exception as e:
            pprint(f"❌ Errore: {e}")

        _esc_pressed.clear()


def main():
    parser = argparse.ArgumentParser(description="Terminal Chatbot with Llama.cpp")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    if args.debug:
        set_debug_mode(True)  # tools.py
        set_frontend_debug(True)  # frontend.py

    os.system('cls' if os.name == 'nt' else 'clear')

    pprint("╔══════════════════════════════════════════╗")
    pprint("║  Terminal Chatbot — Llama.cpp Selector   ║")
    pprint("╚══════════════════════════════════════════╝")
    pprint()

    if not check_server():
        start_server()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    _orig_handler = loop.get_exception_handler()

    def _silent_handler(loop, context):
        exc = context.get("exception")
        if exc is not None and isinstance(
            exc, (asyncio.CancelledError, httpx.ReadError, httpx.ConnectError)
        ):
            return
        if _orig_handler:
            _orig_handler(loop, context)

    loop.set_exception_handler(_silent_handler)
    try:
        loop.run_until_complete(run_chatbot_async(""))
    finally:
        loop.close()


if __name__ == "__main__":
    main()

