#!/usr/bin/env python
"""Tool implementations for Llama.cpp chatbot."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Optional

import yaml
# Lock per sincronizzare output
_spinner_lock = threading.Lock()

# Tools definition per llama.cpp - usato da frontend.py
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

def _load_system_prompts() -> dict:
    """Carica i system prompts dal file di configurazione."""
    config_path = Path(__file__).parent / ".config" / "system_prompts.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Config file not found: {config_path}")


_SYSTEM_PROMPTS = _load_system_prompts()


_DEBUG_MODE = False


def set_debug_mode(enabled: bool) -> None:
    """Abilita/disabilita il debug mode."""
    global _DEBUG_MODE
    _DEBUG_MODE = enabled


def _debug_log(message: str) -> None:
    """Stampa messaggio di debug se il debug mode è abilitato."""
    if _DEBUG_MODE:
        print(f"\n[DEBUG] {message}", flush=True)


# =============================================================================
# PROTOCOL DIFFERENCES NOTE:
# LFM models (LFM2/LFM2.5 family) expect tool results as JSON:
#   {"content": "...", "error": "...", "success": true, ...}
# Other models (Gemma, Qwen, etc.) expect plain text:
#   "File content here"
# The _format_tool_result() helper handles this distinction automatically.
# See: https://docs.liquid.ai/lfm/key-concepts/tool-use
# =============================================================================


def _is_functioncall_model(model_id: str) -> bool:
    """Verifica se il modello è un derivato FunctionCall."""
    return "FunctionCall" in model_id


def _is_lfm_model(model_id: str) -> bool:
    """Verifica se il modello è della famiglia LFM (esclusi FunctionCall)."""
    return "LFM" in model_id and "FunctionCall" not in model_id


def _format_tool_result(result: str, model_id: str) -> str:
    """Formatta il risultato del tool in base al tipo di modello.

    LFM vuole JSON {"content": "..."}, altri modelli preferiscono plain text.
    NEVER change this behavior without updating the protocol note above.
    """
    if _is_lfm_model(model_id):
        # Per LFM, gestiamo i caratteri problematici per llama.cpp:
        # - [digit] -> (digit) per evitare confusione con array
        # - \n\n può confondere il parser del chat template
        try:
            parsed = json.loads(result)
            if "content" in parsed and isinstance(parsed["content"], str):
                import re
                # Sostituisci [digit] con (digit)
                content = re.sub(r"\[(\d+)\]", r"(\1)", parsed["content"])
                # Sostituisci \n\n con \n per evitare problemi di parsing
                content = content.replace("\n\n", "\n")
                parsed["content"] = content
                return json.dumps(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
        return result
    try:
        parsed = json.loads(result)
        if "content" in parsed:
            return parsed["content"]
        if "success" in parsed and parsed.get("success"):
            return f"File saved to: {parsed.get('path', 'unknown')}"
        if "entries" in parsed:
            return "\n".join(f"- {e['type']}: {e['name']}" for e in parsed["entries"])
        if "exists" in parsed:
            return f"{'Path exists' if parsed['exists'] else 'Path does not exist'} ({parsed.get('type', 'unknown')})"
        if "error" in parsed:
            return f"Error: {parsed['error']}"
        if "exit_code" in parsed:
            exit_code = parsed["exit_code"]
            content = parsed.get("content", "")
            if exit_code != 0:
                return f"[Exit code: {exit_code}]\n{content}"
            return content
        return result
    except (json.JSONDecodeError, TypeError):
        return result


def _normalize_path(path: str) -> str:
    """Normalizza path per usare forward slashes."""
    return path.replace("\\", "/")


def _tool_debug(tool_name: str, args: dict):
    """Stampa quando un tool viene chiamato - SEMPRE visibile all'utente."""
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"\n[TOOL] {tool_name}({args_str})", flush=True)


def _read_impl(path: str, offset: int = None, limit: int = None) -> str:
    """Implementazione di read_file."""
    _debug_log(f"read_file called with path='{path}', offset={offset}, limit={limit}")
    if not path:
        return json.dumps({"error": "missing path parameter"})
    try:
        normalized = _normalize_path(path)
        file_path = Path(normalized).resolve()
        _debug_log(f"normalized='{normalized}', resolved='{file_path}'")
        if not file_path.is_absolute():
            file_path = Path.cwd() / path
            file_path = file_path.resolve()
            _debug_log(f"cwd resolved='{file_path}'")

        if not file_path.exists():
            _debug_log("FILE NOT FOUND")
            return json.dumps({"error": f"File not found: {file_path}"})

        content = file_path.read_text(encoding="utf-8")
        _debug_log(f"read {len(content)} chars")

        # Apply offset/limit for pagination (offset is 1-indexed)
        if offset is not None or limit is not None:
            all_lines = content.splitlines()
            start = (offset - 1) if offset is not None else 0  # Convert 1-indexed to 0-indexed

            # Validate offset
            if offset is not None:
                if offset < 1:
                    return json.dumps({"error": f"Offset must be >= 1, got {offset}"})
                if offset > len(all_lines):
                    return json.dumps({"error": f"Offset {offset} is beyond file content (file has {len(all_lines)} line(s))"})

            if limit is not None:
                end = start + limit
            else:
                end = None
            content = "\n".join(all_lines[start:end])
            _debug_log(f"slice [{start}:{end}] -> {len(content)} chars")

        return json.dumps({"content": content})
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"})
    except Exception as e:
        _debug_log(f"EXCEPTION: {e}")
        return json.dumps({"error": str(e)})


def _bash_impl(command: str, timeout: float = None) -> str:
    """Implementazione di bash tool."""
    _debug_log(f"bash called with command='{command}', timeout={timeout}")
    if not command:
        return json.dumps({"error": "missing command parameter"})

    try:
        if os.name == "nt":
            # Su Windows, usa Git Bash
            git_bash_path = r"C:\Program Files\git\bin\bash.exe"
            result = subprocess.run(
                [git_bash_path, "-c", command],
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=None
            )
        else:
            # Su Linux/Mac, usa la shell di sistema
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=None
            )

        # Combina stdout e stderr
        output = result.stdout
        if result.stderr:
            if output:
                output += "\n" + result.stderr
            else:
                output = result.stderr

        # Truncamento a 2000 linee o 50KB
        if output:
            lines = output.splitlines()
            if len(lines) > 2000:
                output = "\n".join(lines[-2000:])
            if len(output) > 50 * 1024:
                output = output[:50 * 1024]

        return json.dumps({
            "content": output or "",
            "exit_code": result.returncode
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout}s"})
    except FileNotFoundError:
        return json.dumps({"error": f"Command not found: {command}"})
    except Exception as e:
        _debug_log(f"BASH EXCEPTION: {e}")
        return json.dumps({"error": str(e)})


def get_tools_definition() -> list:
    """Restituisce la definizione dei tool per llama.cpp."""
    return [
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


if __name__ == "__main__":
    # Enable debug mode for testing
    set_debug_mode(True)
    
    print("=== Testing tools.py functions ===\n")

    
    # Test _read_impl (reads this file itself)
    print("5. Test _read_impl:")
    self_path = str(Path(__file__).resolve())
    read_result = _read_impl("colosseo.txt", offset=2, limit=2)
    print(f"{read_result}\n")
    
    # Test _bash_impl
    print("6. Test _bash_impl:")
    bash_result = _bash_impl("ls -la")
    print(f"   Result: {bash_result}\n")
    
    print("=== Tests complete ===")
