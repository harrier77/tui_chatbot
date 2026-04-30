#!/usr/bin/env python
"""Backend per chatbot - Llama.cpp server management."""

from __future__ import annotations

import os
import subprocess
import threading
import time

import httpx

# ── Config ──────────────────────────────────────────────────────────────────
LLAMA_BASE_URL = "http://127.0.0.1:8080"
OPENAI_API_KEY = "not-needed"  # llama.cpp non richiede auth

os.environ["OPENAI_BASE_URL"] = f"{LLAMA_BASE_URL}/v1"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# ─────────────────────────────────────────────────────────────────────────────


def check_server() -> bool:
    """Verifica se il server llama è attivo."""
    try:
        resp = httpx.get(f"{LLAMA_BASE_URL}/v1/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def start_server() -> None:
    """Avvia il server llama.cpp in un processo subprocess."""
    print("🔄 Server non attivo, avvio llama-server.exe...")
    print("   ( questo potrebbe richiedere qualche secondo )\n")
    subprocess.Popen(
        ["cmd.exe", "/k", r"llama-server.exe --host 0.0.0.0 --models-preset config.ini --models-max 1 --no-warmup --parallel 1 --jinja"],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        cwd=r"C:\down\llama-latest",
    )
    for _ in range(30):
        time.sleep(1)
        if check_server():
            print("✅ Server avviato e pronto!\n")
            return
    raise RuntimeError("Impossibile avviare il server llama")


def load_model(model_id: str) -> None:
    """Carica esplicitamente un modello sul server router."""
    print(f"   [load_model] chiamato con model_id={model_id}")
    try:
        resp = httpx.post(
            f"{LLAMA_BASE_URL}/models/load",
            json={"model": model_id},
            timeout=60
        )
        print(f"   [load_model] response: {resp.status_code} - {resp.text}")
        resp.raise_for_status()
    except Exception as e:
        print(f"   [load_model] errore: {e}")
        pass


def fetch_models() -> list[dict]:
    """Recupera la lista dei modelli dal server llama."""
    resp = httpx.get(f"{LLAMA_BASE_URL}/v1/models", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]


def get_available_models() -> list[str]:
    """Recupera i modelli disponibili dal server."""
    try:
        models = fetch_models()
        return [m["id"] for m in models]
    except Exception as e:
        print(f"Errore nel recupero modelli: {e}")
        return []


