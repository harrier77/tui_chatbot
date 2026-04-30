#!/usr/bin/env python
"""Test program per il protocollo llama.cpp - verifica tool calling con 'Read file'."""

from __future__ import annotations

import sys
import json
import httpx

LLAMA_BASE_URL = "http://127.0.0.1:8080"


def check_server() -> bool:
    """Verifica se il server llama è attivo."""
    try:
        resp = httpx.get(f"{LLAMA_BASE_URL}/v1/models", timeout=5)
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Server non disponibile: {e}")
        return False


def read_file_tool(file_path: str) -> dict:
    """Esegue il tool read - legge il contenuto di un file."""
    try:
        content = open(file_path, "r", encoding="utf-8").read()
        return {"success": True, "content": content}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {file_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_protocol(user_message: str = "Read file ./prova.txt", system_prompt: str = None):
    """Test del protocollo con tool calling."""
    print(f"\n🔧 Test protocollo (modello già caricato su server)")
    print(f"   Messaggio: {user_message}")
    
    # System prompt per convincere il modello a usare i tool
    if system_prompt is None:
        system_prompt = """You have access to tools. When user asks to read a file, use the 'read' tool and then show the content"""
    
    # 2. Prepara messaggi con tool definition
    messages = [{"role": "user", "content": user_message}]
    
    # Tool definition per 'read'
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read the content of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file path to read"
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    ]
    
    payload = {
        "messages": messages,
        "tools": tools,
        "system": system_prompt,
    }
    
    # 3. Prima chiamata - ricevi tool_call
    print("\n📤 Invio richiesta al server...")
    #print(f"   payload: {json.dumps(payload, indent=2)[:500]}...")
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{LLAMA_BASE_URL}/v1/chat/completions",
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"❌ Errore nella chiamata: {e}")
        return
    
    # 4. Analizza risposta - RAW
    print("\n📥 RAW RISPOSTA SERVER (prima chiamata):")
    print("=" * 60)
    #print(json.dumps(data, indent=2))
    print (json.dumps(data['model'],indent=2))
    print(f"campo messaggio:")
    print (json.dumps(data['choices'][0]['message']['content'],indent=2))
    try:
        print (json.dumps(data['choices'][0]['message']['tool_calls'],indent=2))
    except Exception as e:
        print(f"Errore, il campo tool_calls non esiste: {e}")
    print("=" * 60)
    
    if "choices" not in data or len(data["choices"]) == 0:
        print("❌ Nessuna risposta dal server")
        return
    
    choice = data["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls", [])
    
    # 5. Se c'è tool_call, eseguilo
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name")
            args = func.get("arguments", {})
            
            if isinstance(args, str):
                args = json.loads(args)
            
            #print(f"\n🔨 Esecuzione tool: {name}")
            #print(f"   args: {args}")
            #print(f"\n📥 RAW tool_calls received:")
            #print(json.dumps(tool_calls, indent=2))
            
            if name == "read":
                file_path = args.get("path", "")
                result = read_file_tool(file_path)
                #print(f"   risultato: {result}")
                
                # 6. Seconda chiamata - risposta del tool
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [{
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args)
                        }
                    }]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": json.dumps(result)
                })
                
                # Richiesta risposta finale
                payload = {
                    "messages": messages,
                }
                
                #print("\n📤 Invio risposta tool...")
                #print(f"   payload: {json.dumps(payload, indent=2)[:500]}...")
                try:
                    with httpx.Client(timeout=60) as client:
                        resp = client.post(
                            f"{LLAMA_BASE_URL}/v1/chat/completions",
                            json=payload
                        )
                        resp.raise_for_status()
                        data = resp.json()
                except Exception as e:
                    print(f"❌ Errore seconda chiamata: {e}")
                    return
                
                #print("\n📥 RAW RISPOSTA SERVER (seconda chiamata - con tool result):")
                #print("=" * 60)
                #print(json.dumps(data, indent=2))
                #print("=" * 60)
                
                if "choices" in data and len(data["choices"]) > 0:
                    final_msg = data["choices"][0].get("message", {})
                    final_content = final_msg.get("content", "")
                    
                    #print("\n✅ RISPOSTA FINALE:")
                    #print("─" * 40)
                    #print(final_content)
                    #print("─" * 40)
    else:
        # Nessun tool call - mostra risposta diretta
        print("\n✅ RISPOSTA DIRETTA:")
        print("─" * 40)
        print(content)
        print("─" * 40)


def main():
    """Entry point."""
    user_message = "Read file prova.txt"
    
    if len(sys.argv) > 1:
        user_message = " ".join(sys.argv[1:])
    
    if not check_server():
        print("\n❌ Server non attivo su porta 8080")
        print("   Avviare llama-server.exe prima di eseguire il test")
        sys.exit(1)
    
    test_protocol(user_message)


if __name__ == "__main__":
    main()