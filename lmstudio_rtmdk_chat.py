"""
lmstudio_rtmdk_chat.py
Чат с LM Studio + RTMDK v2 память.

Требования:
  1. LM Studio запущен на http://localhost:12345
  2. Модель загружена в LM Studio
  3. pip install requests numpy scipy pydantic

Запуск:
  python lmstudio_rtmdk_chat.py

Команды в чате:
  /stats   - статистика памяти (включая TDA, проекцию, самосупервизию)
  /export  - экспорт состояния поля
  /clear   - очистить память
  /format json|yaml|plain - формат контекста
  /session <id> - переключить сессию
  /quit    - выйти
"""

import sys
import os
import json
import time
import requests
import numpy as np
from rtmdk_memory_v5 import RTMDKConfig, RTMDKMemory, ContextFormat

LM_STUDIO_URL = "http://localhost:12345/v1"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"
CHAT_MODEL = None
MEMORY_FILE = "rtmdk_lmstudio_state.json"


def get_chat_model():
    global CHAT_MODEL
    if CHAT_MODEL:
        return CHAT_MODEL
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)
        models = resp.json().get("data", [])
        if models:
            CHAT_MODEL = models[0]["id"]
            return CHAT_MODEL
    except Exception:
        pass
    return None


def get_embedding(text: str) -> np.ndarray:
    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/embeddings",
            json={"model": EMBED_MODEL, "input": text},
            timeout=30,
        )
        data = resp.json()
        embedding = data["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"\n  [WARN] Embedding error: {e}")
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(768).astype(np.float32) * 0.1


def chat_with_memory(memory: RTMDKMemory, user_input: str, session_id: str = "default") -> str:
    ctx = memory.load_memory_variables({"input": user_input, "session_id": session_id})

    system_prompt = "You are a helpful assistant with long-term memory."
    if memory.config.context_format == ContextFormat.JSON:
        system_prompt += "\n\nRelevant memories (JSON format, higher resonance = more relevant):"
    elif memory.config.context_format == ContextFormat.YAML:
        system_prompt += "\n\nRelevant memories (YAML format, higher resonance = more relevant):"
    else:
        system_prompt += "\n\nRelevant memories from previous conversations:"

    if ctx["rtmdk_context"] and ctx["rtmdk_context"] not in ("No relevant memory.", "[]"):
        system_prompt += f"\n{ctx['rtmdk_context']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    model = get_chat_model()
    if not model:
        return "[ERROR] LM Studio not running or no model loaded."

    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=120,
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] Chat request failed: {e}"


def interactive_session():
    print("=" * 60)
    print("  RTMDK v5 Memory + LM Studio Chat")
    print("=" * 60)

    print(f"\n  Подключение к LM Studio: {LM_STUDIO_URL}")
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)
        models = resp.json().get("data", [])
        if models:
            print(f"  Модель: {models[0]['id']}")
        else:
            print("  [WARN] Нет загруженных моделей.")
    except Exception:
        print("  [ERROR] Не удалось подключиться к LM Studio!")
        print("  Запустите LM Studio и включите сервер (порт 12345).")
        return

    embedder = get_embedding

    config = RTMDKConfig(
        embedding_dim=768,
        latent_dim=64,
        tension_threshold=0.15,
        decay_rate=0.997,
        top_k=5,
        enable_async=False,
        learn_projection=True,
        projection_lr=0.005,
        soft_gates=True,
        self_supervision=True,
        context_format=ContextFormat.PLAIN,
        causal_topological=True,
        do_calculus_validation=True,
        counterfactual_enabled=True,
        meta_adaptive=True,
        self_healing=True,
    )

    memory = None
    if os.path.exists(MEMORY_FILE):
        try:
            memory = RTMDKMemory.import_field(MEMORY_FILE, embedder)
            print(f"  Память загружена: {len(memory.field.nodes)} узлов")
        except Exception:
            pass

    if memory is None:
        memory = RTMDKMemory(config=config, embedder=embedder)
        print("  Новая память инициализирована (v5)")

    session_id = "default"
    print(f"\n  Команды: /stats, /export, /clear, /format <json|yaml|plain>, /session <id>, /quit")
    print(f"           /causal - каузальная сводка, /whatif <JSON> - контрфактуальный запрос")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            break

        if user_input.lower() == "/stats":
            stats = memory.get_stats()
            print(f"\n  Nodes: {stats['active_nodes']}")
            print(f"  Queries: {stats['total_queries']}")
            print(f"  Consolidations: {stats['consolidations']}")
            print(f"  Causal edges: {stats.get('causal_edges', 0)}")
            print(f"  Contradictions: {stats.get('contradictions', 0)}")
            print(f"  Blocked consolidations: {stats.get('blocked_consolidations', 0)}")
            print(f"  Projection updates: {stats.get('projection_updates', 0)}")
            print(f"  Self-sup checks: {stats.get('self_sup_checks', 0)}")
            print(f"  Healing events: {stats.get('healing_events', 0)}")
            if 'tda_trend' in stats:
                print(f"  TDA trend: {stats['tda_trend']}")
            print(f"  Avg response: {stats['avg_response']:.4f}")
            continue

        if user_input.lower() == "/causal":
            summary = memory.get_causal_summary()
            print(f"\n  Causal edges: {summary.get('causal_edges', 0)}")
            print(f"  Contradictions: {summary.get('contradictions', 0)}")
            if summary.get('top_effects'):
                print("  Top causal effects:")
                for effect, strength in summary['top_effects'][:5]:
                    print(f"    {effect}: P={strength:.3f}")
            continue

        if user_input.startswith("/whatif "):
            try:
                query_json = json.loads(user_input[8:])
                intervention = query_json.get("do", {})
                targets = query_json.get("query", [])
                result = memory.counterfactual_query(intervention, targets)
                print(f"\n  Counterfactual: {result.query}")
                print(f"  Confidence: {result.confidence:.3f}")
                for node, prob in result.predicted_outcomes:
                    print(f"    P({node}|do) = {prob:.3f}")
                for step in result.reasoning_path:
                    print(f"    → {step}")
            except json.JSONDecodeError:
                print("\n  Usage: /whatif {\"do\": {\"node\": \"value\"}, \"query\": [\"target1\", \"target2\"]}")
            continue

        if user_input.lower() == "/export":
            memory.export_field(MEMORY_FILE)
            print(f"\n  Exported to {MEMORY_FILE}")
            continue

        if user_input.lower() == "/clear":
            memory.clear()
            print("\n  Memory cleared")
            continue

        if user_input.startswith("/format "):
            fmt_name = user_input.split(" ", 1)[1].strip().lower()
            fmt_map = {"json": ContextFormat.JSON, "yaml": ContextFormat.YAML, "plain": ContextFormat.PLAIN}
            if fmt_name in fmt_map:
                memory.config.context_format = fmt_map[fmt_name]
                print(f"\n  Context format: {fmt_name}")
            continue

        if user_input.startswith("/session "):
            session_id = user_input.split(" ", 1)[1].strip()
            print(f"\n  Session: {session_id}")
            continue

        print("\n  Thinking...", end="", flush=True)
        t0 = time.time()

        response = chat_with_memory(memory, user_input, session_id)
        elapsed = time.time() - t0

        print(f"\r  [{elapsed:.1f}s]{' ' * 20}")
        print(f"\nAssistant: {response}")

        memory.save_context(
            {"input": user_input, "session_id": session_id},
            {"output": response}
        )

    memory.export_field(MEMORY_FILE)
    print(f"\n  Session saved to {MEMORY_FILE}")
    print("  Goodbye!")


if __name__ == "__main__":
    interactive_session()
