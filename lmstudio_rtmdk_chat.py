"""
lmstudio_rtmdk_chat.py
Чат с LM Studio + RTMDK v8 память.

Требования:
  1. LM Studio запущен на http://localhost:12345
  2. Модель загружена в LM Studio
  3. pip install requests numpy scipy pydantic

Запуск:
  python lmstudio_rtmdk_chat.py

Команды в чате:
  /stats        - полная статистика памяти
  /tiers        - распределение по уровням памяти
  /health       - здоровье поля + топология
  /causal       - каузальная сводка
  /contradict   - обнаруженные противоречия
  /whatif JSON  - контрфактуальный запрос
  /imagine JSON - воображение сценариев
  /hyperbolic   - статистика гиперболической геометрии
  /predictive   - статистика предсказательного кодирования
  /privacy      - статус дифференциальной приватности
  /export       - экспорт состояния поля
  /clear        - очистить память
  /format json|yaml|plain - формат контекста
  /session <id> - переключить сессию
  /quit         - выйти
"""

import sys
import os
import json
import time
import requests
import numpy as np
from rtmdk_memory_v8 import (
    RTMDKConfig, RTMDKMemory, ContextFormat,
    detect_tier, detect_modality,
)

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


def print_stats(memory: RTMDKMemory):
    stats = memory.get_stats()
    print(f"\n  === Статистика поля ===")
    print(f"  Nodes: {stats['active_nodes']}")
    print(f"  Queries: {stats['total_queries']}")
    print(f"  Consolidations: {stats['consolidations']}")
    print(f"  Causal edges: {stats.get('causal_edges', 0)}")
    print(f"  Contradictions: {stats.get('contradictions', 0)}")
    print(f"  Blocked consolidations: {stats.get('blocked_consolidations', 0)}")
    print(f"  Projection updates: {stats.get('projection_updates', 0)}")
    print(f"  Self-sup checks: {stats.get('self_sup_checks', 0)}")
    print(f"  Healing events: {stats.get('healing_events', 0)}")
    print(f"  ODE steps: {stats.get('ode_steps', 0)}")
    print(f"  Scenarios generated: {stats.get('scenarios_generated', 0)}")
    print(f"  Meta optimizations: {stats.get('meta_optimizations', 0)}")
    print(f"  Federated syncs: {stats.get('federated_syncs', 0)}")
    if 'tda_trend' in stats:
        print(f"  TDA trend: {stats['tda_trend']}")
    print(f"  Field health: {stats.get('field_health', 'unknown')}")
    print(f"  Tier coherence: {stats.get('tier_coherence', 0.0):.3f}")
    print(f"  Free energy: {stats.get('free_energy', 0.0):.4f}")
    print(f"  Response smoothness: {stats.get('response_smoothness', 0.0):.3f}")
    print(f"  Privacy budget: {stats.get('privacy_budget_spent', 0.0):.3f}")
    print(f"  Avg response: {stats['avg_response']:.4f}")


def print_tiers(memory: RTMDKMemory):
    stats = memory.get_stats()
    dist = stats.get('tier_distribution', {})
    print(f"\n  === Распределение по уровням ===")
    for tier, count in sorted(dist.items()):
        decay = memory.field.cfg.tier_decay.get(tier, '?')
        print(f"  {tier:12s}: {count:3d} узлов  (decay={decay})")
    print(f"  Tier coherence: {stats.get('tier_coherence', 0.0):.3f}")


def print_health(memory: RTMDKMemory):
    health = memory.get_field_health()
    print(f"\n  === Здоровье поля ===")
    for k, v in health.items():
        print(f"  {k}: {v}")


def print_hyperbolic(memory: RTMDKMemory):
    stats = memory.get_stats()
    print(f"\n  === Гиперболическая геометрия ===")
    print(f"  Enabled: {memory.field.cfg.hyperbolic}")
    print(f"  Ball radius: {memory.field.cfg.ball_radius}")
    print(f"  Avg hyperbolic dist: {stats.get('avg_hyperbolic_dist', 0.0):.4f}")


def print_predictive(memory: RTMDKMemory):
    stats = memory.get_stats()
    print(f"\n  === Предсказательное кодирование ===")
    print(f"  Enabled: {memory.field.cfg.predictive_coding}")
    print(f"  Free energy: {stats.get('free_energy', 0.0):.4f}")
    print(f"  Prediction error: {stats.get('prediction_error', 0.0):.4f}")
    print(f"  Surprise level: {stats.get('surprise_level', 0.0):.3f}")


def print_privacy(memory: RTMDKMemory):
    stats = memory.get_stats()
    print(f"\n  === Дифференциальная приватность ===")
    print(f"  Enabled: {memory.field.cfg.differential_privacy}")
    print(f"  Epsilon: {memory.field.cfg.dp_epsilon}")
    print(f"  Budget spent: {stats.get('privacy_budget_spent', 0.0):.3f}")
    print(f"  Updates clipped: {stats.get('updates_clipped', 0)}")


def print_contradictions(memory: RTMDKMemory):
    contradictions = memory.get_contradictions()
    print(f"\n  === Противоречия ({len(contradictions)}) ===")
    for c in contradictions[:5]:
        status = "RESOLVED" if c.resolved else "ACTIVE"
        print(f"  [{status}] {c.id}: {c.effect_node}")
        for cause, strength in c.causes:
            print(f"    do({cause}) → P={strength:.3f}")
        if c.contradiction_reason:
            print(f"    Reason: {c.contradiction_reason}")


def interactive_session():
    print("=" * 60)
    print("  RTMDK v8 Memory + LM Studio Chat")
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
        # Фаза 11
        memory_tiers={"episodic", "semantic", "procedural"},
        hyperbolic=False,
        predictive_coding=False,
        counterfactual_imagination=True,
        differential_privacy=False,
        # Фаза 12
        sparse_routing=False,
        cognitive_compression=True,
        crystallization=True,
        crystallization_freq=100,
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
        print("  Новая память инициализирована (v8)")

    session_id = "default"
    print(f"\n  Команды: /stats, /tiers, /health, /causal, /contradict,")
    print(f"           /hyperbolic, /predictive, /privacy,")
    print(f"           /shards, /crystallize, /compression,")
    print(f"           /whatif JSON, /imagine JSON,")
    print(f"           /format <json|yaml|plain>, /session <id>,")
    print(f"           /export, /clear, /quit")
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
            print_stats(memory)
            continue

        if user_input.lower() == "/tiers":
            print_tiers(memory)
            continue

        if user_input.lower() == "/health":
            print_health(memory)
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

        if user_input.lower() == "/contradict":
            print_contradictions(memory)
            continue

        if user_input.lower() == "/hyperbolic":
            print_hyperbolic(memory)
            continue

        if user_input.lower() == "/predictive":
            print_predictive(memory)
            continue

        if user_input.lower() == "/privacy":
            print_privacy(memory)
            continue

        if user_input.lower() == "/shards":
            stats = memory.get_stats()
            print(f"\n  === Шардирование (MoE-память) ===")
            print(f"  Enabled: {memory.field.cfg.sparse_routing}")
            print(f"  Num shards: {memory.field.cfg.num_shards}")
            print(f"  Top shards: {memory.field.cfg.top_shards}")
            print(f"  Shard hits: {stats.get('shard_hits', 0)}")
            print(f"  Shard misses: {stats.get('shard_misses', 0)}")
            print(f"  Avg query time: {stats.get('avg_shard_query_time_ms', 0.0):.2f}ms")
            continue

        if user_input.lower() == "/crystallize":
            stats = memory.get_stats()
            print(f"\n  === Кристаллизация ===")
            print(f"  Enabled: {memory.field.cfg.crystallization}")
            print(f"  Crystallizations: {stats.get('crystallizations', 0)}")
            print(f"  Crystallized clusters: {stats.get('crystallized_clusters', 0)}")
            continue

        if user_input.lower() == "/compression":
            stats = memory.get_stats()
            print(f"\n  === Когнитивное сжатие ===")
            print(f"  Enabled: {memory.field.cfg.cognitive_compression}")
            print(f"  Compressions: {stats.get('cognitive_compressions', 0)}")
            print(f"  Tokens saved: {stats.get('context_tokens_saved', 0)}")
            continue

        if user_input.lower() == "/crystallize_now":
            memory.field._crystallize_recurring()
            print("\n  Crystallization triggered")
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
                print("\n  Usage: /whatif {\"do\": {\"node\": \"value\"}, \"query\": [\"target1\"]}")
            continue

        if user_input.startswith("/imagine "):
            try:
                query_json = json.loads(user_input[9:])
                base_query = query_json.get("query", user_input)
                intervention = query_json.get("intervention", {})
                results = memory.imagine_counterfactual(base_query, intervention)
                print(f"\n  Imagined {len(results)} scenarios:")
                for r in results:
                    print(f"  [HYPOTHETICAL] node={r['node_id']} conf={r['confidence']:.3f}")
                    print(f"    trajectory steps: {len(r['trajectory'])}")
            except json.JSONDecodeError:
                print("\n  Usage: /imagine {\"query\": \"text\", \"intervention\": {\"n0\": 0.5}}")
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
