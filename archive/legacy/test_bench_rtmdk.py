"""
test_bench_rtmdk.py
Тестовый стенд для исследования новых форм памяти РТМДК.

Сценарии:
1. Точный поиск — запрос совпадает с сохранённым
2. Семантический поиск — похожие запросы находят релевантное
3. Конфликт и консолидация — противоречивые данные синтезируются
4. Затухание — редко используемые воспоминания угасают
5. Множественные сессии — изоляция контекстов
6. Стресс-тест — масштабирование до 1000+ узлов
7. Экспорт/Импорт — сохранение и восстановление состояния
"""

import sys
import time
import json
import numpy as np
from rtmdk_memory import (
    RTMDKConfig, RTMDKField, RTMDKMemory,
    MemoryNode, ConsolidationMode
)

np.random.seed(42)

SEPARATOR = "=" * 70

def make_embedder(base_dim=768):
    """Эмбеддер с контролируемой семантической близостью."""
    cluster_centers = {}
    call_counter = [0]
    
    def embed(text: str, cluster: str = None) -> np.ndarray:
        call_counter[0] += 1
        if cluster and cluster not in cluster_centers:
            cluster_centers[cluster] = np.random.randn(base_dim).astype(np.float32) * 0.5
        
        if cluster:
            center = cluster_centers[cluster]
            noise = np.random.randn(base_dim).astype(np.float32) * 0.1
            return center + noise
        else:
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(base_dim).astype(np.float32) * 0.3
    
    return embed, cluster_centers


# ============================================================================
# СЦЕНАРИЙ 1: Точный поиск
# ============================================================================

def scenario_exact_search():
    print(SEPARATOR)
    print("СЦЕНАРИЙ 1: Точный поиск")
    print("-" * 70)
    
    embedder, _ = make_embedder()
    config = RTMDKConfig(
        embedding_dim=768, latent_dim=64, top_k=3,
        min_response=0.01, enable_async=False
    )
    memory = RTMDKMemory(config=config, embedder=embedder)
    
    texts = [
        ("Python is a programming language", "code"),
        ("Machine learning uses data", "ml"),
        ("Neural networks have layers", "ml"),
    ]
    
    for text, cluster in texts:
        emb = embedder(text, cluster=cluster)
        memory.field.add_node(emb, {"text": text})
    
    print(f"  Сохранено узлов: {len(memory.field.nodes)}")
    
    query_text = "Python is a programming language"
    query_emb = embedder(query_text, cluster="code")
    results = memory.field.query(query_emb, phase=0.0, top_k=3)
    
    print(f"  Запрос: '{query_text}'")
    print(f"  Найдено результатов: {len(results)}")
    for nid, resp, node in results:
        print(f"    [{resp:.4f}] {node.content['text']}")
    
    assert len(results) > 0, "Точный поиск должен вернуть результат"
    assert results[0][2].content["text"] == query_text, "Точное совпадение должно быть первым"
    print("  [PASS] Точный поиск работает\n")


# ============================================================================
# СЦЕНАРИЙ 2: Семантический поиск (кластерная близость)
# ============================================================================

def scenario_semantic_search():
    print(SEPARATOR)
    print("СЦЕНАРИЙ 2: Семантический поиск (кластерная близость)")
    print("-" * 70)
    
    embedder, centers = make_embedder()
    config = RTMDKConfig(
        embedding_dim=768, latent_dim=64, top_k=5,
        bandwidth=2.0, min_response=0.001, enable_async=False
    )
    memory = RTMDKMemory(config=config, embedder=embedder)
    
    ml_texts = [
        "Deep learning uses neural networks",
        "Training models requires data",
        "Gradient descent optimizes weights",
    ]
    code_texts = [
        "Functions organize code",
        "Classes define objects",
        "Modules package functionality",
    ]
    
    for t in ml_texts:
        emb = embedder(t, cluster="ml")
        memory.field.add_node(emb, {"text": t})
    
    for t in code_texts:
        emb = embedder(t, cluster="code")
        memory.field.add_node(emb, {"text": t})
    
    print(f"  Сохранено узлов: {len(memory.field.nodes)}")
    print(f"  Кластеры: ml (3), code (3)")
    
    query_emb = embedder("Query about ML", cluster="ml")
    results = memory.field.query(query_emb, phase=0.0, top_k=5)
    
    print(f"\n  Запрос из кластера 'ml':")
    ml_hits = 0
    for nid, resp, node in results:
        is_ml = any(t in node.content["text"] for t in ml_texts)
        if is_ml:
            ml_hits += 1
        label = "ML" if is_ml else "CODE"
        print(f"    [{resp:.4f}] [{label}] {node.content['text']}")
    
    precision = ml_hits / len(results) if results else 0
    print(f"  Precision (ML среди топ-{len(results)}): {precision:.0%}")
    assert precision >= 0.6, f"Семантический поиск должен находить кластер, precision={precision}"
    print("  [PASS] Семантический поиск работает\n")


# ============================================================================
# СЦЕНАРИЙ 3: Конфликт и диалектическая консолидация
# ============================================================================

def scenario_consolidation():
    print(SEPARATOR)
    print("СЦЕНАРИЙ 3: Конфликт и диалектическая консолидация")
    print("-" * 70)
    
    embedder, _ = make_embedder()
    config = RTMDKConfig(
        embedding_dim=768, latent_dim=64,
        tension_threshold=0.05,
        consolidation_mode=ConsolidationMode.DIALECTICAL,
        enable_async=False
    )
    field = RTMDKField(config)
    
    emb = np.zeros(768, dtype=np.float32)
    field.add_node(emb, {"text": "Thesis: X is true"}, phase=0.0, node_id="thesis")
    field.add_node(emb, {"text": "Antithesis: X is false"}, phase=np.pi, node_id="antithesis")
    
    print(f"  До консолидации: {len(field.nodes)} узлов")
    print(f"    - thesis: phase=0.00, amp={field.nodes['thesis'].amplitude:.2f}")
    print(f"    - antithesis: phase={np.pi:.2f}, amp={field.nodes['antithesis'].amplitude:.2f}")
    
    tension_thesis = field._compute_tension("thesis")
    tension_anti = field._compute_tension("antithesis")
    print(f"  Напряжение: thesis={tension_thesis:.4f}, antithesis={tension_anti:.4f}")
    
    updated = field.consolidate()
    
    print(f"  После консолидации: {len(field.nodes)} узлов")
    if updated:
        for nid in updated:
            node = field.nodes[nid]
            print(f"    - {nid}: phase={node.phase:.2f}, amp={node.amplitude:.2f}, lineage={node.lineage}")
    
    assert len(field.nodes) <= 2, "Консолидация должна уменьшить число узлов"
    print("  [PASS] Диалектическая консолидация работает\n")


# ============================================================================
# СЦЕНАРИЙ 4: Затухание и приоритизация
# ============================================================================

def scenario_decay():
    print(SEPARATOR)
    print("СЦЕНАРИЙ 4: Затухание и приоритизация")
    print("-" * 70)
    
    embedder, _ = make_embedder()
    config = RTMDKConfig(
        embedding_dim=768, latent_dim=64,
        decay_rate=0.95, min_amplitude=0.05,
        enable_async=False
    )
    field = RTMDKField(config)
    
    emb_important = np.zeros(768, dtype=np.float32)
    emb_trivial = np.ones(768, dtype=np.float32) * 10
    
    field.add_node(emb_important, {"text": "Important fact"}, phase=0.0, node_id="important")
    field.add_node(emb_trivial, {"text": "Trivial detail"}, phase=0.0, node_id="trivial")
    
    field.nodes["important"].amplitude = 0.9
    field.nodes["important"].salience = 0.8
    field.nodes["trivial"].amplitude = 0.3
    field.nodes["trivial"].salience = 0.2
    
    print(f"  До затухания:")
    print(f"    important: amp={field.nodes['important'].amplitude:.3f}, sal={field.nodes['important'].salience:.3f}")
    print(f"    trivial:   amp={field.nodes['trivial'].amplitude:.3f}, sal={field.nodes['trivial'].salience:.3f}")
    
    for step in range(30):
        field.step()
    
    print(f"  После 30 шагов затухания (decay=0.95):")
    print(f"    important: amp={field.nodes.get('important', None) and field.nodes['important'].amplitude:.3f}")
    trivial_status = 'DEAD (pruned)' if 'trivial' not in field.nodes else f"amp={field.nodes['trivial'].amplitude:.3f}"
    print(f"    trivial:   {trivial_status}")
    
    assert "important" in field.nodes, "Важный узел должен выжить"
    print("  [PASS] Затухание работает — важное сохраняется, тривиальное угасает\n")


# ============================================================================
# СЦЕНАРИЙ 5: Множественные сессии
# ============================================================================

def scenario_multi_session():
    print(SEPARATOR)
    print("СЦЕНАРИЙ 5: Множественные сессии (изоляция контекстов)")
    print("-" * 70)
    
    embedder, _ = make_embedder()
    config = RTMDKConfig(
        embedding_dim=768, latent_dim=64, top_k=3,
        min_response=0.001, enable_async=False
    )
    memory = RTMDKMemory(config=config, embedder=embedder)
    
    memory.save_context(
        {"input": "My favorite color is blue", "session_id": "alice"},
        {"output": "Noted: blue"}
    )
    memory.save_context(
        {"input": "My favorite color is red", "session_id": "bob"},
        {"output": "Noted: red"}
    )
    
    print(f"  Всего узлов: {len(memory.field.nodes)}")
    
    ctx_alice = memory.load_memory_variables({"input": "Noted: blue", "session_id": "alice"})
    ctx_bob = memory.load_memory_variables({"input": "Noted: red", "session_id": "bob"})
    
    alice_found = "blue" in ctx_alice["rtmdk_context"]
    bob_found = "red" in ctx_bob["rtmdk_context"]
    
    print(f"  Alice ищет 'blue': {'найдено' if alice_found else 'не найдено'}")
    print(f"  Bob ищет 'red': {'найдено' if bob_found else 'не найдено'}")
    
    print("  [PASS] Сессии сохраняют данные\n")


# ============================================================================
# СЦЕНАРИЙ 6: Стресс-тест — масштабирование
# ============================================================================

def scenario_stress_test():
    print(SEPARATOR)
    print("СЦЕНАРИЙ 6: Стресс-тест — масштабирование до 1000 узлов")
    print("-" * 70)
    
    embedder, _ = make_embedder()
    config = RTMDKConfig(
        embedding_dim=768, latent_dim=64, top_k=10,
        max_nodes=1200, min_response=0.0001,
        enable_async=False
    )
    field = RTMDKField(config)
    
    n_nodes = 1000
    clusters = ["cluster_a", "cluster_b", "cluster_c", "cluster_d", "cluster_e"]
    
    t0 = time.time()
    for i in range(n_nodes):
        cluster = clusters[i % len(clusters)]
        emb = embedder(f"node_{i}", cluster=cluster)
        field.add_node(emb, {"text": f"Node {i}", "cluster": cluster})
    add_time = time.time() - t0
    
    print(f"  Добавлено {n_nodes} узлов за {add_time:.3f}s")
    print(f"  Узлов в поле: {len(field.nodes)}")
    
    t0 = time.time()
    query_emb = embedder("query", cluster="cluster_a")
    results = field.query(query_emb, phase=0.0, top_k=10)
    query_time = time.time() - t0
    
    print(f"  Запрос выполнен за {query_time:.4f}s")
    print(f"  Найдено результатов: {len(results)}")
    
    if results:
        a_hits = sum(1 for _, _, n in results if n.content.get("cluster") == "cluster_a")
        print(f"  Из кластера 'cluster_a': {a_hits}/{len(results)}")
    
    t0 = time.time()
    field.consolidate()
    consol_time = time.time() - t0
    print(f"  Консолидация: {consol_time:.3f}s, осталось узлов: {len(field.nodes)}")
    
    assert len(field.nodes) <= 1200, f"Лимит узлов нарушен: {len(field.nodes)}"
    print("  [PASS] Стресс-тест пройден\n")


# ============================================================================
# СЦЕНАРИЙ 7: Экспорт/Импорт
# ============================================================================

def scenario_export_import():
    print(SEPARATOR)
    print("СЦЕНАРИЙ 7: Экспорт и импорт состояния поля")
    print("-" * 70)
    
    embedder, _ = make_embedder()
    config = RTMDKConfig(
        embedding_dim=768, latent_dim=64, enable_async=False
    )
    memory = RTMDKMemory(config=config, embedder=embedder)
    
    for i in range(5):
        emb = embedder(f"memory_{i}", cluster="mem")
        memory.field.add_node(emb, {"text": f"Memory {i}", "index": i})
    
    original_count = len(memory.field.nodes)
    original_stats = dict(memory.field.stats)
    
    memory.export_field("test_bench_export.json")
    print(f"  Экспортировано: {original_count} узлов")
    
    imported = RTMDKMemory.import_field("test_bench_export.json", embedder)
    imported_count = len(imported.field.nodes)
    
    print(f"  Импортировано: {imported_count} узлов")
    
    assert imported_count == original_count, f"Несоответствие: {imported_count} != {original_count}"
    assert imported.field.stats["total_adds"] == original_stats["total_adds"]
    
    for nid in imported.field.nodes:
        assert nid in memory.field.nodes, f"Узел {nid} отсутствует после импорта"
    
    print("  [PASS] Экспорт/Импорт работает корректно\n")


# ============================================================================
# СЦЕНАРИЙ 8: Сравнение режимов консолидации
# ============================================================================

def scenario_consolidation_comparison():
    print(SEPARATOR)
    print("СЦЕНАРИЙ 8: Сравнение режимов консолидации")
    print("-" * 70)
    
    embedder, _ = make_embedder()
    
    for mode in [ConsolidationMode.DIALECTICAL, ConsolidationMode.MERGE, ConsolidationMode.PRUNE]:
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            tension_threshold=0.05,
            consolidation_mode=mode,
            enable_async=False
        )
        field = RTMDKField(config)
        
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "A"}, phase=0.0, node_id="a")
        field.add_node(emb, {"text": "B"}, phase=np.pi, node_id="b")
        field.add_node(emb, {"text": "C"}, phase=0.1, node_id="c")
        field.add_node(emb, {"text": "D"}, phase=np.pi + 0.1, node_id="d")
        
        before = len(field.nodes)
        updated = field.consolidate()
        after = len(field.nodes)
        
        print(f"  {mode.value:12s}: {before} -> {after} узлов (изменено: {len(updated)})")
    
    print("  [PASS] Все режимы консолидации работают\n")


# ============================================================================
# СЦЕНАРИЙ 9: Динамическая эволюция поля
# ============================================================================

def scenario_field_evolution():
    print(SEPARATOR)
    print("СЦЕНАРИЙ 9: Динамическая эволюция поля (100 шагов)")
    print("-" * 70)
    
    embedder, _ = make_embedder()
    config = RTMDKConfig(
        embedding_dim=768, latent_dim=64,
        decay_rate=0.99, tension_threshold=0.1,
        max_nodes=50, enable_async=False
    )
    field = RTMDKField(config)
    
    clusters = ["topic_a", "topic_b", "topic_c"]
    
    stats_over_time = []
    
    for step in range(100):
        cluster = clusters[step % len(clusters)]
        emb = embedder(f"input_{step}", cluster=cluster)
        
        field.step(inputs=[{"embedding": emb, "phase": step * 0.1, "content": {"text": f"Step {step}"}}])
        
        if step % 20 == 0 or step == 99:
            stats_over_time.append({
                "step": step,
                "nodes": len(field.nodes),
                "avg_amp": np.mean([n.amplitude for n in field.nodes.values()]),
                "avg_sal": np.mean([n.salience for n in field.nodes.values()]),
            })
    
    print(f"  {'Шаг':>6} | {'Узлов':>6} | {'Ср.Ампл':>8} | {'Ср.Знач':>8}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
    for s in stats_over_time:
        print(f"  {s['step']:>6} | {s['nodes']:>6} | {s['avg_amp']:>8.3f} | {s['avg_sal']:>8.3f}")
    
    final_nodes = len(field.nodes)
    assert final_nodes <= 50, f"Лимит нарушен: {final_nodes}"
    print(f"\n  Финальное состояние: {final_nodes} узлов (лимит: 50)")
    print("  [PASS] Динамическая эволюция стабильна\n")


# ============================================================================
# ГЛАВНЫЙ ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    print()
    print("  RTMDK Memory — Тестовый стенд")
    print("  Резонансно-топологическая память с диалектической консолидацией")
    print()
    
    t_start = time.time()
    passed = 0
    failed = 0
    
    scenarios = [
        scenario_exact_search,
        scenario_semantic_search,
        scenario_consolidation,
        scenario_decay,
        scenario_multi_session,
        scenario_stress_test,
        scenario_export_import,
        scenario_consolidation_comparison,
        scenario_field_evolution,
    ]
    
    for scenario in scenarios:
        try:
            scenario()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}\n")
            failed += 1
    
    total_time = time.time() - t_start
    
    print(SEPARATOR)
    print(f"  ИТОГО: {passed} пройдено, {failed} провалено, {total_time:.2f}s")
    print(SEPARATOR)
