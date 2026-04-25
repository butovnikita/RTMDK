"""
rtmdk_memory.py
Резонансно-топологическая память с диалектической консолидацией
Интеграция с современным LLM-стеком (LangChain / LlamaIndex / transformers)
"""

from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union, Callable
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# КОНФИГУРАЦИЯ И ТИПЫ ДАННЫХ
# ============================================================================

class ConsolidationMode(Enum):
    DIALECTICAL = "dialectical"  # тезис+антитезис → синтез
    MERGE = "merge"              # простое усреднение
    PRUNE = "prune"              # удаление слабого узла

@dataclass
class RTMDKConfig:
    """Гибкая конфигурация симуляции поля памяти."""
    # Геометрия поля
    embedding_dim: int = 768           # размерность входных эмбеддингов
    latent_dim: int = 64               # внутренняя размерность поля (сжатие)
    
    # Резонанс
    resonance_kernel: str = "gaussian_phase"  # gaussian / cosine / gaussian_phase
    phase_coupling: float = 0.3        # вес фазовой согласованности в отклике
    bandwidth: float = 1.0             # ширина резонансной кривой
    
    # Динамика
    attraction_lr: float = 0.02        # скорость притяжения к входу
    phase_sync_lr: float = 0.01        # скорость синхронизации фаз
    decay_rate: float = 0.998          # множитель затухания амплитуды/значимости
    min_amplitude: float = 0.05        # порог "жизни" узла
    
    # Консолидация
    tension_threshold: float = 0.25    # порог напряжения для синтеза
    consolidation_mode: ConsolidationMode = ConsolidationMode.DIALECTICAL
    max_nodes: Optional[int] = 5000    # лимит узлов (None = неограничено)
    
    # Поиск
    top_k: int = 5                     # сколько результатов возвращать
    min_response: float = 0.1          # минимальный отклик для включения в ответ
    
    # Мета
    enable_async: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        logger.setLevel(getattr(logging, self.log_level.upper()))

@dataclass
class MemoryNode:
    """Атомарный элемент поля памяти — устойчивое возмущение (аттрактор)."""
    id: str
    latent_pos: NDArray[np.float32]    # позиция в сжатом латентном пространстве
    phase: float                       # фаза колебания [0, 2π)
    amplitude: float                   # "сила" узла [0, 1]
    salience: float                    # значимость для системы [0, 1]
    tension: float = 0.0               # локальное топологическое напряжение
    content: Dict = field(default_factory=dict)  # исходные данные (текст, метаданные)
    created_at: float = field(default_factory=time.time)
    last_resonated: float = 0.0
    lineage: List[str] = field(default_factory=list)  # история синтеза: ["id1+id2", ...]
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "latent_pos": self.latent_pos.tolist(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> MemoryNode:
        data["latent_pos"] = np.array(data["latent_pos"], dtype=np.float32)
        return cls(**data)

# ============================================================================
# ЯДРО: РЕЗОНАНСНО-ТОПОЛОГИЧЕСКОЕ ПОЛЕ
# ============================================================================

class RTMDKField:
    """
    Дискретизированная аппроксимация непрерывного семантического многообразия.
    Реализует резонансный поиск, динамику напряжения и диалектическую консолидацию.
    """
    
    def __init__(self, config: RTMDKConfig, projection_matrix: Optional[NDArray] = None):
        self.cfg = config
        self.nodes: Dict[str, MemoryNode] = {}
        self.node_index: List[str] = []  # для быстрого доступа по индексу
        
        # Проекция: embedding_dim → latent_dim (PCA-like, но обучаемая)
        if projection_matrix is None:
            self.projection = np.random.randn(config.embedding_dim, config.latent_dim).astype(np.float32) * 0.1
        else:
            self.projection = projection_matrix.astype(np.float32)
        
        # Статистика для мониторинга
        self.stats = {
            "total_adds": 0, "total_queries": 0, "consolidations": 0,
            "avg_response": 0.0, "active_nodes": 0
        }
    
    def _project(self, embedding: NDArray) -> NDArray:
        """Проекция внешнего эмбеддинга в латентное поле."""
        if embedding.ndim == 1:
            return (embedding @ self.projection).astype(np.float32)
        return (embedding @ self.projection).astype(np.float32)
    
    def _resonance_response(self, query_latent: NDArray, query_phase: float, 
                           node: MemoryNode) -> float:
        """Вычисление резонансного отклика одного узла."""
        # Пространственная близость (ядро)
        if self.cfg.resonance_kernel == "gaussian":
            spatial = np.exp(-np.sum((query_latent - node.latent_pos)**2) / (2 * self.cfg.bandwidth**2))
        elif self.cfg.resonance_kernel == "cosine":
            spatial = 0.5 + 0.5 * np.dot(query_latent, node.latent_pos) / (
                np.linalg.norm(query_latent) * np.linalg.norm(node.latent_pos) + 1e-8)
        else:  # gaussian_phase
            dist = np.linalg.norm(query_latent - node.latent_pos)
            spatial = np.exp(-dist / self.cfg.bandwidth)
        
        # Фазовая согласованность
        phase_align = 0.5 + 0.5 * np.cos(node.phase - query_phase)
        
        # Итоговый отклик
        response = spatial * ((1 - self.cfg.phase_coupling) + self.cfg.phase_coupling * phase_align)
        return response * node.amplitude * node.salience
    
    def query(self, embedding: NDArray, phase: float = 0.0, 
              top_k: Optional[int] = None) -> List[Tuple[str, float, MemoryNode]]:
        """
        Резонансный поиск: возвращает топ узлов по силе отклика.
        """
        top_k = top_k or self.cfg.top_k
        query_latent = self._project(embedding)
        
        results = []
        for nid in self.node_index:
            node = self.nodes[nid]
            resp = self._resonance_response(query_latent, phase, node)
            if resp >= self.cfg.min_response:
                results.append((nid, resp, node))
                node.last_resonated = time.time()  # mark as active
        
        results.sort(key=lambda x: x[1], reverse=True)
        self.stats["total_queries"] += 1
        if results:
            self.stats["avg_response"] = 0.9 * self.stats["avg_response"] + 0.1 * results[0][1]
        
        return results[:top_k]
    
    def add_node(self, embedding: NDArray, content: Dict, 
                 phase: Optional[float] = None, node_id: Optional[str] = None) -> str:
        """Добавление нового аттрактора в поле."""
        nid = node_id or f"n_{len(self.nodes)}_{int(time.time()*1000)}"
        latent = self._project(embedding)
        phase = phase if phase is not None else np.random.uniform(0, 2*np.pi)
        
        node = MemoryNode(
            id=nid,
            latent_pos=latent,
            phase=phase,
            amplitude=0.7,
            salience=0.6,
            content=content,
            lineage=[]
        )
        self.nodes[nid] = node
        self.node_index.append(nid)
        self.stats["total_adds"] += 1
        return nid
    
    def _compute_tension(self, node_id: str, neighborhood_radius: float = 2.0) -> float:
        """Вычисление топологического напряжения узла на основе вариабельности соседей."""
        node = self.nodes[node_id]
        neighbors = []
        
        for other_id in self.node_index:
            if other_id == node_id:
                continue
            other = self.nodes[other_id]
            dist = np.linalg.norm(node.latent_pos - other.latent_pos)
            if dist < neighborhood_radius:
                neighbors.append(other)
        
        if len(neighbors) < 2:
            return 0.0
        
        # Напряжение = комбинация фазового и значимостного рассогласования
        phases = np.array([n.phase for n in neighbors])
        saliences = np.array([n.salience for n in neighbors])
        
        phase_var = np.std(np.cos(phases)) + np.std(np.sin(phases))  # circular std
        salience_var = np.std(saliences)
        
        return 0.6 * phase_var + 0.4 * salience_var
    
    def consolidate(self, mode: Optional[ConsolidationMode] = None) -> List[str]:
        """
        Диалектическая консолидация: поиск узлов с высоким напряжением и их синтез.
        Возвращает список созданных/изменённых узлов.
        """
        mode = mode or self.cfg.consolidation_mode
        updated = []
        
        # 1. Вычисление напряжения для всех узлов
        for nid in self.node_index:
            self.nodes[nid].tension = self._compute_tension(nid)
        
        # 2. Поиск пар для консолидации
        high_tension = [nid for nid in self.node_index 
                       if self.nodes[nid].tension > self.cfg.tension_threshold]
        
        processed = set()
        for nid in high_tension:
            if nid in processed or nid not in self.nodes:
                continue
                
            node = self.nodes[nid]
            # Поиск ближайшего "конфликтного" соседа
            candidates = []
            for other_id in self.node_index:
                if other_id == nid or other_id in processed or other_id not in self.nodes:
                    continue
                other = self.nodes[other_id]
                dist = np.linalg.norm(node.latent_pos - other.latent_pos)
                phase_diff = min(abs(node.phase - other.phase), 2*np.pi - abs(node.phase - other.phase))
                if dist < 2.5 and phase_diff > 1.0:  # близки, но в противофазе → конфликт
                    candidates.append((other_id, dist, phase_diff))
            
            if not candidates:
                continue
                
            candidates.sort(key=lambda x: x[1])  # по расстоянию
            partner_id = candidates[0][0]
            partner = self.nodes[partner_id]
            
            # 3. Синтез в зависимости от режима
            if mode == ConsolidationMode.DIALECTICAL:
                # Синтез: новая конфигурация, снимающая напряжение
                new_latent = 0.5 * (node.latent_pos + partner.latent_pos)
                # Фаза: среднее с учётом цикличности
                new_phase = np.arctan2(
                    0.5*(np.sin(node.phase) + np.sin(partner.phase)),
                    0.5*(np.cos(node.phase) + np.cos(partner.phase))
                ) % (2*np.pi)
                new_amp = min(1.0, 0.8 * (node.amplitude + partner.amplitude))
                new_salience = 0.7 * (node.salience + partner.salience)
                new_lineage = [f"{node.id}+{partner.id}"] + node.lineage + partner.lineage
                
                # Обновляем первый узел
                node.latent_pos = new_latent
                node.phase = new_phase
                node.amplitude = new_amp
                node.salience = new_salience
                node.tension = 0.0
                node.lineage = new_lineage
                node.content["synthesis_note"] = f"Consolidated with {partner_id} at t={time.time():.0f}"
                
                # Удаляем второй узел
                del self.nodes[partner_id]
                self.node_index.remove(partner_id)
                processed.add(partner_id)
                updated.append(nid)
                
            elif mode == ConsolidationMode.MERGE:
                # Простое усреднение
                node.latent_pos = 0.5 * (node.latent_pos + partner.latent_pos)
                node.phase = (node.phase + partner.phase) / 2
                node.amplitude = min(1.0, 0.9 * (node.amplitude + partner.amplitude))
                node.salience = 0.8 * (node.salience + partner.salience)
                node.tension = 0.0
                
                del self.nodes[partner_id]
                self.node_index.remove(partner_id)
                processed.add(partner_id)
                updated.append(nid)
                
            elif mode == ConsolidationMode.PRUNE:
                # Оставляем более сильный, удаляем слабый
                if node.salience * node.amplitude >= partner.salience * partner.amplitude:
                    del self.nodes[partner_id]
                    self.node_index.remove(partner_id)
                    processed.add(partner_id)
                else:
                    del self.nodes[nid]
                    self.node_index.remove(nid)
                    processed.add(nid)
                updated.append(nid if nid in self.nodes else partner_id)
            
            self.stats["consolidations"] += 1
            processed.add(nid)
        
        # 4. Очистка "мёртвых" узлов
        self._prune_dead_nodes()
        
        self.stats["active_nodes"] = len(self.nodes)
        return updated
    
    def _prune_dead_nodes(self):
        """Удаление узлов, потерявших значимость."""
        to_remove = [
            nid for nid in self.node_index
            if self.nodes[nid].amplitude < self.cfg.min_amplitude
            or self.nodes[nid].salience < self.cfg.min_amplitude * 0.5
        ]
        for nid in to_remove:
            del self.nodes[nid]
            self.node_index.remove(nid)
    
    def step(self, inputs: Optional[List[Dict]] = None):
        """Один шаг эволюции поля: возбуждение → напряжение → консолидация → затухание."""
        # 1. Возбуждение от входов
        if inputs:
            for inp in inputs:
                emb = inp["embedding"]
                phase = inp.get("phase", 0.0)
                content = inp.get("content", {})
                
                # Найти ближайший узел и притянуть его
                results = self.query(emb, phase, top_k=1)
                if results and results[0][1] > 0.3:  # если есть резонанс
                    nid, _, node = results[0]
                    target_latent = self._project(emb)
                    node.latent_pos += self.cfg.attraction_lr * (target_latent - node.latent_pos)
                    # Синхронизация фазы
                    phase_diff = (phase - node.phase + np.pi) % (2*np.pi) - np.pi
                    node.phase += self.cfg.phase_sync_lr * phase_diff
                    node.amplitude = min(1.0, node.amplitude + 0.05)
                    node.salience = min(1.0, node.salience + 0.03)
                else:
                    # Создать новый узел, если нет резонанса
                    self.add_node(emb, content, phase)
        
        # 2. Консолидация (периодически)
        if len(self.nodes) > 10 and np.random.random() < 0.15:
            self.consolidate()
        
        # 3. Затухание
        for node in self.nodes.values():
            node.amplitude *= self.cfg.decay_rate
            node.salience *= self.cfg.decay_rate
            node.amplitude = np.clip(node.amplitude, self.cfg.min_amplitude, 1.0)
            node.salience = np.clip(node.salience, self.cfg.min_amplitude * 0.5, 1.0)
        
        # 4. Контроль лимита узлов
        if self.cfg.max_nodes and len(self.nodes) > self.cfg.max_nodes:
            # Удалить наименее значимые
            sorted_nodes = sorted(
                self.node_index,
                key=lambda nid: self.nodes[nid].salience * self.nodes[nid].amplitude
            )
            to_remove = sorted_nodes[:len(self.nodes) - self.cfg.max_nodes]
            for nid in to_remove:
                del self.nodes[nid]
                self.node_index.remove(nid)

# ============================================================================
# ИНТЕГРАЦИЯ С LANGCHAIN: BaseMemory
# ============================================================================

class RTMDKMemory(BaseModel):
    """
    LangChain-совместимая память на основе РТМДК.
    Используется как drop-in замена VectorStore + Memory.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    # Конфигурация
    config: RTMDKConfig = Field(default_factory=RTMDKConfig)
    
    # Внешний эмбеддер (обязательно передаётся при инициализации)
    embedder: Callable[[str], NDArray[np.float32]]
    
    # Внутреннее поле
    field: Optional[RTMDKField] = Field(default=None, exclude=True)
    
    # Кэш фаз для сессий (опционально: можно вычислять из хэша контекста)
    session_phases: Dict[str, float] = Field(default_factory=dict)
    
    @model_validator(mode="before")
    @classmethod
    def _init_field(cls, data):
        if isinstance(data, dict) and data.get("field") is None:
            cfg = data.get("config", RTMDKConfig())
            data = dict(data)
            data["field"] = RTMDKField(cfg)
        return data
    
    def model_post_init(self, __context):
        if self.field is None:
            object.__setattr__(self, "field", RTMDKField(self.config))
    
    @property
    def memory_variables(self) -> List[str]:
        return ["rtmdk_context"]
    
    def _get_phase(self, session_id: Optional[str] = None) -> float:
        """Получение или генерация фазы для сессии/запроса."""
        if session_id and session_id in self.session_phases:
            return self.session_phases[session_id]
        # Фаза как функция от времени + случайность для разнообразия
        phase = (time.time() * 0.01) % (2*np.pi)
        if session_id:
            self.session_phases[session_id] = phase
        return phase
    
    def load_memory_variables(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """LangChain: загрузка релевантного контекста из памяти."""
        query = inputs.get("input", inputs.get("query", ""))
        session_id = inputs.get("session_id", "default")
        
        if not query:
            return {"rtmdk_context": ""}
        
        # Эмбеддинг запроса
        embedding = self.embedder(query)
        phase = self._get_phase(session_id)
        
        # Резонансный поиск
        results = self.field.query(embedding, phase, top_k=self.field.cfg.top_k)
        
        # Формирование контекста
        context_parts = []
        for nid, resp, node in results:
            content = node.content.get("text", "")
            meta = {k: v for k, v in node.content.items() if k != "text"}
            context_parts.append(f"[R:{resp:.2f}|S:{node.salience:.2f}] {content}")
            if meta:
                context_parts[-1] += f" |meta:{json.dumps(meta, ensure_ascii=False)}"
        
        context = "\n".join(context_parts) if context_parts else "No relevant memory."
        return {"rtmdk_context": context}
    
    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        """LangChain: сохранение нового опыта в память."""
        # Извлекаем текст для запоминания (можно кастомизировать)
        text = outputs.get("output", inputs.get("input", ""))
        session_id = inputs.get("session_id", "default")
        
        if not text.strip():
            return
        
        embedding = self.embedder(text)
        phase = self._get_phase(session_id)
        
        content = {
            "text": text,
            "timestamp": time.time(),
            "session": session_id,
            **{k: v for k, v in inputs.items() if k not in ["input", "query", "session_id"]}
        }
        
        self.field.add_node(embedding, content, phase)
        
        # Эволюция поля
        if self.config.enable_async:
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self._evolve_field_async())
            except RuntimeError:
                self.field.step()
        else:
            self.field.step()
    
    async def _evolve_field_async(self):
        """Фоновая эволюция поля без блокировки основного потока."""
        await asyncio.sleep(0.01)  # небольшой сдвиг, чтобы не конкурировать с I/O
        self.field.step()
    
    def clear(self) -> None:
        self.field = RTMDKField(self.config)
        self.session_phases.clear()
    
    # === Утилиты для отладки и мониторинга ===
    
    def get_stats(self) -> Dict:
        self.field.stats["active_nodes"] = len(self.field.nodes)
        return {**self.field.stats, "config": asdict(self.config)}
    
    def export_field(self, path: str):
        """Экспорт состояния поля для сохранения/анализа."""
        config_dict = asdict(self.config)
        config_dict["consolidation_mode"] = config_dict["consolidation_mode"].value
        data = {
            "config": config_dict,
            "nodes": [node.to_dict() for node in self.field.nodes.values()],
            "projection": self.field.projection.tolist(),
            "stats": self.field.stats
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def import_field(cls, path: str, embedder: Callable) -> RTMDKMemory:
        """Импорт ранее сохранённого поля."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        config_data = data["config"]
        if isinstance(config_data.get("consolidation_mode"), str):
            config_data["consolidation_mode"] = ConsolidationMode(config_data["consolidation_mode"])
        config = RTMDKConfig(**config_data)
        memory = cls(config=config, embedder=embedder)
        memory.field = RTMDKField(config, np.array(data["projection"], dtype=np.float32))
        
        for node_data in data["nodes"]:
            node = MemoryNode.from_dict(node_data)
            memory.field.nodes[node.id] = node
            memory.field.node_index.append(node.id)
        
        memory.field.stats = data["stats"]
        return memory

# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def demo_basic():
    """Базовая демонстрация: сохранение, поиск, консолидация."""
    # Простой эмбеддер (для демо: случайные векторы + шум)
    def dummy_embedder(text: str) -> NDArray[np.float32]:
        np.random.seed(hash(text) % 2**32)
        base = np.random.randn(768).astype(np.float32) * 0.1
        # Добавляем семантический сигнал: первые 10 компонент = хэш-сигнатура
        sig = np.array([hash(text + str(i)) % 1000 / 500 for i in range(10)], dtype=np.float32)
        base[:10] = sig
        return base
    
    # Инициализация памяти
    config = RTMDKConfig(
        embedding_dim=768,
        latent_dim=64,
        tension_threshold=0.2,
        decay_rate=0.995,
        top_k=3,
        enable_async=False,
    )
    memory = RTMDKMemory(config=config, embedder=dummy_embedder)
    
    # Сценарий: диалог с постепенным накоплением и противоречием
    interactions = [
        {"input": "Ya lyublyu kofe po utram", "output": "Kofe pomogaet prosnutsya.", "session_id": "user1"},
        {"input": "Kofe vreden dlya serdtsa", "output": "No v umerennykh kolichestvakh on bezopasen.", "session_id": "user1"},
        {"input": "Ya pereshel na chay", "output": "Chay - otlichnaya alternativ.", "session_id": "user1"},
        {"input": "Chto ya pyu po utram?", "output": "", "session_id": "user1"},
    ]
    
    print("=== RTMDK Memory Demo ===\n")
    
    for i, turn in enumerate(interactions, 1):
        print(f"[Step {i}] Input: '{turn['input']}'")
        
        # Сохранение контекста (если есть выход)
        if turn["output"]:
            memory.save_context(turn, turn)
            print(f"  -> Saved to memory (nodes: {len(memory.field.nodes)})")
        
        # Загрузка контекста для ответа
        ctx = memory.load_memory_variables(turn)
        if ctx["rtmdk_context"] != "No relevant memory.":
            lines = ctx['rtmdk_context'].split('\n')
            for line in lines:
                print(f"     {line}")
        
        # Периодическая консолидация
        if i % 2 == 0:
            updated = memory.field.consolidate()
            if updated:
                print(f"  ~ Consolidation: updated nodes {updated}")
        
        print()
    
    # Статистика
    stats = memory.get_stats()
    print("=== Field Statistics ===")
    print(f"Nodes: {stats['active_nodes']}")
    print(f"Queries: {stats['total_queries']}")
    print(f"Consolidations: {stats['consolidations']}")
    print(f"Avg response: {stats['avg_response']:.3f}")
    
    # Экспорт для дальнейшего анализа
    memory.export_field("rtmdk_demo_state.json")
    print(f"\n[OK] Field state exported to rtmdk_demo_state.json")

if __name__ == "__main__":
    demo_basic()