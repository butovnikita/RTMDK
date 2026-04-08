"""debug_realistic_test.py — Test with realistic diverse facts."""
import numpy as np
from embedder_lmstudio import LMStudioEmbedder
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory

emb = LMStudioEmbedder()

# REALISTIC diverse facts with natural language
facts = [
    ("Я пью чёрный кофе без сахара каждое утро в 8 часов", "Что я пью по утрам?", "кофе"),
    ("Мой любимый редактор кода — VS Code с темой Gruvbox", "Какой редактор я использую?", "VS Code"),
    ("В прошлом году я посетил Токио и был в восторге от суши в районе Синдзюку", "Где я был в прошлом году?", "Токио"),
    ("Я бегаю по 5 километров каждое субботнее утро в парке", "Сколько я бегаю?", "5 километров"),
    ("Играю на акустической гитаре уже 10 лет, предпочитаю фингерстайл", "На чём я играю?", "гитаре"),
    ("Python — мой основной язык программирования, особенно нравится FastAPI", "Какой язык программирования я использую?", "Python"),
    ("Сплю в среднем 7 часов, стараюсь ложиться до полуночи", "Сколько я сплю?", "7 часов"),
    ("Обожаю джаз, особенно Майлза Дэвиса и Джона Колтрейна", "Какую музыку я люблю?", "джаз"),
    ("Моя собака — золотистый ретривер по кличке Бакс", "Как зовут мою собаку?", "Бакс"),
    ("Готовлю пасту карбонара по рецепту итальянской бабушки друга", "Что я готовлю?", "карбонара"),
    ("Принимаю витамин D3 2000 ME каждый день после завтрака", "Какие витамины я принимаю?", "витамин D"),
    ("Мой любимый фильм — Интерстеллар Кристофера Нолана", "Какой мой любимый фильм?", "Интерстеллар"),
    ("Работаю удалённо из квартиры с видом на реку", "Откуда я работаю?", "квартиры"),
    ("Коллекционирую виниловые пластинки, уже более 200 штук", "Что я коллекционирую?", "пластинки"),
    ("Изучаю японский язык на уровне N3, занимаюсь каждый день", "Какой язык я изучаю?", "японский"),
    ("Предпочитаю тёмный шоколад с содержанием какао 85%", "Какой шоколад я люблю?", "тёмный"),
    ("Мой любимый сезон — осень, особенно октябрь", "Какое моё любимое время года?", "осень"),
    ("Занимаюсь йогой 3 раза в неделю по утрам", "Чем я занимаюсь 3 раза в неделю?", "йогой"),
    ("Люблю читать научную фантастику, особенно Станислава Лема", "Какие книги я читаю?", "фантастику"),
    ("Мой идеальный завтрак — овсянка с ягодами и мёдом", "Что я ем на завтрак?", "овсянка"),
]

# Use OPTIMIZED config — learn_projection=True but with higher update_freq
# to accommodate IncPCA batch size (n_components <= batch_size for first fit)
config = RTMDKConfig(
    embedding_dim=768, latent_dim=256, top_k=5, min_response=0.005,
    decay_rate=0.999, enable_async=False, bm25_fallback=True,
    use_hnsw=True, learn_projection=True,
    projection_update_freq=300,  # Higher to accommodate IncPCA: n_components <= batch
)
mem = RTMDKMemory(config=config, embedder=emb)

# Store facts
for fact, query, kw in facts:
    mem.save_context({"input": fact, "session_id": "real"}, {"output": fact})
    mem.save_context({"input": query, "session_id": "real"}, {"output": fact})
    mem.save_context({"input": kw, "session_id": "real"}, {"output": fact})

print(f"Nodes: {len(mem.field.nodes)}")

# Test recall
n_correct = 0
for fact, query, kw in facts:
    ctx = mem.load_memory_variables({"input": query, "session_id": "real"})
    c = ctx["rtmdk_context"].lower()
    found = kw.lower() in c
    if not found:
        print(f"  MISS: kw='{kw}' query='{query[:40]}'")
        print(f"    context: {ctx['rtmdk_context'][:120]}")
    else:
        n_correct += 1

recall = n_correct / len(facts)
print(f"\nRecall: {n_correct}/{len(facts)} = {recall:.2%}")
