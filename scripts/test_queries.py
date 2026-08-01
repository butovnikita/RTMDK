import urllib.request
import json

queries = [
    "привет",
    "кто ты",
    "начало разговора",
    "первый коммит",
    "RTMDK project",
]

for q in queries:
    req = urllib.request.Request("http://127.0.0.1:8081/v1/memory/query")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer rtmdk-local")
    req.data = json.dumps({"query": q}).encode()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
            context = data.get("context", "")
            lines = [l for l in context.split("\n") if l.strip()][:3]
            print(f"=== Query: {q} ===")
            for line in lines:
                print(line[:300])
            print()
    except Exception as e:
        print(f'Query "{q}" failed: {e}')
