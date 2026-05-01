import json, os

path = os.path.expanduser('~/.rtmdk/memory.json')
d = json.load(open(path, encoding='utf-8'))
nodes = d.get('nodes', [])
print(f'Before: {len(nodes)} nodes')

# Filter to only RTMDK-related nodes
kw = ['rtmdk','audit','commit','баг','fix','конфигурац','архитектур',
      'сравнени','интеграц','производител','безопасн','надежн','математик',
      'механизм','памят','long-term','rag','sillytavern','system prompt',
      'долгосрочн','резонанс','topolog','hello','hi','unified config']

keep = []
for n in nodes:
    c = n.get('content', {})
    inp = c.get('input_text', '').lower()
    txt = c.get('text', '').lower()
    check = inp if inp else txt
    if 'системная инструкция' in check or 'assistant response' in check:
        continue
    if 'summary' in check and ('nikita' in check or 'morgiana' in check):
        continue
    if any(k in check for k in kw):
        keep.append(n)

print(f'After filter: {len(keep)} nodes')

# Fix nodes without output_text
for n in keep:
    c = n.get('content', {})
    if c.get('version') == '2.0' and c.get('input_text') and not c.get('output_text'):
        c['output_text'] = c['input_text']

# Set context_format
d['config']['context_format'] = 'attention'
d['nodes'] = keep

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)

v = json.load(open(path, encoding='utf-8'))
print(f'Saved: {len(v.get("nodes",[]))} nodes, context_format={v.get("config",{}).get("context_format")}')
for i, n in enumerate(v.get('nodes', [])):
    c = n.get('content', {})
    t = c.get('input_text', '') or c.get('text', '')
    print(f'  [{i+1}] {t[:80]}')
